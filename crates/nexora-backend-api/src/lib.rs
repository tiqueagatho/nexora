use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;
use nexora_core::chat_model::*;
use nexora_core::error::ModelError;
use nexora_core::message::{Message, Role};
use nexora_core::tool::ToolSchema;
use serde::{Deserialize, Serialize};

/// Backend LLM para la API de OpenAI (y compatibles: Azure, Groq, Together, etc.)
pub struct OpenAIBackend {
    client: reqwest::Client,
    api_key: String,
    base_url: String,
    model: String,
}

// ── Request/Response types ────────────────────────────────

#[derive(Debug, Serialize)]
struct OpenAIRequest {
    model: String,
    messages: Vec<OpenAIMessage>,
    #[serde(skip_serializing_if = "Vec::is_empty")]
    tools: Vec<OpenAITool>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    stop: Option<Vec<String>>,
    stream: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIMessage {
    role: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    content: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_calls: Option<Vec<OpenAIToolCall>>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tool_call_id: Option<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAITool {
    #[serde(rename = "type")]
    tool_type: String,
    function: OpenAIFunction,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunction {
    name: String,
    description: String,
    parameters: serde_json::Value,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIToolCall {
    id: String,
    #[serde(rename = "type")]
    call_type: String,
    function: OpenAIFunctionCall,
}

#[derive(Debug, Serialize, Deserialize)]
struct OpenAIFunctionCall {
    name: String,
    arguments: String,
}

#[derive(Debug, Deserialize)]
struct OpenAIResponse {
    choices: Vec<OpenAIChoice>,
    usage: Option<OpenAIUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAIChoice {
    message: OpenAIMessage,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIUsage {
    prompt_tokens: u32,
    completion_tokens: u32,
    total_tokens: u32,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChunk {
    choices: Vec<OpenAIStreamChoice>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamChoice {
    delta: Option<OpenAIStreamDelta>,
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAIStreamDelta {
    #[serde(default)]
    content: Option<String>,
    #[serde(default)]
    #[allow(dead_code)]
    tool_calls: Option<Vec<OpenAIToolCall>>,
}

impl OpenAIBackend {
    pub fn new(api_key: &str, model: &str) -> Self {
        Self::with_base_url(api_key, model, "https://api.openai.com/v1")
    }

    pub fn with_base_url(api_key: &str, model: &str, base_url: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .build()
            .expect("failed to create HTTP client");

        Self {
            client,
            api_key: api_key.to_string(),
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    fn convert_messages(messages: &[Message]) -> Vec<OpenAIMessage> {
        messages
            .iter()
            .map(|m| {
                let role = match m.role {
                    Role::System => "system",
                    Role::User => "user",
                    Role::Assistant => "assistant",
                    Role::Tool => "tool",
                }
                .to_string();

                OpenAIMessage {
                    role,
                    content: m.content.as_text().map(|s| s.to_string()),
                    tool_calls: None,
                    tool_call_id: m.tool_call_id.clone(),
                }
            })
            .collect()
    }

    fn convert_tools(tools: &[ToolSchema]) -> Vec<OpenAITool> {
        tools
            .iter()
            .map(|t| OpenAITool {
                tool_type: "function".to_string(),
                function: OpenAIFunction {
                    name: t.name.clone(),
                    description: t.description.clone(),
                    parameters: t.input_schema.clone(),
                },
            })
            .collect()
    }

    fn convert_response(resp: OpenAIResponse) -> Result<ChatResponse, ModelError> {
        let choice = resp
            .choices
            .into_iter()
            .next()
            .ok_or_else(|| ModelError::Internal("no choices in response".into()))?;

        let tool_calls: Vec<ToolCall> = choice
            .message
            .tool_calls
            .unwrap_or_default()
            .into_iter()
            .map(|tc| {
                let args: serde_json::Value = serde_json::from_str(&tc.function.arguments)
                    .unwrap_or(serde_json::Value::Object(serde_json::Map::new()));
                ToolCall {
                    id: tc.id,
                    name: tc.function.name,
                    arguments: args,
                }
            })
            .collect();

        let finish_reason = match choice.finish_reason.as_deref() {
            Some("stop") => FinishReason::Stop,
            Some("length") => FinishReason::Length,
            Some("tool_calls") => FinishReason::ToolCalls,
            Some("content_filter") => FinishReason::ContentFilter,
            _ => FinishReason::Stop,
        };

        let content = choice.message.content.unwrap_or_default();

        Ok(ChatResponse {
            message: Message::assistant(content),
            tool_calls,
            usage: resp
                .usage
                .map(|u| Usage {
                    prompt_tokens: u.prompt_tokens,
                    completion_tokens: u.completion_tokens,
                    total_tokens: u.total_tokens,
                })
                .unwrap_or_default(),
            finish_reason,
        })
    }
}

#[async_trait]
impl ChatModel for OpenAIBackend {
    fn name(&self) -> &str {
        &self.model
    }

    async fn invoke(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        params: &GenerateParams,
    ) -> Result<ChatResponse, ModelError> {
        let body = OpenAIRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            tools: Self::convert_tools(tools),
            temperature: params.temperature,
            top_p: params.top_p,
            max_tokens: params.max_tokens,
            stop: if params.stop.is_empty() {
                None
            } else {
                Some(params.stop.clone())
            },
            stream: false,
        };

        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(ModelError::Http)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ModelError::Internal(format!("HTTP {}: {}", status, text)));
        }

        let openai_resp: OpenAIResponse = resp
            .json()
            .await
            .map_err(|e| ModelError::Serialization(e.to_string()))?;

        Self::convert_response(openai_resp)
    }

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        params: &GenerateParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, ModelError>> + Send>>, ModelError>
    {
        let body = OpenAIRequest {
            model: self.model.clone(),
            messages: Self::convert_messages(messages),
            tools: Self::convert_tools(tools),
            temperature: params.temperature,
            top_p: params.top_p,
            max_tokens: params.max_tokens,
            stop: if params.stop.is_empty() {
                None
            } else {
                Some(params.stop.clone())
            },
            stream: true,
        };

        let resp = self
            .client
            .post(format!("{}/chat/completions", self.base_url))
            .header("Authorization", format!("Bearer {}", self.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(ModelError::Http)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ModelError::Internal(format!("HTTP {}: {}", status, text)));
        }

        let byte_stream = resp.bytes_stream();
        let stream = futures::stream::unfold(
            (byte_stream, String::new()),
            |(mut byte_stream, mut buffer)| async move {
                use futures::TryStreamExt;

                loop {
                    // Parse SSE: look for "data: " lines
                    if let Some(line_end) = buffer.find('\n') {
                        let line = buffer[..line_end].trim().to_string();
                        buffer = buffer[line_end + 1..].to_string();

                        if line == "data: [DONE]" {
                            return None;
                        }

                        if let Some(data) = line.strip_prefix("data: ") {
                            match serde_json::from_str::<OpenAIStreamChunk>(data) {
                                Ok(chunk) => {
                                    let delta =
                                        chunk.choices.first().and_then(|c| c.delta.as_ref());

                                    let text =
                                        delta.and_then(|d| d.content.clone()).unwrap_or_default();

                                    let done = chunk
                                        .choices
                                        .first()
                                        .map(|c| c.finish_reason.is_some())
                                        .unwrap_or(false);

                                    return Some((
                                        Ok(StreamChunk {
                                            text,
                                            tool_calls: vec![],
                                            done,
                                        }),
                                        (byte_stream, buffer),
                                    ));
                                }
                                Err(_) => continue,
                            }
                        }
                        continue;
                    }

                    // Read more bytes
                    match byte_stream.try_next().await {
                        Ok(Some(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Ok(None) => return None,
                        Err(e) => {
                            return Some((Err(ModelError::Http(e)), (byte_stream, buffer)));
                        }
                    }
                }
            },
        );

        Ok(Box::pin(stream))
    }

    fn capabilities(&self) -> ModelCapabilities {
        ModelCapabilities {
            supports_tools: true,
            supports_streaming: true,
            supports_images: true,
            max_context_length: 128_000,
            backend_type: BackendType::CloudApi,
        }
    }
}
