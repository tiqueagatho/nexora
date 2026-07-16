use std::pin::Pin;
use std::time::Duration;

use async_trait::async_trait;
use futures::Stream;
use nexora_core::chat_model::*;
use nexora_core::error::ModelError;
use nexora_core::message::Message;
use nexora_core::tool::ToolSchema;
use serde::{Deserialize, Serialize};

/// Backend LLM que se conecta a una instancia de Ollama.
///
/// # Ejemplo
///
/// ```ignore
/// let backend = OllamaBackend::new("http://localhost:11434", "llama3.2");
/// let response = backend.invoke(&messages, &[], &GenerateParams::default()).await?;
/// ```
pub struct OllamaBackend {
    client: reqwest::Client,
    base_url: String,
    model: String,
}

#[derive(Debug, Serialize)]
struct OllamaRequest {
    model: String,
    messages: Vec<Message>,
    #[serde(skip_serializing_if = "Option::is_none")]
    tools: Option<Vec<ToolSchema>>,
    stream: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    options: Option<OllamaOptions>,
}

#[derive(Debug, Serialize)]
struct OllamaOptions {
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    num_predict: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    top_p: Option<f32>,
}

#[derive(Debug, Deserialize)]
struct OllamaResponse {
    message: Option<OllamaMessage>,
    #[serde(default)]
    tool_calls: Vec<OllamaToolCall>,
    done: bool,
    #[serde(default)]
    eval_count: Option<u32>,
    #[serde(default)]
    prompt_eval_count: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct OllamaMessage {
    role: String,
    content: String,
}

#[derive(Debug, Deserialize)]
struct OllamaToolCall {
    function: OllamaToolCallFunction,
}

#[derive(Debug, Deserialize)]
struct OllamaToolCallFunction {
    name: String,
    arguments: serde_json::Value,
}

#[derive(Debug, Deserialize)]
struct OllamaStreamChunk {
    message: Option<OllamaMessage>,
    done: bool,
}

impl OllamaBackend {
    /// Crear un nuevo backend Ollama.
    pub fn new(base_url: &str, model: &str) -> Self {
        let client = reqwest::Client::builder()
            .timeout(Duration::from_secs(300))
            .build()
            .expect("failed to create HTTP client");

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    /// Crear con configuración personalizada.
    pub fn with_config(base_url: &str, model: &str, timeout: Duration) -> Self {
        let client = reqwest::Client::builder()
            .timeout(timeout)
            .build()
            .expect("failed to create HTTP client");

        Self {
            client,
            base_url: base_url.trim_end_matches('/').to_string(),
            model: model.to_string(),
        }
    }

    fn build_request(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        params: &GenerateParams,
        stream: bool,
    ) -> OllamaRequest {
        OllamaRequest {
            model: self.model.clone(),
            messages: messages.to_vec(),
            tools: if tools.is_empty() {
                None
            } else {
                Some(tools.to_vec())
            },
            stream,
            options: Some(OllamaOptions {
                temperature: params.temperature,
                num_predict: params.max_tokens,
                top_p: params.top_p,
            }),
        }
    }

    fn convert_response(resp: OllamaResponse) -> ChatResponse {
        let message = resp
            .message
            .map(|m| {
                let role = match m.role.as_str() {
                    "assistant" => nexora_core::message::Role::Assistant,
                    "tool" => nexora_core::message::Role::Tool,
                    _ => nexora_core::message::Role::Assistant,
                };
                Message {
                    role,
                    content: m.content.into(),
                    name: None,
                    tool_call_id: None,
                }
            })
            .unwrap_or_else(|| Message::assistant(""));

        let tool_calls: Vec<ToolCall> = resp
            .tool_calls
            .into_iter()
            .enumerate()
            .map(|(i, tc)| ToolCall {
                id: format!("call_{}", i),
                name: tc.function.name,
                arguments: tc.function.arguments,
            })
            .collect();

        let finish_reason = if !tool_calls.is_empty() {
            FinishReason::ToolCalls
        } else if resp.done {
            FinishReason::Stop
        } else {
            FinishReason::Length
        };

        ChatResponse {
            message,
            tool_calls,
            usage: Usage {
                prompt_tokens: resp.prompt_eval_count.unwrap_or(0),
                completion_tokens: resp.eval_count.unwrap_or(0),
                total_tokens: resp.prompt_eval_count.unwrap_or(0) + resp.eval_count.unwrap_or(0),
            },
            finish_reason,
        }
    }
}

#[async_trait]
impl ChatModel for OllamaBackend {
    fn name(&self) -> &str {
        &self.model
    }

    async fn invoke(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        params: &GenerateParams,
    ) -> Result<ChatResponse, ModelError> {
        let body = self.build_request(messages, tools, params, false);

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
            .json(&body)
            .send()
            .await
            .map_err(ModelError::Http)?;

        if !resp.status().is_success() {
            let status = resp.status();
            let text = resp.text().await.unwrap_or_default();
            return Err(ModelError::Internal(format!("HTTP {}: {}", status, text)));
        }

        let ollama_resp: OllamaResponse = resp
            .json()
            .await
            .map_err(|e| ModelError::Serialization(e.to_string()))?;

        Ok(Self::convert_response(ollama_resp))
    }

    async fn stream(
        &self,
        messages: &[Message],
        tools: &[ToolSchema],
        params: &GenerateParams,
    ) -> Result<Pin<Box<dyn Stream<Item = Result<StreamChunk, ModelError>> + Send>>, ModelError>
    {
        let body = self.build_request(messages, tools, params, true);

        let resp = self
            .client
            .post(format!("{}/api/chat", self.base_url))
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
                    // Try to parse a complete JSON line from the buffer
                    if let Some(newline_pos) = buffer.find('\n') {
                        let line = buffer[..newline_pos].to_string();
                        buffer = buffer[newline_pos + 1..].to_string();

                        if line.trim().is_empty() {
                            continue;
                        }

                        match serde_json::from_str::<OllamaStreamChunk>(&line) {
                            Ok(chunk) => {
                                let text = chunk.message.map(|m| m.content).unwrap_or_default();
                                let done = chunk.done;
                                let result = Ok(StreamChunk {
                                    text,
                                    tool_calls: vec![],
                                    done,
                                });
                                return Some((result, (byte_stream, buffer)));
                            }
                            Err(e) => {
                                return Some((
                                    Err(ModelError::Serialization(e.to_string())),
                                    (byte_stream, buffer),
                                ));
                            }
                        }
                    }

                    // Read more bytes
                    match byte_stream.try_next().await {
                        Ok(Some(bytes)) => {
                            buffer.push_str(&String::from_utf8_lossy(&bytes));
                        }
                        Ok(None) => {
                            // Stream ended
                            if !buffer.is_empty() {
                                let text = std::mem::take(&mut buffer);
                                return Some((
                                    Ok(StreamChunk {
                                        text,
                                        tool_calls: vec![],
                                        done: true,
                                    }),
                                    (byte_stream, buffer),
                                ));
                            }
                            return None;
                        }
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
            supports_images: false,
            max_context_length: 32768,
            backend_type: BackendType::LocalHttp,
        }
    }
}
