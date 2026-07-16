use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Rol de un mensaje en la conversación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "lowercase")]
pub enum Role {
    System,
    User,
    Assistant,
    Tool,
}

/// Contenido de un mensaje: texto plano o partes multimodales.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(untagged)]
pub enum Content {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl Content {
    pub fn text(s: impl Into<String>) -> Self {
        Self::Text(s.into())
    }

    pub fn as_text(&self) -> Option<&str> {
        match self {
            Content::Text(s) => Some(s),
            Content::Parts(parts) => {
                if parts.len() == 1 {
                    match &parts[0] {
                        ContentPart::Text { text } => Some(text),
                        _ => None,
                    }
                } else {
                    None
                }
            }
        }
    }

    pub fn into_text(self) -> Option<String> {
        match self {
            Content::Text(s) => Some(s),
            Content::Parts(mut parts) => {
                if parts.len() == 1 {
                    match parts.remove(0) {
                        ContentPart::Text { text } => Some(text),
                        _ => None,
                    }
                } else {
                    None
                }
            }
        }
    }
}

impl From<String> for Content {
    fn from(s: String) -> Self {
        Self::Text(s)
    }
}

impl From<&str> for Content {
    fn from(s: &str) -> Self {
        Self::Text(s.to_string())
    }
}

/// Parte individual de un contenido multimodal.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type")]
pub enum ContentPart {
    #[serde(rename = "text")]
    Text { text: String },
    #[serde(rename = "image_url")]
    ImageUrl { image_url: ImageUrlDetail },
}

#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ImageUrlDetail {
    pub url: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

/// Mensaje de la conversación.
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Message {
    pub role: Role,
    pub content: Content,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub name: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub tool_call_id: Option<String>,
}

impl Message {
    pub fn system(content: impl Into<Content>) -> Self {
        Self {
            role: Role::System,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    pub fn user(content: impl Into<Content>) -> Self {
        Self {
            role: Role::User,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    pub fn assistant(content: impl Into<Content>) -> Self {
        Self {
            role: Role::Assistant,
            content: content.into(),
            name: None,
            tool_call_id: None,
        }
    }

    pub fn tool(content: impl Into<Content>, tool_call_id: impl Into<String>) -> Self {
        Self {
            role: Role::Tool,
            content: content.into(),
            name: None,
            tool_call_id: Some(tool_call_id.into()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn message_system_creates_correct_role() {
        let msg = Message::system("You are helpful");
        assert_eq!(msg.role, Role::System);
        assert_eq!(msg.content.as_text(), Some("You are helpful"));
        assert!(msg.tool_call_id.is_none());
    }

    #[test]
    fn message_user_from_string() {
        let msg = Message::user("Hello");
        assert_eq!(msg.role, Role::User);
        assert_eq!(msg.content.as_text(), Some("Hello"));
    }

    #[test]
    fn message_assistant_from_string() {
        let msg = Message::assistant("Hi there");
        assert_eq!(msg.role, Role::Assistant);
    }

    #[test]
    fn message_tool_with_id() {
        let msg = Message::tool("result data", "call_123");
        assert_eq!(msg.role, Role::Tool);
        assert_eq!(msg.tool_call_id.as_deref(), Some("call_123"));
        assert_eq!(msg.content.as_text(), Some("result data"));
    }

    #[test]
    fn content_from_str() {
        let c: Content = "hello".into();
        assert_eq!(c.as_text(), Some("hello"));
    }

    #[test]
    fn content_from_string() {
        let c: Content = String::from("world").into();
        assert_eq!(c.as_text(), Some("world"));
    }

    #[test]
    fn content_text_variant() {
        let c = Content::text("test");
        assert_eq!(c.as_text(), Some("test"));
        match c {
            Content::Text(s) => assert_eq!(s, "test"),
            _ => panic!("expected Text variant"),
        }
    }

    #[test]
    fn content_parts_single_text() {
        let c = Content::Parts(vec![ContentPart::Text {
            text: "single".into(),
        }]);
        assert_eq!(c.as_text(), Some("single"));
    }

    #[test]
    fn content_parts_multi_returns_none() {
        let c = Content::Parts(vec![
            ContentPart::Text {
                text: "a".into(),
            },
            ContentPart::Text {
                text: "b".into(),
            },
        ]);
        assert_eq!(c.as_text(), None);
    }

    #[test]
    fn content_into_text() {
        let c = Content::Text("owned".into());
        assert_eq!(c.into_text(), Some("owned".into()));
    }

    #[test]
    fn content_serialize_roundtrip() {
        let msg = Message::user("test message");
        let json = serde_json::to_string(&msg).unwrap();
        let deserialized: Message = serde_json::from_str(&json).unwrap();
        assert_eq!(deserialized.role, Role::User);
        assert_eq!(deserialized.content.as_text(), Some("test message"));
    }

    #[test]
    fn role_serialize_lowercase() {
        assert_eq!(serde_json::to_string(&Role::System).unwrap(), "\"system\"");
        assert_eq!(serde_json::to_string(&Role::User).unwrap(), "\"user\"");
        assert_eq!(
            serde_json::to_string(&Role::Assistant).unwrap(),
            "\"assistant\""
        );
        assert_eq!(serde_json::to_string(&Role::Tool).unwrap(), "\"tool\"");
    }
}
