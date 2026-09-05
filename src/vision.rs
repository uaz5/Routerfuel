// =============================================================================
// src/vision.rs  — RouterFuel v0.6
//
// Vision / Multimodal Support
//
// FIX (this revision): select_vision_model now takes the same `reachable`
// provider filter route_engine.rs's select_reachable/select_for_task added
// — see route_engine.rs's top-of-file comment for why. Image-carrying
// "auto"/"task:" requests were just as exposed to picking an unreachable
// model as the non-vision path was.
// =============================================================================

use serde::{Deserialize, Serialize};

// =============================================================================
// Extended message content types
// =============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MultimodalMessage {
    pub role:    String,
    pub content: MessageContent,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum MessageContent {
    Text(String),
    Parts(Vec<ContentPart>),
}

impl MessageContent {
    pub fn as_text(&self) -> String {
        match self {
            MessageContent::Text(s) => s.clone(),
            MessageContent::Parts(parts) => parts
                .iter()
                .filter_map(|p| match &p.kind {
                    PartKind::Text { text } => Some(text.as_str()),
                    _ => None,
                })
                .collect::<Vec<_>>()
                .join(" "),
        }
    }

    pub fn has_image(&self) -> bool {
        match self {
            MessageContent::Text(_) => false,
            MessageContent::Parts(parts) => parts.iter().any(|p| !matches!(p.kind, PartKind::Text { .. })),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContentPart {
    #[serde(rename = "type")]
    pub part_type: String,
    #[serde(flatten)]
    pub kind: PartKind,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum PartKind {
    Text {
        text: String,
    },
    ImageUrl {
        image_url: ImageUrl,
    },
    ImageBase64 {
        image_data: ImageBase64,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageUrl {
    pub url:    String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub detail: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageBase64 {
    pub media_type: String,
    pub data:       String,
}

// =============================================================================
// Format conversion: OpenAI-compatible ↔ Anthropic ↔ Gemini
// =============================================================================

pub fn to_anthropic_content(msg: &MultimodalMessage) -> serde_json::Value {
    match &msg.content {
        MessageContent::Text(t) => serde_json::json!({
            "role": msg.role,
            "content": t
        }),
        MessageContent::Parts(parts) => {
            let content: Vec<serde_json::Value> = parts.iter().map(|p| {
                match &p.kind {
                    PartKind::Text { text } => serde_json::json!({
                        "type": "text",
                        "text": text
                    }),
                    PartKind::ImageUrl { image_url } => serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "url",
                            "url": image_url.url
                        }
                    }),
                    PartKind::ImageBase64 { image_data } => serde_json::json!({
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_data.media_type,
                            "data": image_data.data
                        }
                    }),
                }
            }).collect();

            serde_json::json!({ "role": msg.role, "content": content })
        }
    }
}

pub fn to_gemini_content(msg: &MultimodalMessage) -> serde_json::Value {
    let role = if msg.role == "assistant" { "model" } else { "user" };

    match &msg.content {
        MessageContent::Text(t) => serde_json::json!({
            "role": role,
            "parts": [{ "text": t }]
        }),
        MessageContent::Parts(parts) => {
            let gemini_parts: Vec<serde_json::Value> = parts.iter().map(|p| {
                match &p.kind {
                    PartKind::Text { text } => serde_json::json!({ "text": text }),
                    PartKind::ImageUrl { image_url } => serde_json::json!({
                        "fileData": {
                            "mimeType": "image/jpeg",
                            "fileUri": image_url.url
                        }
                    }),
                    PartKind::ImageBase64 { image_data } => serde_json::json!({
                        "inlineData": {
                            "mimeType": image_data.media_type,
                            "data":     image_data.data
                        }
                    }),
                }
            }).collect();

            serde_json::json!({ "role": role, "parts": gemini_parts })
        }
    }
}

pub fn to_openai_compatible_content(msg: &MultimodalMessage) -> serde_json::Value {
    match &msg.content {
        MessageContent::Text(t) => serde_json::json!({
            "role": msg.role,
            "content": t
        }),
        MessageContent::Parts(parts) => {
            let content: Vec<serde_json::Value> = parts.iter().map(|p| {
                match &p.kind {
                    PartKind::Text { text } => serde_json::json!({
                        "type": "text",
                        "text": text
                    }),
                    PartKind::ImageUrl { image_url } => serde_json::json!({
                        "type": "image_url",
                        "image_url": {
                            "url": image_url.url,
                            "detail": image_url.detail.clone().unwrap_or_else(|| "auto".into())
                        }
                    }),
                    PartKind::ImageBase64 { image_data } => serde_json::json!({
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:{};base64,{}", image_data.media_type, image_data.data)
                        }
                    }),
                }
            }).collect();

            serde_json::json!({ "role": msg.role, "content": content })
        }
    }
}

// =============================================================================
// Vision-aware routing helper
// =============================================================================

use crate::connectors::Provider;
use crate::route_engine::{RouteEngine, RoutingPriority};
use std::collections::HashSet;

/// Select the best vision-capable model. `reachable`: same meaning as
/// route_engine.rs's select_reachable — `None` for no filtering, `Some(set)`
/// to restrict candidates to providers the calling client can actually
/// reach given their supplied BYOK keys.
pub fn select_vision_model(
    engine:       &RouteEngine,
    input_tokens: u32,
    priority:     RoutingPriority,
    reachable:    Option<&HashSet<Provider>>,
) -> anyhow::Result<crate::route_engine::RoutingDecision> {
    let decision = engine.select_reachable(input_tokens, 1024, priority, reachable)?;

    if decision.model.supports_vision {
        return Ok(decision);
    }

    // Fallback: try quality priority (flagship models are usually vision-capable)
    let fallback = engine.select_reachable(input_tokens, 1024, RoutingPriority::Quality, reachable)?;
    if fallback.model.supports_vision {
        return Ok(fallback);
    }

    // Last resort: pick the best-scoring vision-capable model directly,
    // ignoring anything without the flag — still respecting `reachable`.
    let vision_models = engine.list_vision_capable();
    vision_models
        .into_iter()
        .filter(|m| {
            input_tokens < m.context_window
                && reachable.map_or(true, |r| r.contains(&m.provider))
        })
        .max_by(|a, b| a.quality_score.partial_cmp(&b.quality_score).unwrap())
        .map(|model| crate::route_engine::RoutingDecision {
            reason: format!("{} chosen as best available vision-capable model", model.display_name),
            score: model.quality_score as f64,
            model,
        })
        .ok_or_else(|| anyhow::anyhow!(
            "No vision-capable model available that you have a BYOK key for. Supply a key \
             for a vision-capable provider directly, or an X-OpenRouter-Api-Key as a \
             universal fallback."
        ))
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn text_message_has_no_image() {
        let msg = MultimodalMessage {
            role:    "user".into(),
            content: MessageContent::Text("hello".into()),
        };
        assert!(!msg.content.has_image());
    }

    #[test]
    fn parts_message_detects_image() {
        let msg = MultimodalMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart {
                    part_type: "text".into(),
                    kind: PartKind::Text { text: "What is in this image?".into() },
                },
                ContentPart {
                    part_type: "image_url".into(),
                    kind: PartKind::ImageUrl {
                        image_url: ImageUrl {
                            url:    "https://example.com/food.jpg".into(),
                            detail: Some("high".into()),
                        },
                    },
                },
            ]),
        };
        assert!(msg.content.has_image());
        assert_eq!(msg.content.as_text(), "What is in this image?");
    }

    #[test]
    fn registry_flags_flagship_models_as_vision_capable() {
        let engine = RouteEngine::new();
        assert!(engine.is_vision_capable("claude-opus-4-8"));
        assert!(engine.is_vision_capable("gpt-5.6-sol"));
        assert!(engine.is_vision_capable("gemini-3.1-pro-preview"));
        assert!(!engine.is_vision_capable("deepseek-v4-flash"));
    }

    #[test]
    fn anthropic_conversion_base64() {
        let msg = MultimodalMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart {
                    part_type: "image".into(),
                    kind: PartKind::ImageBase64 {
                        image_data: ImageBase64 {
                            media_type: "image/png".into(),
                            data:       "abc123".into(),
                        },
                    },
                },
            ]),
        };
        let v = to_anthropic_content(&msg);
        let source = &v["content"][0]["source"];
        assert_eq!(source["type"], "base64");
        assert_eq!(source["media_type"], "image/png");
    }

    #[test]
    fn openai_compatible_conversion_base64() {
        let msg = MultimodalMessage {
            role: "user".into(),
            content: MessageContent::Parts(vec![
                ContentPart {
                    part_type: "image_url".into(),
                    kind: PartKind::ImageBase64 {
                        image_data: ImageBase64 {
                            media_type: "image/jpeg".into(),
                            data:       "xyz789".into(),
                        },
                    },
                },
            ]),
        };
        let v = to_openai_compatible_content(&msg);
        let url = v["content"][0]["image_url"]["url"].as_str().unwrap();
        assert!(url.starts_with("data:image/jpeg;base64,"));
    }

    #[test]
    fn select_vision_model_respects_reachable_filter() {
        let engine = RouteEngine::new();
        let mut only_openai = HashSet::new();
        only_openai.insert(Provider::OpenAI);
        let decision = select_vision_model(&engine, 5_000, RoutingPriority::Balanced, Some(&only_openai)).unwrap();
        assert_eq!(decision.model.provider, Provider::OpenAI);
        assert!(decision.model.supports_vision);
    }
}
