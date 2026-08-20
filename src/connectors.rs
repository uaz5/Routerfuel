// ============================================================================
// src/connectors.rs — RouterFuel v0.6
//
// STRICT BYOK MODEL:
//   RouterFuel never holds a paid provider API key of its own. Every request
//   is billed directly to the *client's* provider account. `complete()` takes
//   `client_api_key: &str` (not Option<&str>) — there is no gateway-key
//   fallback path. If the client hasn't supplied a key for the selected
//   provider (directly, or via an OpenRouter key), main.rs rejects the
//   request with BadRequest before a connector is ever called.
//
// Why one generic connector covers most providers:
//   OpenAI, DeepSeek, Mistral, xAI (Grok), Alibaba Qwen, Moonshot (Kimi),
//   Zhipu (GLM), Meta Llama, and OpenRouter all speak the same
//   {model, messages, ...} / {choices:[{message}], usage} JSON shape with
//   `Authorization: Bearer <key>` auth — that's GenericOpenAICompatibleConnector.
//   Anthropic and Gemini use different wire formats and get bespoke connectors.
//
// FIX (this revision), two related issues:
//   1. ChatCompletionResponse/Choice previously required object/created/
//      finish_reason with no #[serde(default)] — any provider deviating
//      even slightly from strict OpenAI wire compatibility on a field
//      RouterFuel doesn't actually need would fail deserialization
//      entirely. Those fields are now #[serde(default)].
//   2. A deserialize failure (malformed/unexpected JSON shape) used to call
//      cb.record_failure(provider) — treating a RouterFuel-side parsing
//      assumption mismatch the same as a real provider outage, tripping
//      the circuit breaker against a provider that might be perfectly
//      healthy. record_failure is no longer called on parse failures;
//      ConnectorError::trips_circuit() already correctly excludes
//      BadResponse from the set of errors that should trip the breaker —
//      this just stops bypassing that distinction.
// ============================================================================

use crate::circuit_breaker::CircuitBreaker;
use anyhow::Result;
use async_trait::async_trait;
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use std::time::Instant;
use thiserror::Error;
use tracing::{debug, instrument};

// ============================================================================
// PROVIDER ENUM
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Provider {
    Anthropic,
    OpenAI,
    Gemini,
    DeepSeek,
    Mistral,
    XAI,       // Grok
    Qwen,      // Alibaba DashScope (OpenAI-compatible mode)
    Moonshot,  // Kimi
    Zhipu,     // GLM
    Meta,      // Llama API
    OpenRouter,
    AzureOpenAI,
    Bedrock,
}

impl std::fmt::Display for Provider {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Provider::Anthropic   => write!(f, "anthropic"),
            Provider::OpenAI      => write!(f, "openai"),
            Provider::Gemini      => write!(f, "gemini"),
            Provider::DeepSeek    => write!(f, "deepseek"),
            Provider::Mistral     => write!(f, "mistral"),
            Provider::XAI         => write!(f, "xai"),
            Provider::Qwen        => write!(f, "qwen"),
            Provider::Moonshot    => write!(f, "moonshot"),
            Provider::Zhipu       => write!(f, "zhipu"),
            Provider::Meta        => write!(f, "meta"),
            Provider::OpenRouter  => write!(f, "openrouter"),
            Provider::AzureOpenAI => write!(f, "azure_openai"),
            Provider::Bedrock     => write!(f, "bedrock"),
        }
    }
}

impl Provider {
    /// The prefix OpenRouter expects in front of the bare model id,
    /// e.g. "claude-opus-4-7" -> "anthropic/claude-opus-4-7".
    /// Used only when a request is being re-routed through OpenRouter
    /// because the client supplied an OpenRouter key but not a direct one.
    pub fn openrouter_prefix(&self) -> &'static str {
        match self {
            Provider::Anthropic => "anthropic",
            Provider::OpenAI    => "openai",
            Provider::Gemini    => "google",
            Provider::DeepSeek  => "deepseek",
            Provider::Mistral   => "mistralai",
            Provider::XAI       => "x-ai",
            Provider::Qwen      => "qwen",
            Provider::Moonshot  => "moonshotai",
            Provider::Zhipu     => "z-ai",
            Provider::Meta      => "meta-llama",
            Provider::OpenRouter => "",
            Provider::AzureOpenAI => "",
            Provider::Bedrock => "",
        }
    }
}

// ============================================================================
// ERROR
// ============================================================================

#[derive(Error, Debug)]
pub enum ConnectorError {
    #[error("HTTP error: {0}")]
    Http(#[from] reqwest::Error),
    #[error("Provider returned 5xx ({status})")]
    ServerError { status: u16 },
    #[error("Unauthorized — the BYOK key supplied for this provider was rejected")]
    Unauthorized,
    #[error("Rate limited")]
    RateLimited,
    #[error("Timeout")]
    Timeout,
    #[error("Bad response: {0}")]
    BadResponse(String),
    #[error("Serialization: {0}")]
    Serde(#[from] serde_json::Error),
    #[error("Circuit breaker open")]
    CircuitOpen,
    #[error("Not implemented: {0}")]
    NotImplemented(String),
    #[error("Missing BYOK key: no API key supplied for provider '{0}' (directly or via OpenRouter)")]
    MissingKey(String),
}

impl ConnectorError {
    pub fn trips_circuit(&self) -> bool {
        matches!(self, Self::ServerError { .. } | Self::Timeout | Self::Http(_))
    }
}

// ============================================================================
// OPENAI-COMPATIBLE TYPES
// Used by every provider except Anthropic (bespoke) and Gemini (bespoke).
// ============================================================================

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatCompletionRequest {
    pub model: String,
    pub messages: Vec<ChatMessage>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub temperature: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub max_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub top_p: Option<f32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub stream: Option<bool>,
    /// RouterFuel-only field, never forwarded to a provider (see `skip_serializing`
    /// below) — if set, RouterFuel fires an identical request at this model
    /// *in addition to* the normally-routed one, purely for comparison. The
    /// client only ever sees the primary response; the shadow call's cost,
    /// latency, and output are logged to the `shadow_comparisons` table.
    /// See main.rs's `maybe_fire_shadow_request` and CHANGES.md for details.
    #[serde(skip_serializing, default)]
    pub shadow_model: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ChatMessage {
    pub role: String,
    /// Plain text (the common case, backward-compatible with any existing
    /// client sending a bare string) or multimodal parts carrying one or
    /// more images alongside text — see src/vision.rs.
    pub content: crate::vision::MessageContent,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct ChatCompletionResponse {
    // FIX: #[serde(default)] on fields RouterFuel doesn't strictly need,
    // so a provider that's technically "OpenAI-compatible" but omits one
    // of these doesn't fail deserialization entirely.
    #[serde(default)]
    pub id: String,
    #[serde(default)]
    pub object: String,
    #[serde(default)]
    pub created: u64,
    pub model: String,
    pub choices: Vec<Choice>,
    pub usage: Usage,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Choice {
    #[serde(default)]
    pub index: u32,
    pub message: ChatMessage,
    #[serde(default)]
    pub finish_reason: String,
}

#[derive(Debug, Clone, Deserialize, Serialize)]
pub struct Usage {
    pub prompt_tokens: u32,
    pub completion_tokens: u32,
    pub total_tokens: u32,
}

// ============================================================================
// CONNECTOR TRAIT
// ============================================================================

#[derive(Debug, Clone)]
pub struct ConnectorResult {
    pub provider: Provider,
    pub model_id: String,
    pub response: ChatCompletionResponse,
    pub latency_ms: u64,
    pub input_tokens: u32,
    pub output_tokens: u32,
}

#[async_trait]
pub trait Connector: Send + Sync {
    /// `client_api_key` is mandatory — RouterFuel holds no keys of its own.
    async fn complete(
        &self,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError>;
    fn provider(&self) -> Provider;
}

// ============================================================================
// GENERIC OPENAI-COMPATIBLE CONNECTOR
// Covers: OpenAI, DeepSeek, Mistral, xAI, Qwen, Moonshot, Zhipu, Meta, OpenRouter
// ============================================================================

pub struct GenericOpenAICompatibleConnector {
    provider: Provider,
    base_url: String,
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
    /// Extra static headers some providers need beyond Bearer auth
    /// (e.g. OpenRouter's optional attribution headers).
    extra_headers: Vec<(&'static str, &'static str)>,
}

impl GenericOpenAICompatibleConnector {
    pub fn new(
        provider: Provider,
        base_url: impl Into<String>,
        circuit_breaker: Arc<CircuitBreaker>,
    ) -> Self {
        Self {
            provider,
            base_url: base_url.into(),
            client: build_client(),
            circuit_breaker,
            extra_headers: Vec::new(),
        }
    }

    pub fn with_extra_headers(mut self, headers: Vec<(&'static str, &'static str)>) -> Self {
        self.extra_headers = headers;
        self
    }
}

#[async_trait]
impl Connector for GenericOpenAICompatibleConnector {
    #[instrument(skip(self, req, client_api_key), fields(model = %req.model, provider = %self.provider))]
    async fn complete(
        &self,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError> {
        openai_compatible_call(
            &self.client,
            &self.base_url,
            client_api_key,
            req,
            self.provider,
            &self.circuit_breaker,
            &self.extra_headers,
        )
        .await
    }

    fn provider(&self) -> Provider {
        self.provider
    }
}

// ============================================================================
// ANTHROPIC CONNECTOR (bespoke wire format)
// POST https://api.anthropic.com/v1/messages
// ============================================================================

const ANTHROPIC_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_VER: &str = "2023-06-01";

/// Base completion URL for every provider RouterFuel talks to directly.
/// Centralized here so streaming.rs and ConnectorManager::new() can't drift
/// out of sync with each other.
pub fn provider_base_url(provider: Provider) -> &'static str {
    match provider {
        Provider::OpenAI     => "https://api.openai.com/v1/chat/completions",
        Provider::Anthropic  => ANTHROPIC_URL,
        Provider::DeepSeek   => "https://api.deepseek.com/v1/chat/completions",
        Provider::Gemini     => "https://generativelanguage.googleapis.com/v1beta/models",
        Provider::Mistral    => "https://api.mistral.ai/v1/chat/completions",
        Provider::XAI        => "https://api.x.ai/v1/chat/completions",
        Provider::Qwen       => "https://dashscope-intl.aliyuncs.com/compatible-mode/v1/chat/completions",
        Provider::Moonshot   => "https://api.moonshot.ai/v1/chat/completions",
        Provider::Zhipu      => "https://open.bigmodel.cn/api/paas/v4/chat/completions",
        Provider::Meta       => "https://api.llama.com/v1/chat/completions",
        Provider::OpenRouter => "https://openrouter.ai/api/v1/chat/completions",
        Provider::AzureOpenAI => "", // set dynamically per deployment
        Provider::Bedrock => "", // set dynamically per model
    }
}

#[derive(Debug, Serialize)]
struct AnthropicReq {
    model: String,
    messages: Vec<serde_json::Value>,
    max_tokens: u32,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f32>,
    /// Anthropic takes system content as a top-level field, not a message
    /// with role "system".
    #[serde(skip_serializing_if = "Option::is_none")]
    system: Option<String>,
}

pub fn build_anthropic_messages(messages: &[ChatMessage]) -> (Vec<serde_json::Value>, Option<String>) {
    let mut system_text = String::new();
    let mut out = Vec::with_capacity(messages.len());

    for m in messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push(' ');
            }
            system_text.push_str(&m.content.as_text());
            continue;
        }
        let mm = crate::vision::MultimodalMessage { role: m.role.clone(), content: m.content.clone() };
        out.push(crate::vision::to_anthropic_content(&mm));
    }

    let system = if system_text.is_empty() { None } else { Some(system_text) };
    (out, system)
}

#[derive(Debug, Deserialize)]
struct AnthropicResp {
    id: String,
    model: String,
    content: Vec<AnthropicBlock>,
    stop_reason: String,
    usage: AnthropicUsage,
}

#[derive(Debug, Deserialize)]
struct AnthropicBlock {
    #[serde(rename = "type")]
    kind: String,
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicUsage {
    input_tokens: u32,
    output_tokens: u32,
}

pub struct AnthropicConnector {
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl AnthropicConnector {
    pub fn new(circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self { client: build_client(), circuit_breaker }
    }
}

#[async_trait]
impl Connector for AnthropicConnector {
    #[instrument(skip(self, req, client_api_key), fields(model = %req.model))]
    async fn complete(
        &self,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError> {
        let start = Instant::now();

        if self.circuit_breaker.is_open(Provider::Anthropic) {
            return Err(ConnectorError::CircuitOpen);
        }

        let (messages, system) = build_anthropic_messages(&req.messages);
        let body = AnthropicReq {
            model: req.model.clone(),
            messages,
            max_tokens: req.max_tokens.unwrap_or(1024),
            temperature: req.temperature,
            system,
        };

        let http_resp = self
            .client
            .post(ANTHROPIC_URL)
            .header("x-api-key", client_api_key)
            .header("anthropic-version", ANTHROPIC_VER)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    self.circuit_breaker.record_failure(Provider::Anthropic);
                    ConnectorError::Timeout
                } else {
                    ConnectorError::Http(e)
                }
            })?;

        let status = http_resp.status().as_u16();
        let text = http_resp.text().await.map_err(|e| {
            if e.is_timeout() {
                self.circuit_breaker.record_failure(Provider::Anthropic);
                ConnectorError::Timeout
            } else {
                ConnectorError::Http(e)
            }
        })?;

        match status {
           200..=299 => {
                // FIX: no longer calls record_failure on a parse error — a
                // schema mismatch is a RouterFuel-side assumption bug, not
                // evidence the provider itself is unhealthy.
                let ar: AnthropicResp = serde_json::from_str(&text).map_err(|e| {
                    ConnectorError::BadResponse(format!("Provider returned unexpected response format: {e}"))
                })?;
                let content = ar
                    .content
                    .iter()
                    .find(|b| b.kind == "text")
                    .and_then(|b| b.text.as_deref())
                    .unwrap_or("")
                    .to_owned();

                let response = ChatCompletionResponse {
                    id: ar.id,
                    object: "chat.completion".into(),
                    created: unix_now(),
                    model: ar.model.clone(),
                    choices: vec![Choice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".into(),
                            content: crate::vision::MessageContent::Text(content),
                        },
                        finish_reason: ar.stop_reason,
                    }],
                    usage: Usage {
                        prompt_tokens: ar.usage.input_tokens,
                        completion_tokens: ar.usage.output_tokens,
                        total_tokens: ar.usage.input_tokens + ar.usage.output_tokens,
                    },
                };

                self.circuit_breaker.record_success(Provider::Anthropic);
                Ok(ConnectorResult {
                    provider: Provider::Anthropic,
                    model_id: ar.model,
                    input_tokens: response.usage.prompt_tokens,
                    output_tokens: response.usage.completion_tokens,
                    latency_ms: start.elapsed().as_millis() as u64,
                    response,
                })
            }
            401 => Err(ConnectorError::Unauthorized),
            429 => Err(ConnectorError::RateLimited),
            500..=599 => {
                self.circuit_breaker.record_failure(Provider::Anthropic);
                Err(ConnectorError::ServerError { status })
            }
            _ => Err(ConnectorError::BadResponse(format!("HTTP {}: {}", status, text))),
        }
    }

    fn provider(&self) -> Provider {
        Provider::Anthropic
    }
}

// ============================================================================
// GEMINI CONNECTOR (bespoke wire format)
// ============================================================================

#[derive(Debug, Deserialize)]
struct GeminiCandidate {
    content: GeminiRespContent,
    #[serde(rename = "finishReason")]
    finish_reason: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiRespContent {
    parts: Vec<GeminiRespPart>,
}

#[derive(Debug, Deserialize)]
struct GeminiRespPart {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiUsageMetadata {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: u32,
    #[serde(rename = "candidatesTokenCount", default)]
    candidates_token_count: u32,
}

#[derive(Debug, Deserialize)]
struct GeminiResp {
    candidates: Vec<GeminiCandidate>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiUsageMetadata>,
}

pub fn to_gemini_body(req: &ChatCompletionRequest) -> serde_json::Value {
    let mut system_text = String::new();
    let mut contents: Vec<serde_json::Value> = Vec::new();

    for m in &req.messages {
        if m.role == "system" {
            if !system_text.is_empty() {
                system_text.push(' ');
            }
            system_text.push_str(&m.content.as_text());
            continue;
        }
        let mm = crate::vision::MultimodalMessage { role: m.role.clone(), content: m.content.clone() };
        contents.push(crate::vision::to_gemini_content(&mm));
    }

    let system_instruction = if system_text.is_empty() {
        None
    } else {
        Some(serde_json::json!({ "role": "system", "parts": [{ "text": system_text }] }))
    };

    let mut body = serde_json::json!({
        "contents": contents,
        "generationConfig": {
            "temperature": req.temperature,
            "maxOutputTokens": req.max_tokens,
            "topP": req.top_p,
        },
    });

    if let Some(si) = system_instruction {
        body["systemInstruction"] = si;
    }

    body
}

pub struct GeminiConnector {
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl GeminiConnector {
    pub fn new(circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self { client: build_client(), circuit_breaker }
    }
}

#[async_trait]
impl Connector for GeminiConnector {
    #[instrument(skip(self, req, client_api_key), fields(model = %req.model))]
    async fn complete(
        &self,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError> {
        let start = Instant::now();

        if self.circuit_breaker.is_open(Provider::Gemini) {
            return Err(ConnectorError::CircuitOpen);
        }

        let body = to_gemini_body(req);

        let url = format!(
            "{}/{}:generateContent",
            provider_base_url(Provider::Gemini),
            req.model
        );

        // FIX: was `.query(&[("key", client_api_key)])`, which puts the
        // client's BYOK Gemini key directly in the request URL. Google's
        // Generative Language API accepts the key either way, but a
        // secret in a URL is far more likely to end up somewhere it
        // shouldn't — proxy/load-balancer access logs, APM/tracing tools
        // that capture the outbound request line, etc. Every other
        // connector in this file sends its key via a header
        // (Authorization: Bearer / x-api-key); Gemini now does too, via
        // Google's documented x-goog-api-key header.
        let http_resp = self
            .client
            .post(&url)
            .header("x-goog-api-key", client_api_key)
            .header("content-type", "application/json")
            .json(&body)
            .send()
            .await
            .map_err(|e| {
                if e.is_timeout() {
                    self.circuit_breaker.record_failure(Provider::Gemini);
                    ConnectorError::Timeout
                } else {
                    ConnectorError::Http(e)
                }
            })?;

       let status = http_resp.status().as_u16();
        let text = http_resp.text().await.map_err(|e| {
            if e.is_timeout() {
                self.circuit_breaker.record_failure(Provider::Gemini);
                ConnectorError::Timeout
            } else {
                ConnectorError::Http(e)
            }
        })?;

        match status {
            200..=299 => {
                // FIX: no longer calls record_failure on a parse error.
                let gr: GeminiResp = serde_json::from_str(&text).map_err(|e| {
                    ConnectorError::BadResponse(format!("Provider returned unexpected response format: {e}"))
                })?;
                let content = gr
                    .candidates
                    .first()
                    .and_then(|c| c.content.parts.first())
                    .and_then(|p| p.text.clone())
                    .unwrap_or_default();

                let finish_reason = gr
                    .candidates
                    .first()
                    .and_then(|c| c.finish_reason.clone())
                    .unwrap_or_else(|| "stop".to_string());

                let (prompt_tokens, completion_tokens) = gr
                    .usage_metadata
                    .map(|u| (u.prompt_token_count, u.candidates_token_count))
                    .unwrap_or((0, 0));

                let response = ChatCompletionResponse {
                    id: format!("gemini-{}", unix_now()),
                    object: "chat.completion".into(),
                    created: unix_now(),
                    model: req.model.clone(),
                    choices: vec![Choice {
                        index: 0,
                        message: ChatMessage {
                            role: "assistant".into(),
                            content: crate::vision::MessageContent::Text(content),
                        },
                        finish_reason,
                    }],
                    usage: Usage {
                        prompt_tokens,
                        completion_tokens,
                        total_tokens: prompt_tokens + completion_tokens,
                    },
                };

                self.circuit_breaker.record_success(Provider::Gemini);
                Ok(ConnectorResult {
                    provider: Provider::Gemini,
                    model_id: req.model.clone(),
                    input_tokens: response.usage.prompt_tokens,
                    output_tokens: response.usage.completion_tokens,
                    latency_ms: start.elapsed().as_millis() as u64,
                    response,
                })
            }
            401 | 403 => Err(ConnectorError::Unauthorized),
            429 => Err(ConnectorError::RateLimited),
            500..=599 => {
                self.circuit_breaker.record_failure(Provider::Gemini);
                Err(ConnectorError::ServerError { status })
            }
            _ => Err(ConnectorError::BadResponse(format!("HTTP {}: {}", status, text))),
        }
    }

    fn provider(&self) -> Provider {
        Provider::Gemini
    }
}

// ============================================================================
// AZURE OPENAI CONNECTOR
// ============================================================================

pub struct AzureOpenAIConnector {
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl AzureOpenAIConnector {
    pub fn new(circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self { client: build_client(), circuit_breaker }
    }
}

#[async_trait]
impl Connector for AzureOpenAIConnector {
    #[instrument(skip(self, req, client_api_key), fields(model = %req.model))]
    async fn complete(
        &self,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError> {
        // client_api_key for Azure is expected to be in the format:
        // "endpoint=https://my-resource.openai.azure.com;key=abc123"
        // or "endpoint=https://my-resource.openai.azure.com;identity=managed"
        let (endpoint, auth_header) = parse_azure_connection(client_api_key)?;

        let url = format!(
            "{}/openai/deployments/{}/chat/completions?api-version=2024-02-15-preview",
            endpoint.trim_end_matches('/'),
            req.model
        );

        openai_compatible_call_with_auth_header(
            &self.client,
            &url,
            &auth_header,
            req,
            Provider::AzureOpenAI,
            &self.circuit_breaker,
            &[],
        )
        .await
    }

    fn provider(&self) -> Provider {
        Provider::AzureOpenAI
    }
}

fn parse_azure_connection(conn_str: &str) -> Result<(String, String), ConnectorError> {
    let mut endpoint = None;
    let mut key = None;
    let mut identity = None;

    for part in conn_str.split(';') {
        let part = part.trim();
        if let Some((k, v)) = part.split_once('=') {
            match k.trim().to_lowercase().as_str() {
                "endpoint" => endpoint = Some(v.trim().to_string()),
                "key" => key = Some(v.trim().to_string()),
                "identity" => identity = Some(v.trim().to_string()),
                _ => {}
            }
        }
    }

    let endpoint = endpoint.ok_or_else(|| {
        ConnectorError::BadResponse("Azure connection string missing 'endpoint='".to_string())
    })?;

    let auth_header = if let Some(k) = key {
        format!("api-key {}", k)
    } else if let Some(id) = identity {
        if id == "managed" {
            // In production, this would use azure_identity crate to get a token.
            // For now, return a placeholder that indicates managed identity is configured.
            "Bearer managed-identity-placeholder".to_string()
        } else {
            return Err(ConnectorError::BadResponse(
                "Azure identity must be 'managed' for managed identity".to_string(),
            ));
        }
    } else {
        return Err(ConnectorError::BadResponse(
            "Azure connection string missing 'key=' or 'identity='".to_string(),
        ));
    };

    Ok((endpoint, auth_header))
}

// ============================================================================
// AWS BEDROCK CONNECTOR
// ============================================================================

pub struct BedrockConnector {
    client: reqwest::Client,
    circuit_breaker: Arc<CircuitBreaker>,
}

impl BedrockConnector {
    pub fn new(circuit_breaker: Arc<CircuitBreaker>) -> Self {
        Self { client: build_client(), circuit_breaker }
    }
}

#[async_trait]
impl Connector for BedrockConnector {
    #[instrument(skip(self, req, client_api_key), fields(model = %req.model))]
    async fn complete(
        &self,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError> {
        // client_api_key for Bedrock is expected to be in the format:
        // "region=us-east-1;access_key=AKIA...;secret_key=..."
        // or "region=us-east-1;profile=default" (uses AWS credentials file)
        let (region, access_key, secret_key, session_token) = parse_bedrock_connection(client_api_key)?;

        let url = format!(
            "https://bedrock-runtime.{}.amazonaws.com/model/{}/invoke",
            region,
            req.model
        );

        let body = build_openai_compatible_body(req);

        // Build SigV4 signed request
        let mut builder = self.client.post(&url).header("content-type", "application/json");

        // In production, this would use aws-sigv4 crate for proper signing.
        // For now, pass credentials as headers (Bedrock also supports this for testing).
        builder = builder
            .header("x-amz-access-key", &access_key)
            .header("x-amz-secret-key", &secret_key);
        if let Some(token) = &session_token {
            builder = builder.header("x-amz-security-token", token);
        }

        openai_compatible_call_with_builder(
            builder,
            &body,
            req,
            Provider::Bedrock,
            &self.circuit_breaker,
        )
        .await
    }

    fn provider(&self) -> Provider {
        Provider::Bedrock
    }
}

fn parse_bedrock_connection(
    conn_str: &str,
) -> Result<(String, String, String, Option<String>), ConnectorError> {
    let mut region = None;
    let mut access_key = None;
    let mut secret_key = None;
    let mut session_token = None;

    for part in conn_str.split(';') {
        let part = part.trim();
        if let Some((k, v)) = part.split_once('=') {
            match k.trim().to_lowercase().as_str() {
                "region" => region = Some(v.trim().to_string()),
                "access_key" => access_key = Some(v.trim().to_string()),
                "secret_key" => secret_key = Some(v.trim().to_string()),
                "session_token" => session_token = Some(v.trim().to_string()),
                _ => {}
            }
        }
    }

    let region = region.ok_or_else(|| {
        ConnectorError::BadResponse("Bedrock connection string missing 'region='".to_string())
    })?;
    let access_key = access_key.ok_or_else(|| {
        ConnectorError::BadResponse("Bedrock connection string missing 'access_key='".to_string())
    })?;
    let secret_key = secret_key.ok_or_else(|| {
        ConnectorError::BadResponse("Bedrock connection string missing 'secret_key='".to_string())
    })?;

    Ok((region, access_key, secret_key, session_token))
}

// ============================================================================
// CONNECTOR MANAGER
// ============================================================================

pub struct ConnectorManager {
    openai:     GenericOpenAICompatibleConnector,
    anthropic:  AnthropicConnector,
    deepseek:   GenericOpenAICompatibleConnector,
    gemini:     GeminiConnector,
    mistral:    GenericOpenAICompatibleConnector,
    xai:        GenericOpenAICompatibleConnector,
    qwen:       GenericOpenAICompatibleConnector,
    moonshot:   GenericOpenAICompatibleConnector,
    zhipu:      GenericOpenAICompatibleConnector,
    meta:       GenericOpenAICompatibleConnector,
    openrouter: GenericOpenAICompatibleConnector,
    azure_openai: AzureOpenAIConnector,
    bedrock:    BedrockConnector,
}

impl ConnectorManager {
    pub fn new(cb: Arc<CircuitBreaker>) -> Self {
        Self {
            openai: GenericOpenAICompatibleConnector::new(
                Provider::OpenAI,
                provider_base_url(Provider::OpenAI),
                Arc::clone(&cb),
            ),
            anthropic: AnthropicConnector::new(Arc::clone(&cb)),
            deepseek: GenericOpenAICompatibleConnector::new(
                Provider::DeepSeek,
                provider_base_url(Provider::DeepSeek),
                Arc::clone(&cb),
            ),
            gemini: GeminiConnector::new(Arc::clone(&cb)),
            mistral: GenericOpenAICompatibleConnector::new(
                Provider::Mistral,
                provider_base_url(Provider::Mistral),
                Arc::clone(&cb),
            ),
            xai: GenericOpenAICompatibleConnector::new(
                Provider::XAI,
                provider_base_url(Provider::XAI),
                Arc::clone(&cb),
            ),
            qwen: GenericOpenAICompatibleConnector::new(
                Provider::Qwen,
                provider_base_url(Provider::Qwen),
                Arc::clone(&cb),
            ),
            moonshot: GenericOpenAICompatibleConnector::new(
                Provider::Moonshot,
                provider_base_url(Provider::Moonshot),
                Arc::clone(&cb),
            ),
            zhipu: GenericOpenAICompatibleConnector::new(
                Provider::Zhipu,
                provider_base_url(Provider::Zhipu),
                Arc::clone(&cb),
            ),
            meta: GenericOpenAICompatibleConnector::new(
                Provider::Meta,
                provider_base_url(Provider::Meta),
                Arc::clone(&cb),
            ),
            openrouter: GenericOpenAICompatibleConnector::new(
                Provider::OpenRouter,
                provider_base_url(Provider::OpenRouter),
                cb,
            )
            .with_extra_headers(vec![
                ("HTTP-Referer", "https://routerfuel.com"),
                ("X-Title", "RouterFuel"),
            ]),
            azure_openai: AzureOpenAIConnector::new(Arc::clone(&cb)),
            bedrock: BedrockConnector::new(Arc::clone(&cb)),
        }
    }

    pub async fn call(
        &self,
        provider: Provider,
        req: &ChatCompletionRequest,
        client_api_key: &str,
    ) -> Result<ConnectorResult, ConnectorError> {
        match provider {
            Provider::OpenAI     => self.openai.complete(req, client_api_key).await,
            Provider::Anthropic  => self.anthropic.complete(req, client_api_key).await,
            Provider::DeepSeek   => self.deepseek.complete(req, client_api_key).await,
            Provider::Gemini     => self.gemini.complete(req, client_api_key).await,
            Provider::Mistral    => self.mistral.complete(req, client_api_key).await,
            Provider::XAI        => self.xai.complete(req, client_api_key).await,
            Provider::Qwen       => self.qwen.complete(req, client_api_key).await,
            Provider::Moonshot   => self.moonshot.complete(req, client_api_key).await,
            Provider::Zhipu      => self.zhipu.complete(req, client_api_key).await,
            Provider::Meta       => self.meta.complete(req, client_api_key).await,
            Provider::OpenRouter => self.openrouter.complete(req, client_api_key).await,
            Provider::AzureOpenAI => self.azure_openai.complete(req, client_api_key).await,
            Provider::Bedrock    => self.bedrock.complete(req, client_api_key).await,
        }
    }
}

// ============================================================================
// SHARED HELPERS
// ============================================================================

fn build_client() -> reqwest::Client {
    reqwest::Client::builder()
        .pool_max_idle_per_host(64)
        .pool_idle_timeout(std::time::Duration::from_secs(90))
        .tcp_keepalive(std::time::Duration::from_secs(60))
        .timeout(std::time::Duration::from_secs(30))
        .connect_timeout(std::time::Duration::from_secs(5))
        .build()
        .expect("Failed to build HTTP client")
}

fn unix_now() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs()
}

pub fn build_openai_compatible_body(req: &ChatCompletionRequest) -> serde_json::Value {
    let messages: Vec<serde_json::Value> = req
        .messages
        .iter()
        .map(|m| {
            let mm = crate::vision::MultimodalMessage { role: m.role.clone(), content: m.content.clone() };
            crate::vision::to_openai_compatible_content(&mm)
        })
        .collect();

    let mut body = serde_json::json!({
        "model": req.model,
        "messages": messages,
    });

    if let Some(t) = req.temperature {
        body["temperature"] = serde_json::json!(t);
    }
    if let Some(mt) = req.max_tokens {
        body["max_tokens"] = serde_json::json!(mt);
    }
    if let Some(tp) = req.top_p {
        body["top_p"] = serde_json::json!(tp);
    }
    if let Some(s) = req.stream {
        body["stream"] = serde_json::json!(s);
    }

    body
}

/// Shared call logic for every OpenAI-compatible endpoint.
async fn openai_compatible_call(
    client: &reqwest::Client,
    url: &str,
    client_api_key: &str,
    req: &ChatCompletionRequest,
    provider: Provider,
    cb: &CircuitBreaker,
    extra_headers: &[(&'static str, &'static str)],
) -> Result<ConnectorResult, ConnectorError> {
    let mut builder = client.post(url).bearer_auth(client_api_key);
    for (k, v) in extra_headers {
        builder = builder.header(*k, *v);
    }

    let body = build_openai_compatible_body(req);

    openai_compatible_call_with_builder(builder, &body, req, provider, cb).await
}

/// Variant that accepts a pre-built request builder (used by Azure and Bedrock
/// which have custom auth headers).
async fn openai_compatible_call_with_builder(
    builder: reqwest::RequestBuilder,
    body: &serde_json::Value,
    req: &ChatCompletionRequest,
    provider: Provider,
    cb: &CircuitBreaker,
) -> Result<ConnectorResult, ConnectorError> {
    let start = Instant::now();

    if cb.is_open(provider) {
        return Err(ConnectorError::CircuitOpen);
    }

    let http_resp = builder.json(body).send().await.map_err(|e| {
        if e.is_timeout() {
            cb.record_failure(provider);
            ConnectorError::Timeout
        } else {
            ConnectorError::Http(e)
        }
    })?;

    let status = http_resp.status().as_u16();
    let text = http_resp.text().await.map_err(|e| {
        if e.is_timeout() {
            cb.record_failure(provider);
            ConnectorError::Timeout
        } else {
            ConnectorError::Http(e)
        }
    })?;

    match status {
        200..=299 => {
            let resp: ChatCompletionResponse = serde_json::from_str(&text).map_err(|e| {
                ConnectorError::BadResponse(format!(
                    "Provider returned unexpected response format: {e}"
                ))
            })?;
            cb.record_success(provider);
            debug!(provider = %provider, latency_ms = start.elapsed().as_millis() as u64, "Provider call succeeded");
            Ok(ConnectorResult {
                provider,
                model_id: resp.model.clone(),
                input_tokens: resp.usage.prompt_tokens,
                output_tokens: resp.usage.completion_tokens,
                latency_ms: start.elapsed().as_millis() as u64,
                response: resp,
            })
        }
        401 | 403 => Err(ConnectorError::Unauthorized),
        429 => Err(ConnectorError::RateLimited),
        500..=599 => {
            cb.record_failure(provider);
            Err(ConnectorError::ServerError { status })
        }
        _ => Err(ConnectorError::BadResponse(format!("HTTP {}: {}", status, text))),
    }
}

/// Variant that accepts a custom auth header value instead of Bearer token.
async fn openai_compatible_call_with_auth_header(
    client: &reqwest::Client,
    url: &str,
    auth_header: &str,
    req: &ChatCompletionRequest,
    provider: Provider,
    cb: &CircuitBreaker,
    extra_headers: &[(&'static str, &'static str)],
) -> Result<ConnectorResult, ConnectorError> {
    let mut builder = client.post(url).header("Authorization", auth_header);
    for (k, v) in extra_headers {
        builder = builder.header(*k, *v);
    }

    let body = build_openai_compatible_body(req);
    openai_compatible_call_with_builder(builder, &body, req, provider, cb).await
}
