// =============================================================================
// src/auth.rs — RouterFuel v0.4
//
// How it works:
//   1. Client sends:   X-API-Key: rf_live_abc123yoursecretkey
//   2. Middleware SHA-256 hashes the raw key
//   3. Compares hash against the in-memory store (no plaintext ever stored)
//   4. If valid → injects X-Routerfuel-Client-Id header for downstream handlers
//   5. If invalid → returns 401 immediately, request never reaches the handler
//   6. Also provides ClientProviderKeys extractor for BYOK provider headers
//
// Key store is loaded once at startup from the ROUTERFUEL_API_KEYS env var.
// Format:  sha256hex:ClientName,sha256hex:ClientName,...
//
// To generate a key hash on the command line:
//   echo -n "rf_live_yoursecretkey" | sha256sum | awk '{print $1}'
// =============================================================================

use axum::{
    body::Body,
    extract::State,
    http::{HeaderMap, Request, StatusCode},
    middleware::Next,
    response::{IntoResponse, Json, Response},
};
use serde::Serialize;
use sha2::{Digest, Sha256};
use std::{collections::HashMap, sync::Arc};
use tracing::{debug, warn};

// =============================================================================
// ClientProviderKeys — BYOK (Bring Your Own Key) header container
// =============================================================================

#[derive(Debug, Default, Clone, PartialEq, Eq)]
pub struct ClientProviderKeys {
    pub openai: Option<String>,
    pub anthropic: Option<String>,
    pub deepseek: Option<String>,
    pub gemini: Option<String>,
    pub mistral: Option<String>,
    pub xai: Option<String>,
    pub qwen: Option<String>,
    pub moonshot: Option<String>,
    pub zhipu: Option<String>,
    pub meta: Option<String>,
    /// OpenRouter acts as a universal fallback: if a client supplies only
    /// this key, RouterFuel routes *any* model through OpenRouter instead of
    /// requiring a separate key per lab — see main.rs `resolve_byok_route`.
    pub openrouter: Option<String>,
    /// Azure OpenAI connection string: "endpoint=...;key=..." or "endpoint=...;identity=managed"
    pub azure_openai: Option<String>,
    /// AWS Bedrock connection string: "region=...;access_key=...;secret_key=..."
    pub bedrock: Option<String>,
}

impl ClientProviderKeys {
    /// Extracts BYOK headers from incoming request headers.
    pub fn from_headers(headers: &HeaderMap) -> Self {
        Self {
            openai: extract_header_string(headers, "x-openai-api-key"),
            anthropic: extract_header_string(headers, "x-anthropic-api-key"),
            deepseek: extract_header_string(headers, "x-deepseek-api-key"),
            gemini: extract_header_string(headers, "x-gemini-api-key"),
            mistral: extract_header_string(headers, "x-mistral-api-key"),
            xai: extract_header_string(headers, "x-xai-api-key")
                .or_else(|| extract_header_string(headers, "x-grok-api-key")),
            qwen: extract_header_string(headers, "x-qwen-api-key")
                .or_else(|| extract_header_string(headers, "x-dashscope-api-key")),
            moonshot: extract_header_string(headers, "x-moonshot-api-key")
                .or_else(|| extract_header_string(headers, "x-kimi-api-key")),
            zhipu: extract_header_string(headers, "x-zhipu-api-key")
                .or_else(|| extract_header_string(headers, "x-glm-api-key")),
            meta: extract_header_string(headers, "x-meta-api-key")
                .or_else(|| extract_header_string(headers, "x-llama-api-key")),
            openrouter: extract_header_string(headers, "x-openrouter-api-key"),
            azure_openai: extract_header_string(headers, "x-azure-openai-connection"),
            bedrock: extract_header_string(headers, "x-bedrock-connection"),
        }
    }

    /// Returns the client's key for a specific provider, if supplied.
    pub fn for_provider(&self, provider: crate::connectors::Provider) -> Option<&str> {
        use crate::connectors::Provider;
        match provider {
            Provider::OpenAI     => self.openai.as_deref(),
            Provider::Anthropic  => self.anthropic.as_deref(),
            Provider::DeepSeek   => self.deepseek.as_deref(),
            Provider::Gemini     => self.gemini.as_deref(),
            Provider::Mistral    => self.mistral.as_deref(),
            Provider::XAI        => self.xai.as_deref(),
            Provider::Qwen       => self.qwen.as_deref(),
            Provider::Moonshot   => self.moonshot.as_deref(),
            Provider::Zhipu      => self.zhipu.as_deref(),
            Provider::Meta       => self.meta.as_deref(),
            Provider::OpenRouter => self.openrouter.as_deref(),
            Provider::AzureOpenAI => self.azure_openai.as_deref(),
            Provider::Bedrock    => self.bedrock.as_deref(),
        }
    }

    /// Returns true if at least one BYOK provider key was supplied.
    #[allow(dead_code)]
    pub fn has_any(&self) -> bool {
        self.openai.is_some()
            || self.anthropic.is_some()
            || self.deepseek.is_some()
            || self.gemini.is_some()
            || self.mistral.is_some()
            || self.xai.is_some()
            || self.qwen.is_some()
            || self.moonshot.is_some()
            || self.zhipu.is_some()
            || self.meta.is_some()
            || self.openrouter.is_some()
            || self.azure_openai.is_some()
            || self.bedrock.is_some()
    }
}

/// Helper function to safely pull and sanitize header strings
fn extract_header_string(headers: &HeaderMap, key: &str) -> Option<String> {
    headers
        .get(key)
        .and_then(|h| h.to_str().ok())
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

// =============================================================================
// ApiKeyStore — loaded once, shared via Arc, read-only after init
// =============================================================================

pub struct ApiKeyStore {
    /// sha256_hex → client display name
    keys: HashMap<String, String>,
}

impl ApiKeyStore {
    /// Parse the ROUTERFUEL_API_KEYS env var.
    /// Expected format:  "sha256hex:ClientA,sha256hex:ClientB"
    pub fn from_env_string(raw: &str) -> Self {
        let mut keys = HashMap::new();

        for entry in raw.split(',') {
            let entry = entry.trim();
            if entry.is_empty() { continue; }

            match entry.split_once(':') {
                Some((hash, name)) => {
                    let hash = hash.trim().to_lowercase();
                    let name = name.trim().to_string();
                    if hash.len() == 64 {
                        keys.insert(hash, name);
                    } else {
                        tracing::error!(
                            "Invalid API key hash '{}' — expected 64-char SHA-256 hex string",
                            hash
                        );
                    }
                }
                None => {
                    tracing::error!(
                        "Malformed ROUTERFUEL_API_KEYS entry '{}' — expected 'sha256hex:ClientName'",
                        entry
                    );
                }
            }
        }

        tracing::info!("Loaded {} API key(s) into auth store", keys.len());
        Self { keys }
    }

   /// Validate a raw API key. Returns (client_hash, client_name) if valid.
    /// `client_hash` is the canonical identifier used everywhere else in
    /// the system — rate_limiter, spend_guard, loop_guard, and the
    /// client_tiers table all key off SHA-256(raw_key) (see
    /// migrations/003_client_tiers.sql). `client_name` is display-only.
    pub fn validate(&self, raw_key: &str) -> Option<(&str, &str)> {
        let hash = sha256_hex(raw_key);
        self.keys.get_key_value(&hash).map(|(h, n)| (h.as_str(), n.as_str()))
    }

    pub fn len(&self) -> usize {
        self.keys.len()
    }

    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

// =============================================================================
// Middleware
// =============================================================================

#[derive(Serialize)]
struct AuthError {
    error: AuthErrorDetail,
}

#[derive(Serialize)]
struct AuthErrorDetail {
    message: &'static str,
    code:    &'static str,
}

fn unauthorized(message: &'static str) -> Response {
    let body = Json(AuthError {
        error: AuthErrorDetail {
            message,
            code: "unauthorized",
        },
    });
    (StatusCode::UNAUTHORIZED, body).into_response()
}

/// Axum middleware — validates X-API-Key, injects client ID, passes through.
pub async fn api_key_middleware(
    State(store): State<Arc<ApiKeyStore>>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    // Extract raw key from header
    let raw_key = match request.headers().get("x-api-key") {
        Some(v) => match v.to_str() {
            Ok(s) => s.to_string(),
            Err(_) => {
                warn!("X-API-Key header contains non-UTF-8 bytes");
                return unauthorized("X-API-Key header must be a valid UTF-8 string");
            }
        },
        None => {
            debug!("Request missing X-API-Key header");
            return unauthorized("Missing X-API-Key header");
        }
    };

    // Validate against the store
    match store.validate(&raw_key) {
        Some((client_hash, client_name)) => {
            debug!(client = client_name, client_hash = &client_hash[..8], "API key validated");

            // FIX: inject the HASH, not the display name — this is what
            // client_registry.rs registers tiers under, and what
            // client_tiers.client_id in Postgres actually stores. Injecting
            // the display name here silently orphaned every configured
            // tier: every client fell through RateLimiter::check()'s
            // "unregistered → auto-register at default tier" path forever.
            request.headers_mut().insert(
                "x-routerfuel-client-id",
                client_hash.parse().expect("sha256 hex digest is always valid header text"),
            );

            next.run(request).await
        }
        None => {
            warn!(
                key_prefix = &raw_key[..raw_key.len().min(8)],
                "Invalid API key"
            );
            unauthorized("Invalid API key")
        }
    }
}

// =============================================================================
// Helpers
// =============================================================================

fn sha256_hex(input: &str) -> String {
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    fn store_with(raw_key: &str, name: &str) -> ApiKeyStore {
        let hash = sha256_hex(raw_key);
        let env_str = format!("{}:{}", hash, name);
        ApiKeyStore::from_env_string(&env_str)
    }

    #[test]
    fn valid_key_returns_client_name() {
        let store = store_with("rf_live_supersecret", "AcmeCorp");
        let hash = sha256_hex("rf_live_supersecret");
        // FIX: validate() now returns Some((hash, name)) instead of Some(name)
        assert_eq!(store.validate("rf_live_supersecret"), Some((hash.as_str(), "AcmeCorp")));
    }
    
    #[test]
    fn wrong_key_returns_none() {
        let store = store_with("rf_live_supersecret", "AcmeCorp");
        assert!(store.validate("rf_live_wrongkey").is_none());
    }

    #[test]
    fn empty_env_string_produces_empty_store() {
        let store = ApiKeyStore::from_env_string("");
        assert!(store.is_empty());
    }

    
    #[test]
    fn multiple_keys_parsed_correctly() {
        let k1 = sha256_hex("key_one");
        let k2 = sha256_hex("key_two");
        let env_str = format!("{k1}:ClientA,{k2}:ClientB");
        let store = ApiKeyStore::from_env_string(&env_str);
        assert_eq!(store.len(), 2);
        // FIX: same tuple-return update
        assert_eq!(store.validate("key_one"), Some((k1.as_str(), "ClientA")));
        assert_eq!(store.validate("key_two"), Some((k2.as_str(), "ClientB")));
    }

    #[test]
    fn short_hash_rejected() {
        // Only 10 chars — not a valid SHA-256 hex
        let store = ApiKeyStore::from_env_string("abc123:Client");
        assert!(store.is_empty());
    }

    #[test]
    fn key_not_stored_in_plaintext() {
        let store = store_with("rf_live_plaintextkey", "TestClient");
        // The raw key must not appear anywhere in the keys map
        for k in store.keys.keys() {
            assert_ne!(k, "rf_live_plaintextkey");
        }
    }

    #[test]
    fn extracts_byok_headers_correctly() {
        use axum::http::HeaderValue;

        let mut headers = HeaderMap::new();
        headers.insert("x-openai-api-key", HeaderValue::from_static("sk-proj-openai123"));
        headers.insert("x-anthropic-api-key", HeaderValue::from_static("sk-ant-anthropic456"));

        let keys = ClientProviderKeys::from_headers(&headers);
        assert_eq!(keys.openai.as_deref(), Some("sk-proj-openai123"));
        assert_eq!(keys.anthropic.as_deref(), Some("sk-ant-anthropic456"));
        assert!(keys.deepseek.is_none());
        assert!(keys.gemini.is_none());
        assert!(keys.has_any());
    }
}
// =============================================================================
// CURSOR BRIDGE
//
// Cursor's "Override OpenAI Base URL" integration sends exactly one opaque
// string as an `Authorization: Bearer <token>` header — no custom headers,
// no way to send X-API-Key and X-<Provider>-Api-Key separately the way
// RouterFuel's own auth model expects.
//
// This middleware sits in front of api_key_middleware and translates a
// composite Bearer token into the two headers RouterFuel already knows how
// to read. It's a pure adapter — resolve_byok_route, ClientProviderKeys,
// ApiKeyStore are all untouched.
//
// Token format (what the user pastes into Cursor's API Key field):
//
//     <routerfuel_api_key>:<provider>:<byok_provider_key>
//
// e.g.  rf_live_abc123:anthropic:sk-ant-...
//       rf_live_abc123:openrouter:sk-or-...   (universal fallback — works
//                                               for any model if you're not
//                                               sure which provider to pick)
//
// Split with splitn(3, ':') so a colon-containing provider key (unusual,
// but not impossible for some providers) still lands entirely in the third
// field rather than getting truncated.
//
// Only activates when the request has no X-API-Key already — existing API
// clients that already send X-API-Key + X-<Provider>-Api-Key directly are
// completely unaffected by this layer.
// =============================================================================

pub async fn cursor_bridge_middleware(mut request: Request<Body>, next: Next) -> Response {
    let headers = request.headers();

    if headers.contains_key("x-api-key") {
        // Caller already speaks RouterFuel's native header pair — nothing
        // to bridge.
        return next.run(request).await;
    }

    let bearer_token = headers
        .get(axum::http::header::AUTHORIZATION)
        .and_then(|v| v.to_str().ok())
        .and_then(|s| s.strip_prefix("Bearer "))
        .map(str::to_string);

    let Some(token) = bearer_token else {
        // No composite token and no X-API-Key — let it fall through to
        // api_key_middleware, which will reject it with the normal 401.
        return next.run(request).await;
    };

    let parts: Vec<&str> = token.splitn(3, ':').collect();
    let [routerfuel_key, provider, byok_key] = parts[..] else {
        return unauthorized(
            "Authorization Bearer token is not a valid RouterFuel composite key. \
             Expected format: <routerfuel_api_key>:<provider>:<byok_provider_key>, \
             e.g. rf_live_xxx:anthropic:sk-ant-xxx. See /docs/cursor for setup.",
        );
    };

   let provider_header_name = match provider.to_lowercase().as_str() {
        "openai"               => "x-openai-api-key",
        "anthropic"            => "x-anthropic-api-key",
        "deepseek"             => "x-deepseek-api-key",
        "gemini"               => "x-gemini-api-key",
        "mistral"              => "x-mistral-api-key",
        "xai" | "grok"         => "x-xai-api-key",
        "qwen" | "dashscope"   => "x-qwen-api-key",
        "moonshot" | "kimi"    => "x-moonshot-api-key",
        "zhipu" | "glm"        => "x-zhipu-api-key",
        "meta" | "llama"       => "x-meta-api-key",
        "openrouter"           => "x-openrouter-api-key",
        "azure" | "azure_openai" => "x-azure-openai-connection",
        "bedrock" | "aws"      => "x-bedrock-connection",
        _ => {
            // FIX: was Box::leak()-ing a formatted String per bad request —
            // this middleware runs before auth/rate-limiting, so an
            // unauthenticated caller could leak memory indefinitely by
            // spamming an unrecognized provider name. Fixed message, no
            // per-request allocation leaked, no echo of attacker input.
            return unauthorized(
                "Unknown provider in composite key. Expected one of: openai, anthropic, \
                 deepseek, gemini, mistral, xai, qwen, moonshot, zhipu, meta, openrouter, \
                 azure, bedrock. See /docs/cursor for the composite key format.",
            );
        }
    };

    let Ok(rf_key_value) = routerfuel_key.parse() else {
        return unauthorized("Composite key's RouterFuel API key segment is not valid header text.");
    };
    let Ok(byok_key_value) = byok_key.parse() else {
        return unauthorized("Composite key's provider API key segment is not valid header text.");
    };

    request.headers_mut().insert("x-api-key", rf_key_value);
    request
        .headers_mut()
        .insert(provider_header_name, byok_key_value);

    next.run(request).await
}

// Reuses the `unauthorized()` fn already defined above in this file (used by
// api_key_middleware) — no need to redefine it here.

// Same as `unauthorized` above but for a message built at runtime (e.g. one
// that echoes back the unrecognized provider name) — AuthErrorDetail.message
// is `&'static str`, so an owned String has to be leaked to fit it. This
// only runs on a malformed-request error path, never in steady-state
// traffic, so the leak is bounded by bad requests, not by legitimate load.

