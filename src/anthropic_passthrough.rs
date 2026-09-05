// =============================================================================
// src/anthropic_passthrough.rs — RouterFuel
//
// Native Anthropic Messages API surface: POST /v1/messages and
// POST /v1/messages/count_tokens. This is what lets Claude Code (and any
// other client speaking the native protocol) point at RouterFuel via
// ANTHROPIC_BASE_URL.
//
// DESIGN: this is a PASSTHROUGH, not a translation layer. The request body
// is forwarded to api.anthropic.com byte-for-byte and the response is
// relayed back the same way.
//
// That is deliberate. RouterFuel's internal ChatCompletionRequest carries
// only model/messages/temperature/max_tokens/top_p/stream — it has no
// `tools`, no `tool_choice`, no top-level `system`, no `cache_control` and
// no `thinking`. Claude Code is fundamentally a tool-use agent, so routing
// it through that struct would not degrade it, it would break it. Keeping
// the body opaque means tool_use, thinking blocks, prompt caching, the
// `refusal` stop reason, and anything Anthropic ships next all work on day
// one with no mapping to maintain.
//
// WHAT THIS KEEPS: RouterFuel's own auth (including the composite Bearer
// token, so a single ANTHROPIC_AUTH_TOKEN carries both keys), rate
// limiting, SpendGuard reserve/reconcile, LoopGuard, the circuit breaker,
// request_logs, and telemetry.
//
// WHAT THIS GIVES UP: cross-provider routing on this endpoint. An
// Anthropic-format body can only go to Anthropic, so there is no "auto"
// here and no semantic cache. Translating between protocols so Claude Code
// could run on Grok or Gemini is a much larger piece of work — see the
// Option B discussion in CHANGES; this is Option A.
//
// ONE HONEST GAP: spend is reserved from a tiktoken estimate over the
// extracted text and then reconciled against Anthropic's own exact
// `usage`, which makes the *final* accounting exact on this path — better
// than /v1/chat/completions, which never sees a provider's real counts.
// But if the model is not in the registry we have no rates, so the
// reservation is zero and the spend cap does not bind for that request.
// Failing closed was the alternative and it would defeat the point of a
// passthrough (a brand-new Claude would be unusable until someone edited
// route_engine.rs), so it warns loudly instead.
// =============================================================================

use crate::auth::ClientProviderKeys;
use crate::connectors::Provider;
use crate::cost_tracker::CostTracker;
use crate::guardrails::{LoopGuard, SpendGuard};
use crate::route_engine::{normalize_model_id, RouteEngine};
use crate::tokens::{self, TokenCostBreakdown};
use axum::body::{Body, Bytes};
use axum::http::{HeaderMap, StatusCode};
use axum::response::{IntoResponse, Response};
use futures_util::StreamExt;
use std::sync::Arc;
use std::time::Instant;
use tracing::{debug, error, info, warn};
use uuid::Uuid;

const ANTHROPIC_MESSAGES_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_COUNT_TOKENS_URL: &str = "https://api.anthropic.com/v1/messages/count_tokens";
const DEFAULT_ANTHROPIC_VERSION: &str = "2023-06-01";

/// Everything the passthrough needs from AppState. Kept as its own struct
/// so main.rs can hand over exactly these pieces rather than the whole
/// AppState, which carries a lot this endpoint never touches (semantic
/// cache, vision, the OpenAI-shaped connectors).
#[derive(Clone)]
pub struct PassthroughState {
    pub route_engine: Arc<RouteEngine>,
    pub cost_tracker: Arc<CostTracker>,
    pub spend_guard: Arc<SpendGuard>,
    pub loop_guard: Arc<LoopGuard>,
    pub rate_limiter: Arc<crate::rate_limiter::RateLimiter>,
    pub http: reqwest::Client,
}

fn json_error(status: StatusCode, error_type: &str, message: String) -> Response {
    let body = serde_json::json!({
        // Anthropic's error envelope, so a native client parses our errors
        // with the same code path it uses for the provider's.
        "type": "error",
        "error": { "type": error_type, "message": message }
    });
    (status, axum::Json(body)).into_response()
}

/// Pulls every text fragment out of an Anthropic-format body: the
/// top-level `system` (string or block array) plus each message's content
/// (string or block array). Used for the LoopGuard fingerprint and the
/// input-token estimate — never for rewriting the request.
fn extract_text(body: &serde_json::Value) -> String {
    let mut out = String::new();

    let push_content = |v: &serde_json::Value, out: &mut String| match v {
        serde_json::Value::String(s) => {
            out.push_str(s);
            out.push('\n');
        }
        serde_json::Value::Array(blocks) => {
            for b in blocks {
                if let Some(t) = b.get("text").and_then(|t| t.as_str()) {
                    out.push_str(t);
                    out.push('\n');
                }
            }
        }
        _ => {}
    };

    if let Some(sys) = body.get("system") {
        push_content(sys, &mut out);
    }
    if let Some(msgs) = body.get("messages").and_then(|m| m.as_array()) {
        for m in msgs {
            if let Some(c) = m.get("content") {
                push_content(c, &mut out);
            }
        }
    }

    out
}

/// Copies the Anthropic protocol headers a native client sends. Notably
/// `anthropic-beta`, which Claude Code uses for features gated behind beta
/// flags — dropping it would silently disable them.
fn forward_protocol_headers(
    mut rb: reqwest::RequestBuilder,
    headers: &HeaderMap,
) -> reqwest::RequestBuilder {
    let version = headers
        .get("anthropic-version")
        .and_then(|v| v.to_str().ok())
        .unwrap_or(DEFAULT_ANTHROPIC_VERSION);
    rb = rb.header("anthropic-version", version);

    if let Some(beta) = headers.get("anthropic-beta").and_then(|v| v.to_str().ok()) {
        rb = rb.header("anthropic-beta", beta);
    }

    rb.header("content-type", "application/json")
}

/// Resolves the client's Anthropic key, or explains precisely how to supply
/// one. This endpoint is Anthropic-native, so unlike the OpenAI-shaped path
/// there is no OpenRouter fallback to lean on — OpenRouter's completions
/// API does not accept an Anthropic-format body.
fn require_anthropic_key(headers: &HeaderMap) -> Result<String, Response> {
    let keys = ClientProviderKeys::from_headers(headers);
    match keys.anthropic {
        Some(k) => Ok(k),
        None => Err(json_error(
            StatusCode::BAD_REQUEST,
            "authentication_error",
            "No Anthropic API key supplied. RouterFuel is bring-your-own-key and never bills \
             its own account. Send your key as X-Anthropic-Api-Key, or — if your client can \
             only send one auth header, as Claude Code does — set ANTHROPIC_AUTH_TOKEN to a \
             composite token of the form <routerfuel_api_key>:anthropic:<your_anthropic_key>. \
             Note that /v1/messages is Anthropic-native and cannot be served through an \
             OpenRouter key."
                .to_string(),
        )),
    }
}

/// POST /v1/messages — native Anthropic Messages API.
pub async fn messages_handler(
    headers: HeaderMap,
    axum::extract::State(state): axum::extract::State<PassthroughState>,
    body: Bytes,
) -> Response {
    let request_id = Uuid::new_v4().to_string();
    let start = Instant::now();

    let client_id = headers
        .get("x-routerfuel-client-id")
        .and_then(|v| v.to_str().ok())
        .map(str::to_string);
    let rl_key = client_id.clone().unwrap_or_else(|| "anonymous".to_string());

    if state.rate_limiter.check(&rl_key).is_err() {
        warn!(client_id = %rl_key, "Rate limit exceeded on /v1/messages");
        return json_error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "Rate limit exceeded".to_string(),
        );
    }

    let parsed: serde_json::Value = match serde_json::from_slice(&body) {
        Ok(v) => v,
        Err(e) => {
            return json_error(
                StatusCode::BAD_REQUEST,
                "invalid_request_error",
                format!("Request body is not valid JSON: {e}"),
            )
        }
    };

    let Some(requested_model) = parsed
        .get("model")
        .and_then(|m| m.as_str())
        .map(str::to_string)
    else {
        return json_error(
            StatusCode::BAD_REQUEST,
            "invalid_request_error",
            "Missing required field: model".to_string(),
        );
    };

    // Claude Code sends the long-context variant as `claude-opus-5[1m]`.
    // Normalize for the registry lookup AND for the outbound body — see
    // normalize_body_model for why the wire form has to change too.
    let model = normalize_model_id(&requested_model).to_string();
    let is_stream = parsed
        .get("stream")
        .and_then(|s| s.as_bool())
        .unwrap_or(false);

    let api_key = match require_anthropic_key(&headers) {
        Ok(k) => k,
        Err(resp) => return resp,
    };

    let prompt_text = extract_text(&parsed);
    if !prompt_text.is_empty() && state.loop_guard.check_and_record(&rl_key, &prompt_text) {
        return json_error(
            StatusCode::TOO_MANY_REQUESTS,
            "loop_detected_error",
            "This client has sent the same prompt repeatedly in a short window. If this is a \
             retry loop, space out retries or vary the prompt."
                .to_string(),
        );
    }

    let estimated_input = tokens::count_tokens(&prompt_text).unwrap_or(0);
    let estimated_output = parsed
        .get("max_tokens")
        .and_then(|m| m.as_u64())
        .unwrap_or(1024) as u32;

    // Unknown model => no rates => zero reservation. See the header note:
    // this is a deliberate trade so a model Anthropic has shipped but the
    // registry has not yet learned about stays usable.
    let (rate_in, rate_out) = match state.route_engine.get_pricing(&model) {
        Ok(p) => p,
        Err(_) => {
            warn!(
                request_id = %request_id,
                model = %model,
                requested_model = %requested_model,
                "model is not in the registry — passthrough will proceed but the spend cap \
                 cannot bind for this request. Add it to route_engine.rs to restore cost control."
            );
            (0.0, 0.0)
        }
    };

    let estimated = TokenCostBreakdown::new(estimated_input, estimated_output, rate_in, rate_out);
    if !state.spend_guard.try_reserve(&rl_key, estimated.total_cost_cents) {
        return json_error(
            StatusCode::TOO_MANY_REQUESTS,
            "spend_cap_error",
            "This client has hit its spend cap for the current window.".to_string(),
        );
    }

    info!(
        request_id = %request_id,
        model = %model,
        requested_model = %requested_model,
        stream = is_stream,
        "Received native /v1/messages request"
    );

    let rb = state.http.post(ANTHROPIC_MESSAGES_URL).header("x-api-key", &api_key);
    let rb = forward_protocol_headers(rb, &headers);
    let outbound = normalize_body_model(&body, &parsed, &model);

    // Timed separately from `start` so latency_ms stays provider-only and
    // total_latency_ms carries the whole handler — see migration 009.
    let provider_start = Instant::now();
    let upstream = match rb.body(outbound).send().await {
        Ok(r) => r,
        Err(e) => {
            state.spend_guard.release(&rl_key, estimated.total_cost_cents);
            error!(request_id = %request_id, "Anthropic request failed: {e}");
            return json_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("Upstream request to Anthropic failed: {e}"),
            );
        }
    };

    let status = StatusCode::from_u16(upstream.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);

    if is_stream && status.is_success() {
        // Time-to-headers is the only provider-only figure available on a
        // stream; the body arrives incrementally for as long as generation
        // runs.
        let provider_headers_ms = provider_start.elapsed().as_millis() as u64;
        return relay_stream(
            state, upstream, status, request_id, rl_key, client_id, model.to_string(),
            estimated, rate_in, rate_out, start, provider_headers_ms,
        );
    }

    // Non-streaming: read the body, reconcile against Anthropic's exact
    // usage, then return the provider's bytes verbatim.
    let upstream_bytes = match upstream.bytes().await {
        Ok(b) => b,
        Err(e) => {
            state.spend_guard.release(&rl_key, estimated.total_cost_cents);
            error!(request_id = %request_id, "Failed reading Anthropic response: {e}");
            return json_error(
                StatusCode::BAD_GATEWAY,
                "api_error",
                format!("Failed reading Anthropic response: {e}"),
            );
        }
    };

    if !status.is_success() {
        state.spend_guard.release(&rl_key, estimated.total_cost_cents);
        warn!(request_id = %request_id, status = status.as_u16(), "Anthropic returned an error");
        return passthrough_response(status, upstream_bytes, false);
    }

    let (in_tok, out_tok) = usage_from_message(&upstream_bytes).unwrap_or((estimated_input, 0));
    let actual = TokenCostBreakdown::new(in_tok, out_tok, rate_in, rate_out);
    state
        .spend_guard
        .reconcile(&rl_key, estimated.total_cost_cents, actual.total_cost_cents);

    log_passthrough(
        &state,
        request_id,
        model.to_string(),
        &actual,
        client_id,
        provider_start.elapsed().as_millis() as u64,
        start.elapsed().as_millis() as u64,
    );

    passthrough_response(status, upstream_bytes, false)
}

/// POST /v1/messages/count_tokens — pure passthrough. Claude Code calls
/// this to size a prompt before sending it. Free at Anthropic, so there is
/// nothing to reserve or reconcile; it still goes through RouterFuel auth
/// and rate limiting.
pub async fn count_tokens_handler(
    headers: HeaderMap,
    axum::extract::State(state): axum::extract::State<PassthroughState>,
    body: Bytes,
) -> Response {
    let rl_key = headers
        .get("x-routerfuel-client-id")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("anonymous")
        .to_string();

    if state.rate_limiter.check(&rl_key).is_err() {
        return json_error(
            StatusCode::TOO_MANY_REQUESTS,
            "rate_limit_error",
            "Rate limit exceeded".to_string(),
        );
    }

    let api_key = match require_anthropic_key(&headers) {
        Ok(k) => k,
        Err(resp) => return resp,
    };

    let rb = state
        .http
        .post(ANTHROPIC_COUNT_TOKENS_URL)
        .header("x-api-key", &api_key);
    let rb = forward_protocol_headers(rb, &headers);

    // Same bracketed-id exposure as /v1/messages: Anthropic 404s
    // `claude-opus-5[1m]` here too. If the body is unparseable, forward it
    // untouched and let Anthropic return the error — this endpoint has no
    // opinion of its own about the payload.
    let outbound = match serde_json::from_slice::<serde_json::Value>(&body) {
        Ok(parsed) => {
            let model = parsed
                .get("model")
                .and_then(|m| m.as_str())
                .map(|m| normalize_model_id(m).to_string())
                .unwrap_or_default();
            normalize_body_model(&body, &parsed, &model)
        }
        Err(_) => body,
    };

    match rb.body(outbound).send().await {
        Ok(r) => {
            let status =
                StatusCode::from_u16(r.status().as_u16()).unwrap_or(StatusCode::BAD_GATEWAY);
            match r.bytes().await {
                Ok(b) => passthrough_response(status, b, false),
                Err(e) => json_error(
                    StatusCode::BAD_GATEWAY,
                    "api_error",
                    format!("Failed reading count_tokens response: {e}"),
                ),
            }
        }
        Err(e) => json_error(
            StatusCode::BAD_GATEWAY,
            "api_error",
            format!("count_tokens request to Anthropic failed: {e}"),
        ),
    }
}

fn passthrough_response(status: StatusCode, body: Bytes, sse: bool) -> Response {
    let ct = if sse { "text/event-stream" } else { "application/json" };
    Response::builder()
        .status(status)
        .header("content-type", ct)
        .body(Body::from(body))
        .unwrap_or_else(|_| (StatusCode::INTERNAL_SERVER_ERROR, "response build failed").into_response())
}

/// Rewrites the body's `model` field to the normalized id when the client
/// sent a bracketed variant suffix.
///
/// Normalizing only for the registry lookup is NOT enough. Verified against
/// the live API: Anthropic rejects the bracketed form with
///
///     404 {"type":"error","error":{"type":"not_found_error",
///          "message":"model: claude-opus-5[1m]"}}
///
/// so the normalized id has to go on the wire as well. Returns the original
/// bytes untouched when nothing changed, which keeps the common path
/// byte-for-byte — the deviation from pure passthrough is exactly one field,
/// and only when the client used a form the provider will not accept.
fn normalize_body_model(body: &Bytes, parsed: &serde_json::Value, model: &str) -> Bytes {
    let raw = parsed.get("model").and_then(|m| m.as_str()).unwrap_or("");
    if raw == model {
        return body.clone();
    }
    let mut patched = parsed.clone();
    patched["model"] = serde_json::Value::String(model.to_string());
    match serde_json::to_vec(&patched) {
        Ok(v) => Bytes::from(v),
        // Re-serializing a Value we just parsed should not fail; if it
        // somehow does, sending the original is better than 500ing.
        Err(_) => body.clone(),
    }
}

/// Reads `usage.input_tokens` / `usage.output_tokens` from a non-streaming
/// Messages response.
fn usage_from_message(body: &[u8]) -> Option<(u32, u32)> {
    let v: serde_json::Value = serde_json::from_slice(body).ok()?;
    let u = v.get("usage")?;
    Some((
        u.get("input_tokens").and_then(|t| t.as_u64()).unwrap_or(0) as u32,
        u.get("output_tokens").and_then(|t| t.as_u64()).unwrap_or(0) as u32,
    ))
}

/// Relays the SSE stream to the client byte-for-byte while sniffing usage
/// out of it for reconciliation.
///
/// Raw byte relay rather than re-emitting through axum's Sse type: Anthropic
/// frames events as `event: <name>\ndata: {...}`, and rebuilding those would
/// risk dropping event names or reordering fields. Passthrough means
/// passthrough.
///
/// Usage arrives in two places — `message_start` carries input_tokens and
/// the final `message_delta` carries output_tokens — so both are tracked as
/// complete `data:` lines go past. A line buffer handles JSON split across
/// chunk boundaries.
#[allow(clippy::too_many_arguments)]
fn relay_stream(
    state: PassthroughState,
    upstream: reqwest::Response,
    status: StatusCode,
    request_id: String,
    rl_key: String,
    client_id: Option<String>,
    model: String,
    estimated: TokenCostBreakdown,
    rate_in: f64,
    rate_out: f64,
    start: Instant,
    provider_headers_ms: u64,
) -> Response {
    let mut upstream_stream = upstream.bytes_stream();

    let relayed = async_stream::stream! {
        let mut line_buf = String::new();
        let mut in_tok = 0u32;
        let mut out_tok = 0u32;

        while let Some(chunk) = upstream_stream.next().await {
            match chunk {
                Ok(bytes) => {
                    // Sniff a copy; the client always gets the original.
                    line_buf.push_str(&String::from_utf8_lossy(&bytes));
                    while let Some(nl) = line_buf.find('\n') {
                        let line: String = line_buf.drain(..=nl).collect();
                        if let Some(payload) = line.trim().strip_prefix("data:") {
                            if let Ok(v) = serde_json::from_str::<serde_json::Value>(payload.trim()) {
                                if let Some(u) = v.get("message").and_then(|m| m.get("usage")) {
                                    if let Some(t) = u.get("input_tokens").and_then(|t| t.as_u64()) {
                                        in_tok = t as u32;
                                    }
                                }
                                if let Some(u) = v.get("usage") {
                                    if let Some(t) = u.get("input_tokens").and_then(|t| t.as_u64()) {
                                        in_tok = t as u32;
                                    }
                                    if let Some(t) = u.get("output_tokens").and_then(|t| t.as_u64()) {
                                        out_tok = t as u32;
                                    }
                                }
                            }
                        }
                    }
                    yield Ok::<Bytes, std::io::Error>(bytes);
                }
                Err(e) => {
                    error!(request_id = %request_id, "Anthropic stream error: {e}");
                    break;
                }
            }
        }

        let actual = TokenCostBreakdown::new(in_tok, out_tok, rate_in, rate_out);
        state
            .spend_guard
            .reconcile(&rl_key, estimated.total_cost_cents, actual.total_cost_cents);
        debug!(
            request_id = %request_id,
            in_tok, out_tok,
            "native /v1/messages stream finished; spend reconciled"
        );
        log_passthrough(
            &state,
            request_id,
            model,
            &actual,
            client_id,
            provider_headers_ms,
            start.elapsed().as_millis() as u64,
        );
    };

    Response::builder()
        .status(status)
        .header("content-type", "text/event-stream")
        .header("cache-control", "no-cache")
        .body(Body::from_stream(relayed))
        .unwrap_or_else(|_| {
            (StatusCode::INTERNAL_SERVER_ERROR, "stream build failed").into_response()
        })
}

/// Writes the request_logs row. `routing_decision_ms` is 0 by construction:
/// this endpoint does no model selection, the client named the model.
fn log_passthrough(
    state: &PassthroughState,
    request_id: String,
    model: String,
    cost: &TokenCostBreakdown,
    client_id: Option<String>,
    // Provider round-trip only. On the streaming path this is time-to-
    // headers, since the body arrives incrementally over the whole
    // generation and lumping that in would repeat the mistake
    // streaming.rs currently makes.
    provider_ms: u64,
    // Full handler wall-clock.
    total_ms: u64,
) {
    state.cost_tracker.record_request(
        request_id,
        Provider::Anthropic,
        model,
        cost,
        // No baseline comparison on this path: the client named the model,
        // so there is no alternative RouterFuel declined to pick and no
        // "savings" figure that would mean anything.
        //
        // Passing the actual cost rather than 0.0 so cost_saved_cents comes
        // out as exactly zero. With 0.0 the stored saving is -cost, which
        // surfaced on the dashboard as a negative "total saved" — verified
        // live before this was corrected.
        cost.total_cost_cents,
        provider_ms,
        // routing_decision_ms is 0 by construction — this endpoint performs
        // no model selection.
        0,
        client_id,
        None,
        None,
        true,  // is_byok — always, this endpoint has no other mode
        false, // from_cache — no semantic cache on the native path
        Some(total_ms),
    );
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_text_reads_system_string_and_message_blocks() {
        let body = serde_json::json!({
            "model": "claude-opus-5",
            "system": "be terse",
            "messages": [
                { "role": "user", "content": "hello" },
                { "role": "assistant", "content": [ { "type": "text", "text": "hi there" } ] },
                { "role": "user", "content": [
                    { "type": "text", "text": "and this" },
                    { "type": "image", "source": { "type": "base64", "data": "AAAA" } }
                ]}
            ]
        });
        let t = extract_text(&body);
        assert!(t.contains("be terse"));
        assert!(t.contains("hello"));
        assert!(t.contains("hi there"));
        assert!(t.contains("and this"));
        // Non-text blocks contribute nothing rather than leaking base64.
        assert!(!t.contains("AAAA"));
    }

    #[test]
    fn extract_text_reads_system_block_array() {
        // The cache_control form Claude Code uses for its system prompt.
        let body = serde_json::json!({
            "system": [
                { "type": "text", "text": "you are a coding agent",
                  "cache_control": { "type": "ephemeral" } }
            ],
            "messages": []
        });
        assert!(extract_text(&body).contains("you are a coding agent"));
    }

    #[test]
    fn extract_text_tolerates_missing_fields() {
        assert_eq!(extract_text(&serde_json::json!({})), "");
        assert_eq!(extract_text(&serde_json::json!({ "messages": [] })), "");
        assert_eq!(
            extract_text(&serde_json::json!({ "messages": [ { "role": "user" } ] })),
            ""
        );
    }

    #[test]
    fn bracketed_model_is_rewritten_on_the_outbound_body() {
        // Regression guard for a bug only a live request caught: Anthropic
        // returns 404 not_found_error for `claude-opus-5[1m]`, so
        // normalizing purely for the registry lookup left the wire form
        // broken.
        let raw = br#"{"model":"claude-opus-5[1m]","max_tokens":16,"messages":[]}"#;
        let body = Bytes::from_static(raw);
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let out = normalize_body_model(&body, &parsed, "claude-opus-5");
        let reparsed: serde_json::Value = serde_json::from_slice(&out).unwrap();
        assert_eq!(reparsed["model"], "claude-opus-5");
        // Everything else survives the round-trip.
        assert_eq!(reparsed["max_tokens"], 16);
    }

    #[test]
    fn unbracketed_model_body_is_passed_through_byte_for_byte() {
        let raw = br#"{"model":"claude-opus-5","max_tokens":16,"messages":[]}"#;
        let body = Bytes::from_static(raw);
        let parsed: serde_json::Value = serde_json::from_slice(&body).unwrap();

        let out = normalize_body_model(&body, &parsed, "claude-opus-5");
        assert_eq!(
            out.as_ref(),
            raw.as_slice(),
            "the common path must not be re-serialized"
        );
    }

    #[test]
    fn usage_is_read_from_a_messages_response() {
        let body = br#"{"id":"msg_1","type":"message","role":"assistant",
            "content":[{"type":"text","text":"hi"}],
            "usage":{"input_tokens":123,"output_tokens":45}}"#;
        assert_eq!(usage_from_message(body), Some((123, 45)));
    }

    #[test]
    fn usage_is_none_for_an_error_envelope() {
        let body = br#"{"type":"error","error":{"type":"invalid_request_error","message":"bad"}}"#;
        assert_eq!(usage_from_message(body), None);
    }

    #[test]
    fn usage_defaults_missing_counters_to_zero() {
        let body = br#"{"usage":{"input_tokens":10}}"#;
        assert_eq!(usage_from_message(body), Some((10, 0)));
    }
}
