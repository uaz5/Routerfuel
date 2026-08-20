// =============================================================================
// src/streaming.rs  — RouterFuel v0.7
//
// FIX (this revision): the streaming path never touched SpendGuard at all
// — no check before firing, and (unlike the non-streaming path) no
// record/reconcile after, since the audit_tx task only ever fed
// cost_tracker. A client using only streaming requests accrued zero spend
// against their cap regardless of how much they actually used. stream_handler
// now takes spend_guard/rl_key/estimated_cost_cents; main.rs reserves before
// calling this, and the audit task here reconciles once real usage is known.
// On any early failure path (send error, non-2xx status) the reservation is
// released immediately rather than waiting for the audit task, since those
// paths never reach it.
// =============================================================================

use crate::concurrency::ConcurrencyLimiter;
use crate::connectors::{provider_base_url, to_gemini_body, ChatCompletionRequest, Provider};
use crate::cost_tracker::CostTracker;
use crate::guardrails::SpendGuard;
use crate::route_engine::RouteEngine;
use crate::tokens::TokenCostBreakdown;
use async_stream::try_stream;
use axum::response::sse::{Event, KeepAlive, Sse};
use futures_util::StreamExt;
use serde::Deserialize;
use std::convert::Infallible;
use std::sync::Arc;
use std::time::Instant;
use tokio::sync::mpsc;
use tokio_stream::Stream;
use tracing::{debug, error, instrument};

// SSE chunk types
#[derive(Debug, Deserialize)]
struct OpenAiStreamChunk {
    choices: Option<Vec<OpenAiStreamChoice>>,
    usage: Option<OpenAiStreamUsage>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamChoice {
    delta: OpenAiStreamDelta,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamDelta {
    content: Option<String>,
}

#[derive(Debug, Deserialize)]
struct OpenAiStreamUsage {
    prompt_tokens: Option<u32>,
    completion_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamEvent {
    #[serde(rename = "type")]
    event_type: String,
    delta: Option<AnthropicStreamDelta>,
    usage: Option<AnthropicStreamUsage>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamDelta {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct AnthropicStreamUsage {
    output_tokens: Option<u32>,
}

#[derive(Debug, Deserialize)]
struct GeminiStreamChunk {
    candidates: Option<Vec<GeminiStreamCandidate>>,
    #[serde(rename = "usageMetadata")]
    usage_metadata: Option<GeminiStreamUsage>,
}

#[derive(Debug, Deserialize)]
struct GeminiStreamCandidate {
    content: Option<GeminiStreamContent>,
}

#[derive(Debug, Deserialize)]
struct GeminiStreamContent {
    parts: Option<Vec<GeminiStreamPart>>,
}

#[derive(Debug, Deserialize)]
struct GeminiStreamPart {
    text: Option<String>,
}

#[derive(Debug, Deserialize)]
struct GeminiStreamUsage {
    #[serde(rename = "promptTokenCount")]
    prompt_token_count: Option<u32>,
    #[serde(rename = "candidatesTokenCount")]
    candidates_token_count: Option<u32>,
}

#[allow(clippy::too_many_arguments)]
#[instrument(skip(route_engine, cost_tracker, http_client, req, api_key, concurrency_limiter, spend_guard))]
pub async fn stream_handler(
    request_id: String,
    provider: Provider,
    model_api_id: String,
    api_key: String,
    req: ChatCompletionRequest,
    client_id: Option<String>,
    is_byok: bool,
    route_engine: Arc<RouteEngine>,
    cost_tracker: Arc<CostTracker>,
    http_client: reqwest::Client,
    concurrency_limiter: Arc<ConcurrencyLimiter>,
    // FIX: new params — spend_guard + the reservation key/amount main.rs
    // already reserved before calling this function.
    spend_guard: Arc<SpendGuard>,
    rl_key: String,
    estimated_cost_cents: f64,
) -> Sse<impl Stream<Item = Result<Event, Infallible>>> {
    let start = Instant::now();

    let (audit_tx, mut audit_rx) = mpsc::channel::<AuditPayload>(1);

    let cost_tracker_clone = Arc::clone(&cost_tracker);
    let route_engine_clone = Arc::clone(&route_engine);
    let request_id_clone = request_id.clone();
    let model_api_id_clone = model_api_id.clone();
    let client_id_clone = client_id.clone();
    let spend_guard_clone = Arc::clone(&spend_guard);
    let rl_key_clone = rl_key.clone();

    tokio::spawn(async move {
        if let Some(payload) = audit_rx.recv().await {
            let (cost_in, cost_out) = route_engine_clone
                .get_pricing(&model_api_id_clone)
                .unwrap_or((500.0, 3000.0));

            let token_cost = TokenCostBreakdown::new(
                payload.input_tokens,
                payload.output_tokens,
                cost_in,
                cost_out,
            );
            let baseline_cost =
                TokenCostBreakdown::new(payload.input_tokens, payload.output_tokens, 500.0, 3000.0);

            // FIX: reconcile the reservation main.rs made before this
            // stream started against the real cost now known — this is the
            // only place a streaming request's spend ever gets recorded.
            spend_guard_clone.reconcile(&rl_key_clone, estimated_cost_cents, token_cost.total_cost_cents);

            cost_tracker_clone.record_request(
                request_id_clone,
                payload.provider,
                model_api_id_clone,
                &token_cost,
                baseline_cost.total_cost_cents,
                payload.latency_ms,
                0,
                client_id_clone,
                None,
                Some("streaming".to_string()),
                is_byok,
                false, // streaming never serves from the semantic cache
            );
        } else {
            // FIX: audit_tx was dropped without ever sending — the stream
            // ended (or errored) before reaching any of the send points
            // below. Release the reservation rather than leaving it stuck
            // against the client's cap forever.
            spend_guard_clone.release(&rl_key_clone, estimated_cost_cents);
        }
    });

    let (url, body): (String, serde_json::Value) = match provider {
        Provider::Anthropic => {
            let mut body = serde_json::json!({
                "model": req.model,
                "messages": req.messages,
                "max_tokens": req.max_tokens.unwrap_or(1024),
                "stream": true,
            });
            if let Some(t) = req.temperature {
                body["temperature"] = serde_json::json!(t);
            }
            (provider_base_url(Provider::Anthropic).to_string(), body)
        }
        Provider::Gemini => {
            let body = to_gemini_body(&req);
            // FIX: key moved out of the URL and into the x-goog-api-key
            // header below (see connectors.rs's GeminiConnector for the
            // matching non-streaming fix and the full rationale) — a
            // secret embedded in a URL is far more exposure-prone than
            // one carried in a header.
            let url = format!(
                "{}/{}:streamGenerateContent?alt=sse",
                provider_base_url(Provider::Gemini),
                req.model
            );
            (url, body)
        }
        Provider::AzureOpenAI => {
            // Azure OpenAI streaming uses the same OpenAI-compatible format
            // but with a deployment-specific URL constructed from the connection string.
            // The api_key contains the full connection string; we parse it here.
            let (endpoint, _key) = parse_azure_connection_for_streaming(&api_key);
            let url = format!(
                "{}/openai/deployments/{}/chat/completions?api-version=2024-02-15-preview",
                endpoint.trim_end_matches('/'),
                req.model
            );
            let mut body = serde_json::to_value(&req).unwrap_or_default();
            body["stream"] = serde_json::Value::Bool(true);
            (url, body)
        }
        Provider::Bedrock => {
            // Bedrock streaming uses the InvokeModelWithResponseStream API
            let (region, _access_key, _secret_key, _session_token) = parse_bedrock_connection_for_streaming(&api_key);
            let url = format!(
                "https://bedrock-runtime.{}.amazonaws.com/model/{}/invoke-with-response-stream",
                region,
                req.model
            );
            let mut body = serde_json::to_value(&req).unwrap_or_default();
            body["stream"] = serde_json::Value::Bool(true);
            (url, body)
        }
        _ => {
            let mut body = serde_json::to_value(&req).unwrap_or_default();
            body["stream"] = serde_json::Value::Bool(true);
            (provider_base_url(provider).to_string(), body)
        }
    };

    let is_gemini = matches!(provider, Provider::Gemini);
    let is_anthropic = matches!(provider, Provider::Anthropic);
    let is_azure = matches!(provider, Provider::AzureOpenAI);
    let is_bedrock = matches!(provider, Provider::Bedrock);

    let http_req = if is_gemini {
        http_client
            .post(&url)
            .header("x-goog-api-key", &api_key)
            .header("content-type", "application/json")
            .json(&body)
    } else if is_anthropic {
        http_client
            .post(&url)
            .header("x-api-key", &api_key)
            .header("anthropic-version", "2023-06-01")
            .header("content-type", "application/json")
            .json(&body)
    } else if is_azure {
        let (_endpoint, key) = parse_azure_connection_for_streaming(&api_key);
        let auth_header = format!("api-key {}", key);
        http_client
            .post(&url)
            .header("api-key", &auth_header)
            .header("content-type", "application/json")
            .json(&body)
    } else if is_bedrock {
        let (_region, access_key, secret_key, session_token) = parse_bedrock_connection_for_streaming(&api_key);
        let mut builder = http_client
            .post(&url)
            .header("content-type", "application/json")
            .header("x-amz-access-key", &access_key)
            .header("x-amz-secret-key", &secret_key);
        if let Some(token) = &session_token {
            builder = builder.header("x-amz-security-token", token);
        }
        builder.json(&body)
    } else {
        http_client.post(&url).bearer_auth(&api_key).json(&body)
    };

    let stream = try_stream! {
        let _permit = concurrency_limiter.acquire().await;

        let response = match http_req.send().await {
            Ok(r) => r,
            Err(e) => {
                error!("Streaming request failed: {}", e);
                // audit_tx drops here without sending -> the audit task's
                // else-branch above releases the reservation.
                yield Event::default().data(format!(r#"{{"error":"upstream failed: {e}"}}"#));
                return;
            }
        };

        if !response.status().is_success() {
            let status = response.status().as_u16();
            let body_text = response.text().await.unwrap_or_default();
            error!(status = status, body = %body_text, "Provider returned error");
            // Same as above — audit_tx drops unsent, reservation released.
            yield Event::default().data(format!(r#"{{"error":"provider error {status}"}}"#));
            return;
        }

        let mut byte_stream = response.bytes_stream();

        let mut full_text = String::new();
        let mut input_tokens = 0u32;
        let mut output_tokens = 0u32;
        let mut chunk_count = 0u32;

        while let Some(chunk) = byte_stream.next().await {
            let bytes = match chunk {
                Ok(b) => b,
                Err(e) => {
                    error!("Stream read error: {}", e);
                    break;
                }
            };

            let text = match String::from_utf8(bytes.to_vec()) {
                Ok(t) => t,
                Err(_) => continue,
            };

            for line in text.lines() {
                if line.is_empty() || line.starts_with(':') { continue; }

                let data = if let Some(d) = line.strip_prefix("data: ") {
                    d.trim()
                } else {
                    continue;
                };

                if data == "[DONE]" {
                    debug!(chunks = chunk_count, chars = full_text.len(), "Stream complete");
                    let _ = audit_tx.send(AuditPayload {
                        provider, input_tokens, output_tokens,
                        latency_ms: start.elapsed().as_millis() as u64,
                    }).await;
                    yield Event::default().data("[DONE]");
                    return;
                }

                if is_anthropic {
                    if let Ok(event) = serde_json::from_str::<AnthropicStreamEvent>(data) {
                        if let Some(delta) = &event.delta {
                            if let Some(t) = &delta.text {
                                full_text.push_str(t);
                                chunk_count += 1;
                            }
                        }
                        if let Some(usage) = &event.usage {
                            if let Some(ot) = usage.output_tokens { output_tokens = ot; }
                        }
                        if event.event_type == "message_stop" {
                            debug!(chunks = chunk_count, "Anthropic stream complete");
                            let _ = audit_tx.send(AuditPayload {
                                provider, input_tokens, output_tokens,
                                latency_ms: start.elapsed().as_millis() as u64,
                            }).await;
                            yield Event::default().data("[DONE]");
                            return;
                        }
                    }
                    yield Event::default().data(data);
                } else if is_gemini {
                    if let Ok(chunk) = serde_json::from_str::<GeminiStreamChunk>(data) {
                        if let Some(candidates) = &chunk.candidates {
                            for c in candidates {
                                if let Some(content) = &c.content {
                                    if let Some(parts) = &content.parts {
                                        for p in parts {
                                            if let Some(t) = &p.text {
                                                full_text.push_str(t);
                                                chunk_count += 1;
                                            }
                                        }
                                    }
                                }
                            }
                        }
                        if let Some(usage) = &chunk.usage_metadata {
                            if let Some(pt) = usage.prompt_token_count { input_tokens = pt; }
                            if let Some(ct) = usage.candidates_token_count { output_tokens = ct; }
                        }
                    }
                    yield Event::default().data(data);
                } else if is_azure || is_bedrock {
                    // Azure and Bedrock both use OpenAI-compatible streaming format
                    if let Ok(chunk) = serde_json::from_str::<OpenAiStreamChunk>(data) {
                        if let Some(choices) = &chunk.choices {
                            for choice in choices {
                                if let Some(content) = &choice.delta.content {
                                    full_text.push_str(content);
                                    chunk_count += 1;
                                }
                            }
                        }
                        if let Some(usage) = &chunk.usage {
                            if let Some(pt) = usage.prompt_tokens { input_tokens = pt; }
                            if let Some(ct) = usage.completion_tokens { output_tokens = ct; }
                        }
                    }
                    yield Event::default().data(data);
                } else {
                    if let Ok(chunk) = serde_json::from_str::<OpenAiStreamChunk>(data) {
                        if let Some(choices) = &chunk.choices {
                            for choice in choices {
                                if let Some(content) = &choice.delta.content {
                                    full_text.push_str(content);
                                    chunk_count += 1;
                                }
                            }
                        }
                        if let Some(usage) = &chunk.usage {
                            if let Some(pt) = usage.prompt_tokens { input_tokens = pt; }
                            if let Some(ct) = usage.completion_tokens { output_tokens = ct; }
                        }
                    }
                    yield Event::default().data(data);
                }
            }
        }

        // Loop ended without a [DONE]/message_stop (e.g. a mid-stream read
        // error via `break` above) — still send whatever was accumulated so
        // the reservation gets reconciled to the real (possibly partial)
        // cost instead of the audit task's release-on-drop path treating a
        // partially-successful stream as a total failure.
        let _ = audit_tx.send(AuditPayload {
            provider, input_tokens, output_tokens,
            latency_ms: start.elapsed().as_millis() as u64,
        }).await;
    };

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("keep-alive"),
    )
}

#[derive(Debug)]
struct AuditPayload {
    provider: Provider,
    input_tokens: u32,
    output_tokens: u32,
    latency_ms: u64,
}

/// Parse Azure connection string for streaming (returns endpoint and api-key value)
fn parse_azure_connection_for_streaming(conn_str: &str) -> (String, String) {
    let mut endpoint = String::new();
    let mut key = String::new();

    for part in conn_str.split(';') {
        let part = part.trim();
        if let Some((k, v)) = part.split_once('=') {
            match k.trim().to_lowercase().as_str() {
                "endpoint" => endpoint = v.trim().to_string(),
                "key" => key = v.trim().to_string(),
                _ => {}
            }
        }
    }

    (endpoint, key)
}

/// Parse Bedrock connection string for streaming (returns region and credentials)
fn parse_bedrock_connection_for_streaming(conn_str: &str) -> (String, String, String, Option<String>) {
    let mut region = String::new();
    let mut access_key = String::new();
    let mut secret_key = String::new();
    let mut session_token = None;

    for part in conn_str.split(';') {
        let part = part.trim();
        if let Some((k, v)) = part.split_once('=') {
            match k.trim().to_lowercase().as_str() {
                "region" => region = v.trim().to_string(),
                "access_key" => access_key = v.trim().to_string(),
                "secret_key" => secret_key = v.trim().to_string(),
                "session_token" => session_token = Some(v.trim().to_string()),
                _ => {}
            }
        }
    }

    (region, access_key, secret_key, session_token)
}
