// =============================================================================
// src/openrouter_catalog.rs — RouterFuel v0.7
//
// GET https://openrouter.ai/api/v1/models is public — no API key required to
// list it, only to actually call a model. RouterFuel fetches it once at
// startup and merges every entry into the registry as a `Provider::OpenRouter`
// model, so:
//   - "100+ LLMs via OpenRouter" stays true automatically as OpenRouter adds
//     models, instead of RouterFuel maintaining a second hardcoded list that
//     goes stale the day after someone writes it.
//   - A client who only has an OpenRouter key (very common — see main.rs's
//     BYOK fallback logic) can route to anything OpenRouter carries, not
//     just the ~40 models route_engine.rs curates with hand-tuned
//     cost/latency/quality figures for its own direct integrations.
//
// Curated direct-integration entries in route_engine.rs always win on id
// collisions (RouteEngine::extend_registry skips any id already present) —
// this catalog only *adds* coverage, it never overrides a hand-tuned entry.
// =============================================================================

use crate::connectors::Provider;
use crate::route_engine::ModelConfig;
use anyhow::{Context, Result};
use serde::Deserialize;
use tracing::debug;

const OPENROUTER_MODELS_URL: &str = "https://openrouter.ai/api/v1/models";

/// Neutral defaults for catalog entries we have no first-party benchmark
/// data for. Deliberately mediocre-but-not-terrible so a hand-tuned direct
/// integration in route_engine.rs still wins routing ties against its own
/// OpenRouter-catalog duplicate would, if one existed (it won't — dedup
/// happens by exact id in RouteEngine::extend_registry).
const DEFAULT_LATENCY_MS: u64 = 250;
const DEFAULT_QUALITY_SCORE: f32 = 0.72;
const DEFAULT_CONTEXT_WINDOW: u32 = 32_000;

#[derive(Debug, Deserialize)]
struct OpenRouterModelsResponse {
    data: Vec<OpenRouterModel>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterModel {
    id: String,
    name: Option<String>,
    context_length: Option<u32>,
    architecture: Option<OpenRouterArchitecture>,
    pricing: Option<OpenRouterPricing>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterArchitecture {
    input_modalities: Option<Vec<String>>,
}

#[derive(Debug, Deserialize)]
struct OpenRouterPricing {
    /// USD per single token (not per 1M) — OpenRouter's convention.
    prompt: Option<String>,
    completion: Option<String>,
}

/// Fetch and translate the full public OpenRouter catalog into RouterFuel's
/// `ModelConfig` shape. Returns an error (never panics) on any network or
/// parse failure — callers should treat that as non-fatal, since the
/// curated registry in route_engine.rs works fine on its own.
pub async fn fetch_openrouter_catalog(client: &reqwest::Client) -> Result<Vec<ModelConfig>> {
    let resp = client
        .get(OPENROUTER_MODELS_URL)
        .send()
        .await
        .context("failed to reach OpenRouter's /models endpoint")?
        .error_for_status()
        .context("OpenRouter's /models endpoint returned an error status")?;

    let parsed: OpenRouterModelsResponse = resp
        .json()
        .await
        .context("failed to parse OpenRouter's /models response")?;

    let models = parsed.data.into_iter().filter_map(to_model_config).collect::<Vec<_>>();

    debug!(count = models.len(), "Translated OpenRouter catalog into RouterFuel ModelConfig entries");

    Ok(models)
}

fn to_model_config(m: OpenRouterModel) -> Option<ModelConfig> {
    if m.id.trim().is_empty() {
        return None;
    }

    let pricing = m.pricing.unwrap_or(OpenRouterPricing { prompt: None, completion: None });

    let cost_per_1m_input = pricing
        .prompt
        .as_deref()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|per_token_usd| per_token_usd * 1_000_000.0 * 100.0)
        .unwrap_or(0.0);

    let cost_per_1m_output = pricing
        .completion
        .as_deref()
        .and_then(|s| s.parse::<f64>().ok())
        .map(|per_token_usd| per_token_usd * 1_000_000.0 * 100.0)
        .unwrap_or(0.0);

    let supports_vision = m
        .architecture
        .as_ref()
        .and_then(|a| a.input_modalities.as_ref())
        .map(|modalities| modalities.iter().any(|mo| mo == "image"))
        .unwrap_or(false);

    let display_name = format!("{} (via OpenRouter)", m.name.unwrap_or_else(|| m.id.clone()));

    Some(ModelConfig {
        api_id: m.id,
        display_name,
        provider: Provider::OpenRouter,
        cost_per_1m_input,
        cost_per_1m_output,
        latency_ms: DEFAULT_LATENCY_MS,
        quality_score: DEFAULT_QUALITY_SCORE,
        context_window: m.context_length.unwrap_or(DEFAULT_CONTEXT_WINDOW),
        supports_vision,
        // Unknown from this endpoint alone — OpenRouter doesn't expose a
        // simple open/closed-weight flag in /models. Default conservatively
        // to false rather than guess; it only affects display/filtering,
        // not routing eligibility.
        open_weight: false,
        enabled: true,
        // OpenRouter's /models exposes flat per-token rates only, with no
        // long-context break, so there is nothing to populate here.
        long_context_tier: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn converts_pricing_from_per_token_to_per_million_cents() {
        let m = OpenRouterModel {
            id: "openai/gpt-4o".into(),
            name: Some("OpenAI: GPT-4o".into()),
            context_length: Some(128_000),
            architecture: Some(OpenRouterArchitecture { input_modalities: Some(vec!["text".into(), "image".into()]) }),
            pricing: Some(OpenRouterPricing { prompt: Some("0.0000025".into()), completion: Some("0.00001".into()) }),
        };
        let cfg = to_model_config(m).unwrap();
        assert_eq!(cfg.provider, Provider::OpenRouter);
        assert!((cfg.cost_per_1m_input - 250.0).abs() < 0.001); // $0.0000025 * 1e6 * 100 = 250 cents
        assert!((cfg.cost_per_1m_output - 1000.0).abs() < 0.001);
        assert!(cfg.supports_vision);
    }

    #[test]
    fn missing_id_is_skipped() {
        let m = OpenRouterModel { id: "".into(), name: None, context_length: None, architecture: None, pricing: None };
        assert!(to_model_config(m).is_none());
    }
}
