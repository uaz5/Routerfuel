// ============================================================================
// src/route_engine.rs — RouterFuel v0.6 — accurate as of July 24, 2026
//
// Registry covers every major lab RouterFuel supports, direct + via OpenRouter:
//   Anthropic, OpenAI, Google Gemini, xAI (Grok), DeepSeek (incl. open-weight),
//   Mistral, Alibaba Qwen, Moonshot (Kimi), Zhipu (GLM), Meta (Llama).
//
// Pricing/context/latency figures are the best public numbers available at
// write time and drift constantly — treat cost_per_1m_* as the routing
// signal it is, not a billing source of truth. RouterFuel is pure BYOK: it
// never pays a provider bill itself, so a stale price here only skews which
// model gets picked, not what anyone is actually charged (providers bill the
// client's own key directly).
//
// FIX (this revision): the `impl RouteEngine` block used to close early
// (right after `openrouter_catalog_has`), which left build_registry(),
// select(), select_for_task(), extend_registry(), find(),
// select_provider(), get_pricing(), list_enabled(), list_vision_capable(),
// is_vision_capable(), and set_enabled() as free-standing module functions
// with an illegal bare `&self` param — this did not compile. All of those
// are now back inside one single `impl RouteEngine` block.
//
// FIX (this revision): added `select_reachable()`, which main.rs's
// resolve_model() and vision.rs's select_vision_model() already called but
// which never existed in this file — the BYOK-reachability filter (only
// route "auto"/"task:" requests to providers the client actually supplied a
// key for) was documented via comments in main.rs but not implemented.
// `select_for_task` also gained the same `reachable` parameter so task
// routing respects it too, both on its "preferred model" fast path and its
// scored fallback.
//
// FIX (this revision): Azure and Bedrock are pure BYOK — the client's key
// arrives per-request via headers, not at server startup. The static
// registry cannot pre-populate their models. When a request includes an
// X-Azure-OpenAI-Connection or X-Bedrock-Connection header, the route
// engine now treats the model name as valid and routes it directly to the
// corresponding connector, bypassing the normal "is this model in the
// registry" check for those two providers.
//
// FIX (this revision): the BYOK fallback previously required the model name
// to start with "azure/" or "bedrock/". That meant a request with model
// "DeepSeek-V4-Pro" and an X-Azure-OpenAI-Connection header was still
// rejected because the model name didn't have the prefix. Now the presence
// of the header alone is sufficient — any model name is accepted for Azure
// or Bedrock when the corresponding connection header is present.
// ============================================================================

use crate::connectors::Provider;
use anyhow::{anyhow, Result};
use parking_lot::RwLock;
use std::collections::HashSet;
use tracing::{debug, info, instrument};

#[derive(Debug, Clone)]
pub struct ModelConfig {
    /// The exact string you pass as the model field in API requests
    pub api_id: String,
    pub display_name: String,
    pub provider: Provider,
    /// Cost in CENTS per 1 million input tokens
    pub cost_per_1m_input: f64,
    /// Cost in CENTS per 1 million output tokens
    pub cost_per_1m_output: f64,
    /// Typical median latency ms (real-world, not marketing)
    pub latency_ms: u64,
    /// Subjective 0.0–1.0 quality score used in routing math
    pub quality_score: f32,
    /// Maximum input context tokens
    pub context_window: u32,
    /// Whether this model accepts image inputs (see src/vision.rs)
    pub supports_vision: bool,
    /// Open-weight model — can be self-hosted, not tied to one lab's API
    pub open_weight: bool,
    pub enabled: bool,
}

#[derive(Debug, Clone, Copy)]
pub enum RoutingPriority {
    Cost,
    Balanced,
    Quality,
    Speed,
}

#[derive(Debug, Clone)]
pub struct RoutingDecision {
    pub model: ModelConfig,
    pub score: f64,
    pub reason: String,
}

/// Tasks your meeting-assistant product exposes to callers.
/// Callers send `"task": "summarise"` — RouterFuel picks the model.
#[derive(Debug, Clone, Copy)]
pub enum MeetingTask {
    Summarise,
    AnswerQuestion,
    ExtractActionItems,
    DraftResponse,
    Classify,
}

impl std::str::FromStr for MeetingTask {
    type Err = anyhow::Error;
    fn from_str(s: &str) -> Result<Self> {
        match s {
            "summarise" | "summarize"    => Ok(Self::Summarise),
            "answer_question" | "qa"     => Ok(Self::AnswerQuestion),
            "extract_action_items"       => Ok(Self::ExtractActionItems),
            "draft_response" | "draft"   => Ok(Self::DraftResponse),
            "classify"                   => Ok(Self::Classify),
            _ => Err(anyhow!("Unknown task '{}'. Valid: summarise, answer_question, extract_action_items, draft_response, classify", s)),
        }
    }
}

pub struct RouteEngine {
    models: RwLock<Vec<ModelConfig>>,
}

macro_rules! model {
    (
        api_id: $api_id:expr,
        display_name: $display_name:expr,
        provider: $provider:expr,
        cost_in: $cost_in:expr,
        cost_out: $cost_out:expr,
        latency_ms: $latency_ms:expr,
        quality: $quality:expr,
        context: $context:expr,
        vision: $vision:expr,
        open_weight: $open_weight:expr,
        enabled: $enabled:expr $(,)?
    ) => {
        ModelConfig {
            api_id: $api_id.into(),
            display_name: $display_name.into(),
            provider: $provider,
            cost_per_1m_input: $cost_in,
            cost_per_1m_output: $cost_out,
            latency_ms: $latency_ms,
            quality_score: $quality,
            context_window: $context,
            supports_vision: $vision,
            open_weight: $open_weight,
            enabled: $enabled,
        }
    };
}

impl RouteEngine {
    pub fn new() -> Self {
        Self {
            models: RwLock::new(Self::build_registry()),
        }
    }

    /// Returns true if `candidate` exists as a literal id in the registry
    /// under Provider::OpenRouter — i.e. it's a real slug OpenRouter's own
    /// catalog reported at startup (see openrouter_catalog.rs's
    /// extend_registry call in main.rs), not just an assumed
    /// "{prefix}/{model}" formula. resolve_byok_route in main.rs uses this
    /// to verify a guessed OpenRouter slug before sending it, instead of
    /// trusting the formula blindly — this is the same class of bug that
    /// broke gemini-3-flash (OpenRouter's real slug needed a "-preview"
    /// suffix the formula didn't produce), just generalized into a check
    /// instead of a single hand-added exception.
    pub fn openrouter_catalog_has(&self, candidate: &str) -> bool {
        self.models
            .read()
            .iter()
            .any(|m| m.provider == Provider::OpenRouter && m.api_id == candidate)
    }

    /// Master registry — every model RouterFuel knows how to route to,
    /// as of July 2026. Cost figures are USD cents per 1M tokens.
    fn build_registry() -> Vec<ModelConfig> {
        vec![
            // ================================================================
            // ANTHROPIC — POST https://api.anthropic.com/v1/messages
            // Headers: x-api-key, anthropic-version: 2023-06-01
            // ================================================================
            model!(api_id: "claude-opus-5", display_name: "Claude Opus 5", provider: Provider::Anthropic,
                cost_in: 500.0, cost_out: 2500.0, latency_ms: 140, quality: 0.98, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "claude-opus-4-8", display_name: "Claude Opus 4.8", provider: Provider::Anthropic,
                cost_in: 500.0, cost_out: 2500.0, latency_ms: 260, quality: 0.93, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "claude-sonnet-5", display_name: "Claude Sonnet 5", provider: Provider::Anthropic,
                cost_in: 300.0, cost_out: 1500.0, latency_ms: 170, quality: 0.94, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "claude-haiku-4-5", display_name: "Claude Haiku 4.5", provider: Provider::Anthropic,
                cost_in: 100.0, cost_out: 500.0, latency_ms: 90, quality: 0.80, context: 200_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "claude-fable-5", display_name: "Claude Fable 5", provider: Provider::Anthropic,
                cost_in: 1000.0, cost_out: 5000.0, latency_ms: 320, quality: 0.99, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "claude-opus-4-7", display_name: "Claude Opus 4.7", provider: Provider::Anthropic,
                cost_in: 500.0, cost_out: 2500.0, latency_ms: 270, quality: 0.97, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "claude-opus-4-6", display_name: "Claude Opus 4.6", provider: Provider::Anthropic,
                cost_in: 500.0, cost_out: 2500.0, latency_ms: 260, quality: 0.96, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "claude-sonnet-4-6", display_name: "Claude Sonnet 4.6", provider: Provider::Anthropic,
                cost_in: 300.0, cost_out: 1500.0, latency_ms: 170, quality: 0.91, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            // ================================================================
            // OPENAI — POST https://api.openai.com/v1/chat/completions
            // Header: Authorization: Bearer <key>
            // ================================================================
            model!(api_id: "gpt-5.6-sol", display_name: "GPT-5.6 Sol", provider: Provider::OpenAI,
                cost_in: 500.0, cost_out: 3000.0, latency_ms: 250, quality: 0.99, context: 1_050_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gpt-5.6-terra", display_name: "GPT-5.6 Terra", provider: Provider::OpenAI,
                cost_in: 250.0, cost_out: 1500.0, latency_ms: 180, quality: 0.92, context: 1_050_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gpt-5.6-luna", display_name: "GPT-5.6 Luna", provider: Provider::OpenAI,
                cost_in: 100.0, cost_out: 600.0, latency_ms: 110, quality: 0.80, context: 1_050_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gpt-5.5", display_name: "GPT-5.5", provider: Provider::OpenAI,
                cost_in: 500.0, cost_out: 3000.0, latency_ms: 240, quality: 0.97, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gpt-5.4", display_name: "GPT-5.4", provider: Provider::OpenAI,
                cost_in: 250.0, cost_out: 1500.0, latency_ms: 175, quality: 0.90, context: 400_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gpt-5.4-mini", display_name: "GPT-5.4 Mini", provider: Provider::OpenAI,
                cost_in: 75.0, cost_out: 450.0, latency_ms: 130, quality: 0.78, context: 400_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gpt-5.4-nano", display_name: "GPT-5.4 Nano", provider: Provider::OpenAI,
                cost_in: 20.0, cost_out: 125.0, latency_ms: 85, quality: 0.65, context: 400_000,
                vision: false, open_weight: false, enabled: true),

            model!(api_id: "gpt-oss-20b", display_name: "GPT-OSS 20B (open weight)", provider: Provider::OpenAI,
                cost_in: 3.0, cost_out: 15.0, latency_ms: 95, quality: 0.62, context: 128_000,
                vision: false, open_weight: true, enabled: false), // enable once self-hosted/OpenRouter route configured

            // ================================================================
            // GOOGLE GEMINI — POST https://generativelanguage.googleapis.com/
            //   v1beta/models/{model_id}:generateContent?key={API_KEY}
            // ================================================================
            model!(api_id: "gemini-3.1-pro", display_name: "Gemini 3.1 Pro", provider: Provider::Gemini,
                cost_in: 200.0, cost_out: 1200.0, latency_ms: 210, quality: 0.96, context: 2_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gemini-3.5-flash", display_name: "Gemini 3.5 Flash", provider: Provider::Gemini,
                cost_in: 75.0, cost_out: 450.0, latency_ms: 120, quality: 0.87, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gemini-3-flash", display_name: "Gemini 3 Flash", provider: Provider::Gemini,
                cost_in: 50.0, cost_out: 300.0, latency_ms: 105, quality: 0.85, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gemini-3.1-flash-lite", display_name: "Gemini 3.1 Flash-Lite", provider: Provider::Gemini,
                cost_in: 10.0, cost_out: 40.0, latency_ms: 70, quality: 0.68, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gemini-2.5-pro", display_name: "Gemini 2.5 Pro", provider: Provider::Gemini,
                cost_in: 125.0, cost_out: 1000.0, latency_ms: 200, quality: 0.89, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gemini-2.5-flash-lite", display_name: "Gemini 2.5 Flash-Lite", provider: Provider::Gemini,
                cost_in: 10.0, cost_out: 40.0, latency_ms: 65, quality: 0.66, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            // ================================================================
            // xAI GROK — POST https://api.x.ai/v1/chat/completions
            // OpenAI-compatible schema
            // ================================================================
            model!(api_id: "grok-4.5", display_name: "Grok 4.5", provider: Provider::XAI,
                cost_in: 200.0, cost_out: 600.0, latency_ms: 190, quality: 0.93, context: 2_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "grok-4.3", display_name: "Grok 4.3", provider: Provider::XAI,
                cost_in: 125.0, cost_out: 250.0, latency_ms: 160, quality: 0.88, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "grok-4.20", display_name: "Grok 4.20", provider: Provider::XAI,
                cost_in: 125.0, cost_out: 250.0, latency_ms: 165, quality: 0.88, context: 2_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "grok-4.1-fast", display_name: "Grok 4.1 Fast", provider: Provider::XAI,
                cost_in: 20.0, cost_out: 50.0, latency_ms: 90, quality: 0.75, context: 2_000_000,
                vision: false, open_weight: false, enabled: true),

            model!(api_id: "grok-code-fast-1", display_name: "Grok Code Fast 1", provider: Provider::XAI,
                cost_in: 20.0, cost_out: 150.0, latency_ms: 100, quality: 0.78, context: 256_000,
                vision: false, open_weight: false, enabled: true),

            // ================================================================
            // DEEPSEEK — POST https://api.deepseek.com/v1/chat/completions
            // OpenAI-compatible schema — includes open-weight releases
            // ================================================================
            model!(api_id: "deepseek-v4-flash", display_name: "DeepSeek V4 Flash", provider: Provider::DeepSeek,
                cost_in: 14.0, cost_out: 28.0, latency_ms: 140, quality: 0.85, context: 1_000_000,
                vision: false, open_weight: true, enabled: true),

            model!(api_id: "deepseek-v4-pro", display_name: "DeepSeek V4 Pro", provider: Provider::DeepSeek,
                cost_in: 43.5, cost_out: 87.0, latency_ms: 185, quality: 0.91, context: 1_000_000,
                vision: false, open_weight: true, enabled: true),

            model!(api_id: "deepseek-v3.2", display_name: "DeepSeek V3.2 (legacy)", provider: Provider::DeepSeek,
                cost_in: 12.0, cost_out: 24.0, latency_ms: 150, quality: 0.80, context: 128_000,
                vision: false, open_weight: true, enabled: true),

            // ================================================================
            // MISTRAL — POST https://api.mistral.ai/v1/chat/completions
            // OpenAI-compatible schema
            // ================================================================
            model!(api_id: "mistral-large-3", display_name: "Mistral Large 3", provider: Provider::Mistral,
                cost_in: 50.0, cost_out: 150.0, latency_ms: 165, quality: 0.86, context: 128_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "mistral-small-4", display_name: "Mistral Small 4", provider: Provider::Mistral,
                cost_in: 10.0, cost_out: 30.0, latency_ms: 100, quality: 0.72, context: 128_000,
                vision: true, open_weight: true, enabled: true),

            model!(api_id: "codestral-2", display_name: "Codestral 2", provider: Provider::Mistral,
                cost_in: 30.0, cost_out: 90.0, latency_ms: 110, quality: 0.79, context: 256_000,
                vision: false, open_weight: false, enabled: true),

            model!(api_id: "ministral-8b", display_name: "Ministral 8B", provider: Provider::Mistral,
                cost_in: 10.0, cost_out: 10.0, latency_ms: 60, quality: 0.58, context: 128_000,
                vision: false, open_weight: true, enabled: true),

            // ================================================================
            // ALIBABA QWEN — POST https://dashscope-intl.aliyuncs.com/
            //   compatible-mode/v1/chat/completions  (OpenAI-compatible mode)
            // ================================================================
            model!(api_id: "qwen3-max", display_name: "Qwen3 Max", provider: Provider::Qwen,
                cost_in: 78.0, cost_out: 390.0, latency_ms: 200, quality: 0.90, context: 262_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "qwen3-235b-a22b", display_name: "Qwen3-235B-A22B", provider: Provider::Qwen,
                cost_in: 70.0, cost_out: 280.0, latency_ms: 190, quality: 0.87, context: 262_000,
                vision: false, open_weight: true, enabled: true),

            model!(api_id: "qwen-turbo", display_name: "Qwen Turbo", provider: Provider::Qwen,
                cost_in: 5.0, cost_out: 20.0, latency_ms: 80, quality: 0.62, context: 1_000_000,
                vision: false, open_weight: true, enabled: true),

            // ================================================================
            // MOONSHOT / KIMI — POST https://api.moonshot.ai/v1/chat/completions
            // OpenAI-compatible schema — open-weight releases
            // ================================================================
            model!(api_id: "kimi-k3", display_name: "Kimi K3", provider: Provider::Moonshot,
                cost_in: 300.0, cost_out: 1500.0, latency_ms: 230, quality: 0.97, context: 1_048_576,
                vision: true, open_weight: true, enabled: true),

            model!(api_id: "kimi-k2.6", display_name: "Kimi K2.6", provider: Provider::Moonshot,
                cost_in: 95.0, cost_out: 400.0, latency_ms: 175, quality: 0.89, context: 256_000,
                vision: true, open_weight: true, enabled: true),

            model!(api_id: "kimi-k2.5", display_name: "Kimi K2.5", provider: Provider::Moonshot,
                cost_in: 60.0, cost_out: 250.0, latency_ms: 170, quality: 0.84, context: 262_000,
                vision: true, open_weight: true, enabled: true),

            // ================================================================
            // ZHIPU / GLM — POST https://open.bigmodel.cn/api/paas/v4/chat/completions
            // OpenAI-compatible schema
            // ================================================================
            model!(api_id: "glm-5", display_name: "GLM-5", provider: Provider::Zhipu,
                cost_in: 57.0, cost_out: 258.0, latency_ms: 180, quality: 0.86, context: 200_000,
                vision: false, open_weight: true, enabled: true),

            // ================================================================
            // META LLAMA — POST https://api.llama.com/v1/chat/completions
            // OpenAI-compatible schema — open-weight
            // ================================================================
            model!(api_id: "llama-4-maverick", display_name: "Llama 4 Maverick", provider: Provider::Meta,
                cost_in: 20.0, cost_out: 60.0, latency_ms: 150, quality: 0.83, context: 1_000_000,
                vision: true, open_weight: true, enabled: true),

            model!(api_id: "llama-4-scout", display_name: "Llama 4 Scout", provider: Provider::Meta,
                cost_in: 8.0, cost_out: 30.0, latency_ms: 120, quality: 0.75, context: 10_000_000,
                vision: true, open_weight: true, enabled: true),

            model!(api_id: "llama-3.3-70b", display_name: "Llama 3.3 70B (legacy)", provider: Provider::Meta,
                cost_in: 12.0, cost_out: 40.0, latency_ms: 130, quality: 0.70, context: 128_000,
                vision: false, open_weight: true, enabled: true),

            // ================================================================
            // OPENROUTER — POST https://openrouter.ai/api/v1/chat/completions
            // Catch-all: reachable when a client only supplies an
            // OpenRouter key. Model ids use OpenRouter's "vendor/model" slug
            // directly, so these entries are picked verbatim (no rewrite).
            // ================================================================
            model!(api_id: "openrouter/auto", display_name: "OpenRouter Auto (best-available)", provider: Provider::OpenRouter,
                cost_in: 100.0, cost_out: 300.0, latency_ms: 200, quality: 0.85, context: 128_000,
                vision: false, open_weight: false, enabled: true),
        ]
    }

    // ===================================================================
    // SCORE-BASED ROUTING
    // ===================================================================

    #[instrument(skip(self))]
    pub fn select(
        &self,
        input_tokens: u32,
        max_output_tokens: u32,
        priority: RoutingPriority,
    ) -> Result<RoutingDecision> {
        self.select_reachable(input_tokens, max_output_tokens, priority, None)
    }

    /// Same scoring as `select`, but restricted to providers in `reachable`
    /// when it's `Some(set)`. `None` means "no filtering" (used when the
    /// client supplied an OpenRouter key, which is a universal fallback —
    /// see main.rs::reachable_providers).
    #[instrument(skip(self, reachable))]
    pub fn select_reachable(
        &self,
        input_tokens: u32,
        max_output_tokens: u32,
        priority: RoutingPriority,
        reachable: Option<&HashSet<Provider>>,
    ) -> Result<RoutingDecision> {
        let models = self.models.read();

        // Weight tuples: (cost, latency, quality, context_headroom)
        let (wc, wl, wq, wx) = match priority {
            RoutingPriority::Cost     => (0.60, 0.15, 0.20, 0.05),
            RoutingPriority::Balanced => (0.35, 0.25, 0.30, 0.10),
            RoutingPriority::Quality  => (0.10, 0.15, 0.65, 0.10),
            RoutingPriority::Speed    => (0.10, 0.70, 0.15, 0.05),
        };

        let mut best: Option<(ModelConfig, f64)> = None;

        for m in models.iter().filter(|m| m.enabled) {
            if let Some(r) = reachable {
                if !r.contains(&m.provider) {
                    continue;
                }
            }

            if input_tokens >= m.context_window {
                debug!(model = %m.api_id, "Skipped — context overflow");
                continue;
            }

            let cost = (input_tokens as f64 / 1_000_000.0) * m.cost_per_1m_input
                     + (max_output_tokens as f64 / 1_000_000.0) * m.cost_per_1m_output;

            let s_cost    = 1.0 / (1.0 + cost / 10.0);
            let s_latency = 1.0 / (1.0 + m.latency_ms as f64 / 200.0);
            let s_quality = m.quality_score as f64;
            let s_context = 1.0 - (input_tokens as f64 / m.context_window as f64);

            let score = wc * s_cost + wl * s_latency + wq * s_quality + wx * s_context;

            debug!(model = %m.api_id, score = format!("{:.4}", score));

            if best.as_ref().map_or(true, |(_, b)| score > *b) {
                best = Some((m.clone(), score));
            }
        }

        let (model, score) = best.ok_or_else(|| {
            anyhow!("No eligible model — check that at least one model is enabled, reachable given your supplied BYOK keys, and your input fits its context window")
        })?;

        let reason = format!(
            "{} (score={:.4}, priority={:?})",
            model.display_name, score, priority
        );
        info!("{}", reason);
        Ok(RoutingDecision { model, score, reason })
    }

    // ===================================================================
    // TASK-BASED ROUTING  (for meeting assistant)
    // ===================================================================

    pub fn select_for_task(
        &self,
        task: MeetingTask,
        input_tokens: u32,
        reachable: Option<&HashSet<Provider>>,
    ) -> Result<RoutingDecision> {
        let (priority, preferred) = match task {
            MeetingTask::Summarise          => (RoutingPriority::Balanced, "claude-sonnet-5"),
            MeetingTask::AnswerQuestion      => (RoutingPriority::Speed,    "gemini-3-flash"),
            MeetingTask::ExtractActionItems  => (RoutingPriority::Cost,     "deepseek-v4-flash"),
            MeetingTask::DraftResponse       => (RoutingPriority::Quality,  "claude-opus-5"),
            MeetingTask::Classify            => (RoutingPriority::Cost,     "gemini-3.1-flash-lite"),
        };

        // Try the preferred model first (best UX for each task) — but only
        // if it's actually reachable given the client's BYOK keys.
        if let Ok(m) = self.find(preferred) {
            let is_reachable = reachable.map_or(true, |r| r.contains(&m.provider));
            if m.enabled && is_reachable && input_tokens < m.context_window {
                let reason = format!("{} chosen as task-optimal for {:?}", m.display_name, task);
                info!("{}", reason);
                return Ok(RoutingDecision { model: m, score: 1.0, reason });
            }
        }

        // Preferred not available/reachable → fall back to score-based,
        // still respecting the reachability filter.
        self.select_reachable(input_tokens, 1024, priority, reachable)
    }

    // ===================================================================
    // HELPERS
    // ===================================================================

    /// Merges additional models into the registry — used to fold in the
    /// dynamically-fetched OpenRouter catalog (see openrouter_catalog.rs) on
    /// top of the curated direct-integration entries above. Any id that
    /// already exists (curated entries always win) is skipped, so this can
    /// never override a hand-tuned cost/latency/quality figure.
    pub fn extend_registry(&self, extra: Vec<ModelConfig>) -> usize {
        let mut models = self.models.write();
        let mut seen: std::collections::HashSet<String> =
            models.iter().map(|m| m.api_id.clone()).collect();

        let mut added = 0;
        for m in extra {
            if seen.insert(m.api_id.clone()) {
                models.push(m);
                added += 1;
            }
        }

        info!(added, total = models.len(), "Extended model registry");
        added
    }

    pub fn find(&self, api_id: &str) -> Result<ModelConfig> {
        self.models.read()
            .iter()
            .find(|m| m.api_id == api_id)
            .cloned()
            .ok_or_else(|| anyhow!("Unknown model id: {}", api_id))
    }

    /// Resolve a provider for a model name that may be a BYOK-only target
    /// (Azure or Bedrock). The presence of the corresponding connection
    /// header is sufficient — the model name does not need any special
    /// prefix. This allows any model name to be routed to Azure or Bedrock
    /// when the client supplies the appropriate BYOK header.
    pub fn resolve_byok_provider(
        &self,
        _model_name: &str,
        has_azure_header: bool,
        has_bedrock_header: bool,
    ) -> Option<Provider> {
        if has_azure_header {
            return Some(Provider::AzureOpenAI);
        }
        if has_bedrock_header {
            return Some(Provider::Bedrock);
        }
        None
    }

    /// Returns the `Provider` for a given model name. For Azure and Bedrock
    /// models the registry does not contain entries; if the request includes
    /// the corresponding BYOK header we treat the model as valid and return
    /// the provider directly.
    pub fn select_provider(
        &self,
        model_name: &str,
        has_azure_header: bool,
        has_bedrock_header: bool,
    ) -> Result<Provider> {
        // First try the static registry.
        if let Ok(m) = self.find(model_name) {
            return Ok(m.provider);
        }

        // Fall back to BYOK-only providers.
        if let Some(provider) = self.resolve_byok_provider(model_name, has_azure_header, has_bedrock_header) {
            return Ok(provider);
        }

        Err(anyhow!(
            "Unknown model '{}'. Check /v1/models for the list of supported model IDs.",
            model_name
        ))
    }

    pub fn get_pricing(&self, api_id: &str) -> Result<(f64, f64)> {
        let m = self.find(api_id)?;
        Ok((m.cost_per_1m_input, m.cost_per_1m_output))
    }

    pub fn list_enabled(&self) -> Vec<ModelConfig> {
        self.models.read().iter().filter(|m| m.enabled).cloned().collect()
    }

    pub fn list_vision_capable(&self) -> Vec<ModelConfig> {
        self.models.read().iter().filter(|m| m.enabled && m.supports_vision).cloned().collect()
    }

    pub fn is_vision_capable(&self, api_id: &str) -> bool {
        self.find(api_id).map(|m| m.supports_vision).unwrap_or(false)
    }

    /// Enable/disable a model at runtime (e.g. an admin endpoint flipping a
    /// provider off after sustained errors, independent of the circuit breaker).
    pub fn set_enabled(&self, api_id: &str, enabled: bool) -> Result<()> {
        let mut models = self.models.write();
        let m = models
            .iter_mut()
            .find(|m| m.api_id == api_id)
            .ok_or_else(|| anyhow!("Unknown model id: {}", api_id))?;
        m.enabled = enabled;
        Ok(())
    }
}

pub fn openrouter_slug_override(direct_api_id: &str) -> Option<&'static str> {
    match direct_api_id {
        "gemini-3-flash" => Some("google/gemini-3-flash-preview"),
        _ => None,
    }
}

impl Default for RouteEngine {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn balanced_picks_a_model() {
        let e = RouteEngine::new();
        e.select(5_000, 1_000, RoutingPriority::Balanced).unwrap();
    }

    #[test]
    fn cost_does_not_pick_flagship_opus() {
        let e = RouteEngine::new();
        let d = e.select(5_000, 1_000, RoutingPriority::Cost).unwrap();
        assert_ne!(d.model.api_id, "claude-opus-5");
    }

    #[test]
    fn overflow_model_excluded() {
        let e = RouteEngine::new();
        let d = e.select(150_000, 1_000, RoutingPriority::Balanced).unwrap();
        assert!(d.model.context_window >= 150_001);
    }

    #[test]
    fn task_routing_extract_picks_deepseek() {
        let e = RouteEngine::new();
        let d = e.select_for_task(MeetingTask::ExtractActionItems, 5_000, None).unwrap();
        assert_eq!(d.model.api_id, "deepseek-v4-flash");
    }

    #[test]
    fn find_opus_pricing() {
        let e = RouteEngine::new();
        let (input, output) = e.get_pricing("claude-opus-5").unwrap();
        assert_eq!(input, 500.0);
        assert_eq!(output, 2500.0);
    }

    #[test]
    fn every_provider_has_at_least_one_model() {
        let e = RouteEngine::new();
        let models = e.list_enabled();
        for provider in [
            Provider::Anthropic, Provider::OpenAI, Provider::Gemini, Provider::DeepSeek,
            Provider::Mistral, Provider::XAI, Provider::Qwen, Provider::Moonshot,
            Provider::Zhipu, Provider::Meta, Provider::OpenRouter,
        ] {
            assert!(
                models.iter().any(|m| m.provider == provider),
                "no enabled model registered for {:?}", provider
            );
        }
    }

    #[test]
    fn vision_models_are_flagged() {
        let e = RouteEngine::new();
        assert!(e.is_vision_capable("claude-opus-5"));
        assert!(e.is_vision_capable("gpt-5.6-sol"));
        assert!(e.is_vision_capable("gemini-3.1-pro"));
        assert!(!e.is_vision_capable("deepseek-v4-flash"));
        assert!(!e.is_vision_capable("grok-4.1-fast"));
    }

    #[test]
    fn select_reachable_filters_by_provider() {
        let e = RouteEngine::new();
        let mut only_deepseek = HashSet::new();
        only_deepseek.insert(Provider::DeepSeek);
        let d = e.select_reachable(5_000, 1_000, RoutingPriority::Balanced, Some(&only_deepseek)).unwrap();
        assert_eq!(d.model.provider, Provider::DeepSeek);
    }

    #[test]
    fn select_reachable_errors_when_nothing_matches() {
        let e = RouteEngine::new();
        let empty = HashSet::new();
        assert!(e.select_reachable(5_000, 1_000, RoutingPriority::Balanced, Some(&empty)).is_err());
    }

    #[test]
    fn task_routing_falls_back_when_preferred_unreachable() {
        let e = RouteEngine::new();
        // claude-sonnet-5 (Summarise's preferred model) is Anthropic —
        // restrict to DeepSeek only, forcing the scored fallback.
        let mut only_deepseek = HashSet::new();
        only_deepseek.insert(Provider::DeepSeek);
        let d = e.select_for_task(MeetingTask::Summarise, 5_000, Some(&only_deepseek)).unwrap();
        assert_eq!(d.model.provider, Provider::DeepSeek);
    }

    #[test]
    fn byok_azure_provider_resolved_with_header() {
        let e = RouteEngine::new();
        let provider = e.select_provider("any-model-name", true, false).unwrap();
        assert_eq!(provider, Provider::AzureOpenAI);
    }

    #[test]
    fn byok_bedrock_provider_resolved_with_header() {
        let e = RouteEngine::new();
        let provider = e.select_provider("any-model-name", false, true).unwrap();
        assert_eq!(provider, Provider::Bedrock);
    }

    #[test]
    fn byok_azure_rejected_without_header() {
        let e = RouteEngine::new();
        assert!(e.select_provider("any-model-name", false, false).is_err());
    }

    #[test]
    fn byok_bedrock_rejected_without_header() {
        let e = RouteEngine::new();
        assert!(e.select_provider("any-model-name", false, false).is_err());
    }
}
