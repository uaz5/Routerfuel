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
    /// Price break that kicks in once a request's *input* is large enough.
    ///
    /// `None` for every model whose vendor charges one flat rate per
    /// direction, which is almost all of them. See `PriceTier` for why this
    /// is a multiplier over the base rates rather than a second pair of
    /// absolute rates.
    pub long_context_tier: Option<PriceTier>,
}

/// A long-context price break, expressed as multipliers over
/// `ModelConfig`'s base rates.
///
/// Multipliers rather than a second pair of absolute rates so there is
/// exactly one place a model's price lives: correcting a base rate (which
/// happens — see the two Gemini corrections below) automatically carries
/// through to the tiered rate instead of leaving a stale second copy that
/// silently disagrees.
///
/// Note the asymmetry, which is the vendor's and not ours: the threshold is
/// measured on *input* tokens but the multipliers apply to input and output
/// alike, for the whole request. A request one token over the line costs
/// `output_mult` more on every output token it goes on to generate.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PriceTier {
    /// Applies when input tokens are *strictly greater* than this.
    pub above_input_tokens: u32,
    pub input_mult: f64,
    pub output_mult: f64,
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

// ============================================================================
// REQUEST CLASSIFICATION FOR "auto" ROUTING
//
// Two axes, both feeding the *existing* scored path rather than a parallel
// one: classification's whole output is which `RoutingPriority` and which
// `SelectionLimits` get handed to `select_filtered` — the same function
// every other route already goes through.
// ============================================================================

/// Coarse task buckets for `model: "auto"` requests.
///
/// Deliberately only two. `ModelConfig` has no per-model task-suitability
/// field (`supports_vision` is the only capability flag), so `Code` is the
/// one bucket with genuinely distinct targets in the registry —
/// `grok-code-fast-1` and `codestral-2508`. A `Creative` bucket was considered
/// and dropped: nothing in the registry serves it differently from
/// `General`, so it would have detected poorly and then routed identically.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TaskKind {
    Code,
    General,
}

/// How much model the request looks like it needs.
///
/// Only `Simple` changes routing today — it downgrades to
/// `RoutingPriority::Cost` behind a quality floor and a price ceiling
/// (see `SelectionLimits::simple`). `Moderate` and `Complex`
/// both map to `Balanced`, which is exactly what `"auto"` did before
/// classification existed, so this can never make a request more expensive
/// than it already was. `Complex` is still computed and reported (see the
/// annotated `RoutingDecision::reason`) so there's real traffic data behind
/// any later decision to escalate it to `Quality`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Difficulty {
    Simple,
    Moderate,
    Complex,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RequestShape {
    pub task: TaskKind,
    pub difficulty: Difficulty,
}

impl RequestShape {
    pub fn priority(&self) -> RoutingPriority {
        match self.difficulty {
            Difficulty::Simple => RoutingPriority::Cost,
            // Unchanged from the pre-classification default. See Difficulty.
            Difficulty::Moderate | Difficulty::Complex => RoutingPriority::Balanced,
        }
    }
}

// The bias is deliberately toward "not simple": a request wrongly called
// simple gets a weak answer, while one wrongly called complex only costs
// more. So `Simple` requires *every* condition to hold, while `Complex`
// needs only one.
const SIMPLE_MAX_INPUT_TOKENS: u32 = 500;
const SIMPLE_MAX_OUTPUT_TOKENS: u32 = 512;
const COMPLEX_MIN_INPUT_TOKENS: u32 = 4_000;
const COMPLEX_MIN_OUTPUT_TOKENS: u32 = 2_000;
const COMPLEX_MIN_MESSAGES: usize = 6;

/// Quality floor for the `Simple` path — stops a cheap-but-weak model from
/// capturing simple traffic. Never allowed to fail a request; see
/// `select_for_shape`.
const SIMPLE_MIN_QUALITY: f32 = 0.60;

/// Price ceiling for the `Simple` path, in cents per 1M tokens, applied to
/// `cost_per_1m_input + cost_per_1m_output`.
///
/// This is load-bearing, not insurance — without it the simple path does
/// not actually route to a cheap model. `RoutingPriority::Cost` alone is
/// not enough, because `s_cost` is `1/(1 + cost/10)`: on a 300-input /
/// 256-output request every candidate costs between 0.006c and 0.09c, so
/// s_cost only spans 0.9909..0.9994. That 0.0085 spread times the 0.60 cost
/// weight is ~0.005 of final score, while the quality spread (0.58..0.85)
/// times the 0.20 quality weight is ~0.054 — an order of magnitude more.
/// Cost weighting therefore ranks by *quality* on precisely the small
/// requests we want to cheapen, and only starts behaving like its name at
/// around 1 cent per request. Measured: without this ceiling, a simple
/// request picks gemini-3-flash-preview (50+300 = 350 blended) over
/// deepseek-v4-flash (14+28 = 42) — an 8x price difference in the wrong
/// direction.
///
/// 60 admits the flash-lite / small-model tier and excludes the mid-tier
/// flash and code models. Like the floor, it never fails a request.
const SIMPLE_MAX_BLENDED_COST_PER_1M: f64 = 60.0;

/// Optional pre-filters applied before the weighted comparison in
/// `select_filtered`. Grouped into a struct rather than added as loose
/// parameters so the selection signature stays readable as policy knobs
/// accumulate. `Default` is "no limits", i.e. exactly what every caller
/// before shape-based routing did.
#[derive(Debug, Clone, Copy, Default)]
pub struct SelectionLimits {
    pub min_quality: Option<f32>,
    /// Ceiling on `cost_per_1m_input + cost_per_1m_output`, in cents.
    pub max_blended_cost_per_1m: Option<f64>,
}

impl SelectionLimits {
    /// The limits that define "cheap, fast, general" for the simple path.
    fn simple() -> Self {
        Self {
            min_quality: Some(SIMPLE_MIN_QUALITY),
            max_blended_cost_per_1m: Some(SIMPLE_MAX_BLENDED_COST_PER_1M),
        }
    }
}

/// Cheap, code-tuned models preferred for `Simple` + `Code`, in order.
///
/// grok-code-fast-1 stays listed first but is now a disabled registry entry
/// (absent from xAI's model list — see build_registry), so in practice this
/// resolves to codestral-2508. The loop below already gates on `enabled`, so
/// leaving it here is harmless and keeps the preference order recorded for
/// whenever xAI's catalog is re-checked.
/// A short explicit list rather than a scored field, for the same reason
/// `select_for_task` keeps a preferred-model fast path: these are the only
/// registry entries with a real task specialization, and inventing a
/// `code_score` for all ~45 curated entries (plus a default for the ~300
/// OpenRouter catalog entries merged at startup) would be far more invented
/// numbers than this earns.
const SIMPLE_CODE_MODELS: &[&str] = &["grok-code-fast-1", "codestral-2508"];

/// Substrings that mark a request as coding work. Matched case-insensitively.
///
/// Precision matters much less here than it appears: `TaskKind::Code` only
/// changes routing when the request is *also* `Difficulty::Simple`, where
/// the choice is between two cheap fast models. A false positive swaps
/// gemini-3.1-flash-lite for grok-code-fast-1 and costs nothing meaningful;
/// task type has no effect at all on Moderate/Complex routing.
const CODE_MARKERS: &[&str] = &[
    "```",
    "def ",
    "fn ",
    "func ",
    "function ",
    "import ",
    "#include",
    "public static",
    "let mut ",
    "traceback",
    "stack trace",
    "stacktrace",
    "segmentation fault",
    "null pointer",
    "compile error",
    "syntax error",
    "unit test",
    "refactor",
    "npm ",
    "cargo ",
    "git commit",
    "dockerfile",
    ".py",
    ".rs",
    ".ts",
    ".js",
    ".java",
    ".cpp",
    ".sql",
];

/// Classifies an incoming `"auto"` request along both axes using only data
/// the request path has already computed — no extra provider call. A
/// pre-flight classification call was considered and rejected: every call
/// here is BYOK-billed to the client and reserved through SpendGuard, so it
/// would double the request count and need its own key selection and spend
/// accounting just to decide a request was cheap.
///
/// `max_output_tokens` is the client's `max_tokens` *as supplied*. `None`
/// deliberately reads as "no signal" — it neither blocks `Simple` nor
/// triggers `Complex`. Passing the resolved default (1024) instead would
/// mean no request that omits `max_tokens` could ever be `Simple`, which is
/// most of them.
pub fn classify(
    messages: &[crate::connectors::ChatMessage],
    input_tokens: u32,
    max_output_tokens: Option<u32>,
    has_image: bool,
) -> RequestShape {
    let task = if messages.iter().any(|m| {
        let text = m.content.as_text().to_lowercase();
        CODE_MARKERS.iter().any(|marker| text.contains(marker))
    }) {
        TaskKind::Code
    } else {
        TaskKind::General
    };

    // A system prompt alongside one user turn is still a single-shot
    // request, so count user turns rather than total messages here.
    let user_turns = messages.iter().filter(|m| m.role == "user").count();

    let output_is_small = max_output_tokens.map_or(true, |n| n <= SIMPLE_MAX_OUTPUT_TOKENS);
    let output_is_large = max_output_tokens.map_or(false, |n| n > COMPLEX_MIN_OUTPUT_TOKENS);

    let difficulty = if input_tokens < SIMPLE_MAX_INPUT_TOKENS
        && output_is_small
        && user_turns <= 1
        && !has_image
    {
        Difficulty::Simple
    } else if input_tokens > COMPLEX_MIN_INPUT_TOKENS
        || output_is_large
        || messages.len() > COMPLEX_MIN_MESSAGES
    {
        Difficulty::Complex
    } else {
        Difficulty::Moderate
    };

    RequestShape { task, difficulty }
}

/// Appends the classification to a routing reason string. `reason` is
/// already the carrier for "why this model" — it's logged, stored on
/// `RoutingDecision`, and surfaced by the admin dashboard — so the
/// classification rides along there instead of needing a new field
/// threaded through the request path.
fn annotate_reason(reason: &str, shape: RequestShape) -> String {
    format!("{} [task={:?}, difficulty={:?}]", reason, shape.task, shape.difficulty)
}

pub struct RouteEngine {
    models: RwLock<Vec<ModelConfig>>,
}

macro_rules! model {
    // With a long-context price break. Must precede the plain arm below,
    // since macro arms are matched in order.
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
        enabled: $enabled:expr,
        tier: $tier:expr $(,)?
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
            long_context_tier: Some($tier),
        }
    };
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
            long_context_tier: None,
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
    /// broke Gemini 3 Flash (OpenRouter's real slug needed a "-preview"
    /// suffix the formula didn't produce), just generalized into a check
    /// instead of a single hand-added exception.
    ///
    /// Note this guards the OpenRouter path only. There is no equivalent
    /// check on the direct path — nothing validates a curated api_id against
    /// the vendor's own model list, which is why the Gemini and xAI errors
    /// corrected in build_registry went unnoticed. A startup reconciliation
    /// against each provider's GET /models would close that gap; tracked
    /// separately.
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

            // 5.1 takes over 0.99 as Anthropic's top entry and claude-fable-5
            // steps down to 0.98 (below), so the two rank on quality outright
            // rather than tying and falling through to registry order.
            model!(api_id: "claude-fable-5-1", display_name: "Claude Fable 5.1", provider: Provider::Anthropic,
                cost_in: 1000.0, cost_out: 5000.0, latency_ms: 320, quality: 0.99, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "claude-fable-5", display_name: "Claude Fable 5", provider: Provider::Anthropic,
                cost_in: 1000.0, cost_out: 5000.0, latency_ms: 320, quality: 0.98, context: 1_000_000,
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
            // Launched 2026-09-03. Two things here are not like the others:
            //
            // 1. `enabled: false` on purpose, not as a placeholder. Astra is
            //    rolling out to Trusted Access Program enterprises first,
            //    with general API access "in the coming days". Because
            //    find() ignores `enabled` while select_*() honours it, this
            //    gives exactly the right semantics for a limited rollout: a
            //    client who names "gpt-6-astra" explicitly gets routed to
            //    OpenAI, but "auto"/"task:" routing will not pick a model
            //    most clients cannot call yet, and /v1/models (which uses
            //    list_enabled) does not advertise it. Flip to true on GA.
            //
            // 2. The cost figures below are OpenAI's SHORT-context rates, and
            //    `tier` now carries the long-context break that ModelConfig
            //    previously could not express. OpenAI's model page states it
            //    verbatim: "Prompts with more than 272K input tokens are
            //    priced at 2x input and cache rates and 1.5x output for the
            //    full request." get_pricing_for() applies it; plain
            //    get_pricing() still returns the base rates, which is what
            //    the display/reporting callers want.
            //
            //    Two further pricing axes remain unexpressed and are NOT
            //    modelled here, deliberately: Fast mode (2x both directions,
            //    $20/$100) and cache rates ($1/1M cached input, $12.50/1M
            //    cache writes). Neither is reachable through RouterFuel
            //    today — nothing sets a service tier and nothing tracks cache
            //    hits per provider — so modelling them would be inventing
            //    fields no caller can populate.
            //
            // 3. ASSUMPTION (docs-only, no authenticated request made):
            //    context corrected 1_100_000 -> 1_050_000 to match OpenAI's
            //    published "1,050,000 context window", which is also what the
            //    gpt-5.6 family already carries here. The old figure would
            //    have had select_reachable admit requests between 1.05M and
            //    1.1M tokens that OpenAI then rejects — the failure mode the
            //    GLM-5.3 note below argues against.
            //
            // 4. BLOCKER on flipping `enabled`, separate from GA: Astra
            //    rejects custom `temperature` and `top_p` (OpenAI's migration
            //    guidance is to remove both, and to omit `logprobs` on Chat
            //    Completions). build_openai_compatible_body() in connectors.rs
            //    forwards both whenever the client supplies them, so enabling
            //    Astra today would 400 every request from a client that sets
            //    a temperature — routine on an OpenAI-compatible surface.
            //    Tool calling additionally requires the Responses API, which
            //    RouterFuel does not speak. Enabling this needs per-model
            //    parameter filtering built first; it is not a bool flip.
            model!(api_id: "gpt-6-astra", display_name: "GPT-6 Astra", provider: Provider::OpenAI,
                cost_in: 1000.0, cost_out: 5000.0, latency_ms: 340, quality: 0.99, context: 1_050_000,
                vision: true, open_weight: false, enabled: false,
                tier: PriceTier { above_input_tokens: 272_000, input_mult: 2.0, output_mult: 1.5 }),

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
            // ASSUMPTION (docs-only, no authenticated request made): renamed
            // "gemini-3.1-pro" -> "gemini-3.1-pro-preview". Google's model
            // list publishes the "-preview" form and the bare id appears
            // nowhere in current docs, so the direct path was sending an id
            // generativelanguage would 404 on. Same class as the
            // gemini-3-flash rename below. display_name deliberately keeps
            // the marketing name — the suffix is an API detail, not something
            // to surface on the dashboard.
            model!(api_id: "gemini-3.1-pro-preview", display_name: "Gemini 3.1 Pro", provider: Provider::Gemini,
                cost_in: 200.0, cost_out: 1200.0, latency_ms: 210, quality: 0.96, context: 2_000_000,
                vision: true, open_weight: false, enabled: true),

            // The 3.6/3.7/3.8 Flash line all price identically (75/375) and
            // sit on the same 1,048,576-token window; they differ only in
            // generation, so quality is stepped 0.88/0.89/0.90 to keep the
            // newest preferred without disturbing anything else.
            model!(api_id: "gemini-3.8-flash", display_name: "Gemini 3.8 Flash", provider: Provider::Gemini,
                cost_in: 75.0, cost_out: 375.0, latency_ms: 115, quality: 0.90, context: 1_048_576,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gemini-3.7-flash", display_name: "Gemini 3.7 Flash", provider: Provider::Gemini,
                cost_in: 75.0, cost_out: 375.0, latency_ms: 115, quality: 0.89, context: 1_048_576,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gemini-3.6-flash", display_name: "Gemini 3.6 Flash", provider: Provider::Gemini,
                cost_in: 75.0, cost_out: 375.0, latency_ms: 115, quality: 0.88, context: 1_048_576,
                vision: true, open_weight: false, enabled: true),

            // Corrected: this entry carried 75/450, but Google's current
            // list price is 150/900 — the 3.6/3.7/3.8 Flash models above are
            // what costs 75/375 now. Left uncorrected, the router would rank
            // 3.5 Flash as comparable to its successors while it is actually
            // twice the price.
            model!(api_id: "gemini-3.5-flash", display_name: "Gemini 3.5 Flash", provider: Provider::Gemini,
                cost_in: 150.0, cost_out: 900.0, latency_ms: 120, quality: 0.87, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gemini-3.5-flash-lite", display_name: "Gemini 3.5 Flash-Lite", provider: Provider::Gemini,
                cost_in: 30.0, cost_out: 250.0, latency_ms: 75, quality: 0.74, context: 1_048_576,
                vision: true, open_weight: false, enabled: true),

            // ASSUMPTION (docs-only, no authenticated request made): renamed
            // "gemini-3-flash" -> "gemini-3-flash-preview".
            //
            // This id was already known to need the "-preview" suffix — that
            // is what openrouter_slug_override was added for — but the fix
            // landed on the OpenRouter axis only. The DIRECT path sends
            // api_id verbatim into the generativelanguage URL (see
            // GeminiConnector::complete), so a client with a direct Gemini
            // key has been getting a 404 this whole time while a client on an
            // OpenRouter key was fine. The override table could never have
            // masked that: it is consulted only when the client has no direct
            // key for the provider.
            //
            // Renaming here makes the OpenRouter override redundant — the
            // "{prefix}/{model}" formula now yields the real slug
            // "google/gemini-3-flash-preview" on its own — so that entry is
            // deleted below rather than left as a no-op.
            model!(api_id: "gemini-3-flash-preview", display_name: "Gemini 3 Flash", provider: Provider::Gemini,
                cost_in: 50.0, cost_out: 300.0, latency_ms: 105, quality: 0.85, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            // Corrected: this entry carried 10/40, which is 2.5 Flash-Lite's
            // price, not 3.1's. Google lists 3.1 Flash-Lite at 25/150.
            model!(api_id: "gemini-3.1-flash-lite", display_name: "Gemini 3.1 Flash-Lite", provider: Provider::Gemini,
                cost_in: 25.0, cost_out: 150.0, latency_ms: 70, quality: 0.68, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gemini-2.5-pro", display_name: "Gemini 2.5 Pro", provider: Provider::Gemini,
                cost_in: 125.0, cost_out: 1000.0, latency_ms: 200, quality: 0.89, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gemini-2.5-flash", display_name: "Gemini 2.5 Flash", provider: Provider::Gemini,
                cost_in: 30.0, cost_out: 250.0, latency_ms: 110, quality: 0.78, context: 1_048_576,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "gemini-2.5-flash-lite", display_name: "Gemini 2.5 Flash-Lite", provider: Provider::Gemini,
                cost_in: 10.0, cost_out: 40.0, latency_ms: 65, quality: 0.66, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            // ================================================================
            // xAI GROK — POST https://api.x.ai/v1/chat/completions
            // OpenAI-compatible schema
            // ================================================================
            // Context corrected across this block against xAI's published
            // model pages. 4.6 and 4.5 are both 500k; 4.3 and 4.20 are both
            // 1M. The previous entries had 4.5 and 4.20 at 2M, which is not a
            // figure xAI publishes for either — the effect was
            // select_reachable admitting 1M–2M-token requests that xAI then
            // rejects. Prices were already right ($2/$6 for 4.5, $1.25/$2.50
            // for 4.20), so only the windows moved.
            //
            // Note this removes the tier inversion the old comment here
            // described: no Grok has a larger window than 4.3/4.20's 1M, so
            // inputs above 1M no longer fall through to an "older Grok" —
            // they leave the xAI family entirely.
            model!(api_id: "grok-4.6", display_name: "Grok 4.6", provider: Provider::XAI,
                cost_in: 200.0, cost_out: 600.0, latency_ms: 200, quality: 0.96, context: 500_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "grok-4.5", display_name: "Grok 4.5", provider: Provider::XAI,
                cost_in: 200.0, cost_out: 600.0, latency_ms: 190, quality: 0.93, context: 500_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "grok-4.3", display_name: "Grok 4.3", provider: Provider::XAI,
                cost_in: 125.0, cost_out: 250.0, latency_ms: 160, quality: 0.88, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "grok-4.20", display_name: "Grok 4.20", provider: Provider::XAI,
                cost_in: 125.0, cost_out: 250.0, latency_ms: 165, quality: 0.88, context: 1_000_000,
                vision: true, open_weight: false, enabled: true),

            // DISABLED, not deleted: this id is absent from xAI's current
            // model list, and xAI's retirement notice says the
            // `grok-4-1-fast` family stopped serving on 2026-05-15 and now
            // redirects to grok-4.3. So every request the router sent here
            // was either a 404 or a silent substitution billed at another
            // model's rates. Kept as a disabled row so `find()` still
            // resolves it for pricing/display on historical request_logs
            // instead of turning old rows into "unknown model".
            //
            // Note the id here is also suspect independently of retirement:
            // xAI wrote it "grok-4-1-fast" (dashes), not "grok-4.1-fast".
            model!(api_id: "grok-4.1-fast", display_name: "Grok 4.1 Fast (retired)", provider: Provider::XAI,
                cost_in: 20.0, cost_out: 50.0, latency_ms: 90, quality: 0.75, context: 2_000_000,
                vision: false, open_weight: false, enabled: false),

            // DISABLED for the same reason: not present in xAI's current
            // model list. Unlike grok-4.1-fast there is no explicit
            // retirement notice naming it, so "gone" is inferred from absence
            // rather than stated — but SIMPLE_CODE_MODELS listed this first,
            // meaning every Simple+Code request was being aimed at it. That
            // path checks `enabled`, so it now falls through to codestral-2.
            model!(api_id: "grok-code-fast-1", display_name: "Grok Code Fast 1 (unlisted)", provider: Provider::XAI,
                cost_in: 20.0, cost_out: 150.0, latency_ms: 100, quality: 0.78, context: 256_000,
                vision: false, open_weight: false, enabled: false),

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
            // ASSUMPTION (docs-only, no authenticated request made): all four
            // ids renamed from bare marketing names to Mistral's dated ids,
            // quoted verbatim from Mistral's own changelog:
            //
            //   mistral-large-3 -> mistral-large-2512
            //   mistral-small-4 -> mistral-small-2603
            //   codestral-2     -> codestral-2508
            //   ministral-8b    -> ministral-8b-2512
            //
            // Mistral's API keys models by date code; the bare names were
            // product labels, not ids, so all four were 404s on the direct
            // path. Same defect class as the two Gemini ids.
            //
            // Dated ids rather than the "-latest" aliases that also exist
            // (mistral-large-latest and friends), because every entry here
            // carries hand-tuned cost/quality/context for one specific
            // release. A floating alias invalidates all three silently the
            // day Mistral rolls it — and the changelog shows codestral-latest
            // having pointed at an older build than the then-current release,
            // which is that failure mode already happening once.
            //
            // NOT re-verified: the cost/latency/quality/context figures below
            // are the ones the old bare-name entries carried. The ids are now
            // right; the numbers are inherited on the assumption that these
            // are the same releases those figures were written for. That is
            // shakiest for mistral-small-2603, which the changelog calls "a
            // hybrid model unifying instruct, reasoning, and coding" -- not
            // obviously the model a 0.72 quality score was chosen for. Worth
            // a pricing pass; renaming does not settle it.
            model!(api_id: "mistral-large-2512", display_name: "Mistral Large 3", provider: Provider::Mistral,
                cost_in: 50.0, cost_out: 150.0, latency_ms: 165, quality: 0.86, context: 128_000,
                vision: true, open_weight: false, enabled: true),

            model!(api_id: "mistral-small-2603", display_name: "Mistral Small 4", provider: Provider::Mistral,
                cost_in: 10.0, cost_out: 30.0, latency_ms: 100, quality: 0.72, context: 128_000,
                vision: true, open_weight: true, enabled: true),

            model!(api_id: "codestral-2508", display_name: "Codestral 2508", provider: Provider::Mistral,
                cost_in: 30.0, cost_out: 90.0, latency_ms: 110, quality: 0.79, context: 256_000,
                vision: false, open_weight: false, enabled: true),

            model!(api_id: "ministral-8b-2512", display_name: "Ministral 8B", provider: Provider::Mistral,
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

            // DISABLED: Moonshot retired the kimi-k2.5 and moonshot-v1 series
            // on 2026-08-31; calls now return 404 model-not-found. This was
            // registered `enabled: true`, so auto/task routing could and did
            // pick a model that cannot answer. Kept as a disabled row so
            // historical request_logs still resolve for pricing and display.
            model!(api_id: "kimi-k2.5", display_name: "Kimi K2.5 (retired)", provider: Provider::Moonshot,
                cost_in: 60.0, cost_out: 250.0, latency_ms: 170, quality: 0.84, context: 262_000,
                vision: true, open_weight: true, enabled: false),

            // ================================================================
            // ZHIPU / GLM — POST https://open.bigmodel.cn/api/paas/v4/chat/completions
            // OpenAI-compatible schema
            // ================================================================
            // GLM-5.3 — Zhipu's current flagship. Two conservative calls
            // here, both erring toward "won't be picked" rather than "picked
            // and fails":
            //
            //   context: sources disagree — OpenRouter's catalog reports
            //   1,310,720 while Zhipu's own materials say 1M. Understating
            //   only means very large requests route elsewhere; overstating
            //   means accepting a request the provider then rejects. 1M it
            //   is until confirmed from Zhipu's docs directly.
            //
            //   vision: false — Zhipu ships vision as separate "-v" variants
            //   (glm-4.5v, glm-4.6v, glm-5v-turbo) and OpenRouter reports no
            //   image modality on plain glm-5.3, though some write-ups claim
            //   vision. False keeps vision.rs from routing images at a model
            //   that may reject them.
            //
            //   open_weight: false unlike glm-5 above — GLM's older releases
            //   are open-weight but I found no weight release for 5.3. This
            //   only affects display/filtering, never routing.
            model!(api_id: "glm-5.3", display_name: "GLM-5.3", provider: Provider::Zhipu,
                cost_in: 140.0, cost_out: 440.0, latency_ms: 200, quality: 0.92, context: 1_000_000,
                vision: false, open_weight: false, enabled: true),

            model!(api_id: "glm-5", display_name: "GLM-5", provider: Provider::Zhipu,
                cost_in: 57.0, cost_out: 258.0, latency_ms: 180, quality: 0.86, context: 200_000,
                vision: false, open_weight: true, enabled: true),

            // ================================================================
            // META LLAMA — POST https://api.llama.com/v1/chat/completions
            // OpenAI-compatible schema — open-weight
            // ================================================================
            // DEFERRED, and the earlier note here was wrong about why.
            //
            // These ids are NOT mis-slugged. openrouter_prefix(Meta) is
            // "meta-llama", so resolve_byok_route's formula yields
            // "meta-llama/llama-4-maverick" — OpenRouter's real, live slug.
            // For any client on an OpenRouter key these three work today, and
            // because reachable_providers returns None for an OpenRouter key,
            // they are auto-selectable for those clients too.
            //
            // The actual problem is bigger than an id: the whole PROVIDER is
            // gone. Meta wound the Llama API public preview down on
            // 2026-07-06 -- api.llama.com now returns a sunset response,
            // llama.developer.meta.com redirects to ai.developer.meta.com,
            // and that catalog lists only Muse models (muse-spark-1.3 and
            // siblings), no Llama at all. Meta points developers at
            // third-party hosts. So provider_base_url(Provider::Meta) in
            // connectors.rs addresses a service that no longer exists, and
            // the direct path here is dead no matter what the ids say.
            //
            // Deliberately left alone rather than given the kimi-k2.5
            // treatment: `enabled: false` would also remove the OpenRouter
            // routes that currently succeed. The real fix is to decide
            // whether these become Provider::OpenRouter entries with explicit
            // "meta-llama/..." ids and Provider::Meta retires -- a structural
            // change touching the Provider enum, reachable_providers, and
            // ClientProviderKeys, not a rename. Tracked as its own task.
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
        self.select_filtered(
            input_tokens,
            max_output_tokens,
            priority,
            reachable,
            SelectionLimits::default(),
        )
    }

    /// Same as `select_reachable`, plus `limits` — optional quality-floor
    /// and price-ceiling pre-filters applied before the weighted comparison.
    ///
    /// Layered underneath `select_reachable` (which is itself what `select`
    /// delegates to) so limits are purely additive: every existing caller
    /// keeps its current behaviour via `SelectionLimits::default()`. The
    /// only caller that sets limits today is `select_for_shape`'s simple
    /// path.
    #[instrument(skip(self, reachable))]
    pub fn select_filtered(
        &self,
        input_tokens: u32,
        max_output_tokens: u32,
        priority: RoutingPriority,
        reachable: Option<&HashSet<Provider>>,
        limits: SelectionLimits,
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

            if let Some(floor) = limits.min_quality {
                if m.quality_score < floor {
                    debug!(model = %m.api_id, floor, "Skipped — below min_quality floor");
                    continue;
                }
            }

            if let Some(ceiling) = limits.max_blended_cost_per_1m {
                let blended = m.cost_per_1m_input + m.cost_per_1m_output;
                if blended > ceiling {
                    debug!(model = %m.api_id, blended, ceiling, "Skipped — above price ceiling");
                    continue;
                }
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
    // SHAPE-BASED ROUTING  (for "auto" requests)
    // ===================================================================

    /// Routes an `"auto"` request from its classified shape.
    ///
    /// This is an extension of the scored path, not an alternative to it:
    /// for everything except `Difficulty::Simple` it calls exactly the same
    /// `select_reachable` with exactly the same `Balanced` priority that
    /// `"auto"` used before classification existed, so those requests are
    /// bit-for-bit unchanged. `Simple` is the only new behaviour — it swaps
    /// in `RoutingPriority::Cost` plus `SelectionLimits::simple`.
    pub fn select_for_shape(
        &self,
        shape: RequestShape,
        input_tokens: u32,
        max_output_tokens: u32,
        reachable: Option<&HashSet<Provider>>,
    ) -> Result<RoutingDecision> {
        let priority = shape.priority();

        if shape.difficulty != Difficulty::Simple {
            let mut decision =
                self.select_reachable(input_tokens, max_output_tokens, priority, reachable)?;
            decision.reason = annotate_reason(&decision.reason, shape);
            return Ok(decision);
        }

        // Simple + Code: prefer a cheap code-tuned model the client can
        // actually reach, same preferred-model-then-fall-back-to-scoring
        // shape as select_for_task.
        if shape.task == TaskKind::Code {
            for preferred in SIMPLE_CODE_MODELS {
                if let Ok(m) = self.find(preferred) {
                    let is_reachable = reachable.map_or(true, |r| r.contains(&m.provider));
                    if m.enabled && is_reachable && input_tokens < m.context_window {
                        let reason = format!(
                            "{} chosen as cheap code-tuned model [task=Code, difficulty=Simple]",
                            m.display_name
                        );
                        info!("{}", reason);
                        return Ok(RoutingDecision { model: m, score: 1.0, reason });
                    }
                }
            }
        }

        // Simple + General — and Simple + Code where no code-tuned model was
        // reachable: best-scoring thing inside the cheap tier.
        let mut decision = match self.select_filtered(
            input_tokens,
            max_output_tokens,
            priority,
            reachable,
            SelectionLimits::simple(),
        ) {
            Ok(d) => d,
            // The limits express a preference, not a requirement — they must
            // never be the reason a request fails. Retry unrestricted before
            // giving up, so a client whose only reachable models are pricey
            // or weak still gets served.
            Err(_) => {
                debug!(
                    "no reachable model satisfied the simple-path limits — retrying without them"
                );
                self.select_reachable(input_tokens, max_output_tokens, priority, reachable)?
            }
        };

        decision.reason = annotate_reason(&decision.reason, shape);
        Ok(decision)
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
            MeetingTask::AnswerQuestion      => (RoutingPriority::Speed,    "gemini-3-flash-preview"),
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

    /// Base rates, in cents per 1M tokens, ignoring any long-context tier.
    ///
    /// Correct for display, reporting, and cost comparisons between models.
    /// For anything that has to match what the provider will actually bill —
    /// a SpendGuard reservation or a post-call reconcile — use
    /// `get_pricing_for`, which needs the request's input size to know which
    /// tier applies.
    pub fn get_pricing(&self, api_id: &str) -> Result<(f64, f64)> {
        let m = self.find(api_id)?;
        Ok((m.cost_per_1m_input, m.cost_per_1m_output))
    }

    /// Rates for a request of a given input size, applying the model's
    /// long-context tier when the input crosses its threshold.
    ///
    /// Identical to `get_pricing` for every model with no tier, which is all
    /// but gpt-6-astra today — so callers can use this unconditionally
    /// rather than branching on whether the selected model happens to have
    /// tiered pricing.
    ///
    /// `input_tokens` must be the count that will actually be sent, i.e.
    /// *after* supercompress has run. Both reservation call sites already
    /// recompute it in that order.
    pub fn get_pricing_for(&self, api_id: &str, input_tokens: u32) -> Result<(f64, f64)> {
        let m = self.find(api_id)?;
        Ok(match m.long_context_tier {
            Some(t) if input_tokens > t.above_input_tokens => (
                m.cost_per_1m_input * t.input_mult,
                m.cost_per_1m_output * t.output_mult,
            ),
            _ => (m.cost_per_1m_input, m.cost_per_1m_output),
        })
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

/// Strips a bracketed variant suffix from a model id.
///
/// Claude Code identifies the long-context variant of a model as
/// `claude-opus-5[1m]`, and that string can reach the gateway verbatim on
/// the native `/v1/messages` endpoint. The registry keys on the provider's
/// real api_id (`claude-opus-5`), so a bracketed id would miss `find()` and
/// be reported as an unknown model.
///
/// Only the bracketed *suffix* is removed, and only the portion from the
/// first `[` onward — a model id that legitimately contains no bracket is
/// returned untouched, and this borrows rather than allocating so it is
/// free on the common path.
pub fn normalize_model_id(model: &str) -> &str {
    match model.find('[') {
        Some(i) => model[..i].trim_end(),
        None => model,
    }
}

/// Maps a direct-path api_id to OpenRouter's real slug, for the cases where
/// resolve_byok_route's "{prefix}/{model}" formula guesses wrong.
///
/// Scope worth being precise about, because it was previously misread as a
/// general api_id patch list: this table affects the OPENROUTER path only,
/// and is consulted only when the client supplied no direct key for the
/// selected provider. It cannot correct — or hide — a wrong direct api_id.
/// A model whose direct id is wrong needs the registry entry fixed; see the
/// gemini-3-flash-preview note in build_registry for the case where fixing
/// only this table left the direct path broken for months.
///
/// Corollary: an entry here is only ever needed when the vendor's own id and
/// OpenRouter's slug genuinely disagree *after* the prefix is applied. Once a
/// registry id is corrected to match the vendor, its entry here usually
/// becomes derivable and should be deleted rather than kept as a no-op.
pub fn openrouter_slug_override(direct_api_id: &str) -> Option<&'static str> {
    match direct_api_id {
        // The "gemini-3-flash" entry that used to sit here is gone: the
        // registry id is now "gemini-3-flash-preview", so the formula
        // produces OpenRouter's real "google/gemini-3-flash-preview"
        // unaided. Keeping it would have been a no-op that also implied the
        // direct id was still the bare form.
        //
        // Anthropic's own API ids dash-separate the minor version
        // ("claude-fable-5-1"), but OpenRouter's slug keeps the dot
        // ("anthropic/claude-fable-5.1"), so resolve_byok_route's
        // "{prefix}/{model}" formula guesses "anthropic/claude-fable-5-1" —
        // a slug that does not exist in OpenRouter's catalog. Same class of
        // mismatch as gemini-3-flash above.
        //
        // Note this is specific to the dotted minor version: plain
        // "claude-fable-5" needs no entry, because the formula's
        // "anthropic/claude-fable-5" is already OpenRouter's real slug.
        // grok-4.6 needs no entry either — its direct api_id already
        // carries the dot, so the formula yields the correct
        // "x-ai/grok-4.6".
        "claude-fable-5-1" => Some("anthropic/claude-fable-5.1"),
        // gemini-3.1-pro-preview needs no entry: like the flash rename, the
        // formula now yields "google/gemini-3.1-pro-preview" directly.
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
        assert!(e.is_vision_capable("gemini-3.1-pro-preview"));
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

    #[test]
    fn fable_5_1_and_grok_4_6_are_registered() {
        let e = RouteEngine::new();

        let fable = e.find("claude-fable-5-1").unwrap();
        assert_eq!(fable.provider, Provider::Anthropic);
        assert_eq!(fable.context_window, 1_000_000);
        assert_eq!(e.get_pricing("claude-fable-5-1").unwrap(), (1000.0, 5000.0));

        // 5.1 must outrank plain Fable 5 on quality outright, so Quality-priority
        // routing never depends on which one happens to come first in the vec.
        assert!(fable.quality_score > e.find("claude-fable-5").unwrap().quality_score);

        let grok = e.find("grok-4.6").unwrap();
        assert_eq!(grok.provider, Provider::XAI);
        assert_eq!(grok.context_window, 500_000);
        assert_eq!(e.get_pricing("grok-4.6").unwrap(), (200.0, 600.0));
    }

    fn msg(role: &str, text: &str) -> crate::connectors::ChatMessage {
        crate::connectors::ChatMessage {
            role: role.to_string(),
            content: crate::vision::MessageContent::Text(text.to_string()),
        }
    }

    #[test]
    fn short_single_turn_prompt_is_simple() {
        let m = vec![msg("user", "What is the capital of France?")];
        let shape = classify(&m, 30, Some(256), false);
        assert_eq!(shape.difficulty, Difficulty::Simple);
        assert_eq!(shape.task, TaskKind::General);
        assert!(matches!(shape.priority(), RoutingPriority::Cost));
    }

    #[test]
    fn omitted_max_tokens_does_not_block_simple() {
        // The resolved default is 1024, above SIMPLE_MAX_OUTPUT_TOKENS — if
        // classify saw that instead of None, almost nothing would ever be
        // Simple, since most clients omit max_tokens.
        let m = vec![msg("user", "Hi")];
        assert_eq!(classify(&m, 10, None, false).difficulty, Difficulty::Simple);
    }

    #[test]
    fn long_input_is_complex() {
        let m = vec![msg("user", "summarise this")];
        assert_eq!(classify(&m, 50_000, Some(256), false).difficulty, Difficulty::Complex);
    }

    #[test]
    fn large_requested_output_is_complex() {
        let m = vec![msg("user", "write me something long")];
        assert_eq!(classify(&m, 20, Some(8_000), false).difficulty, Difficulty::Complex);
    }

    #[test]
    fn deep_multi_turn_is_complex() {
        let m: Vec<_> = (0..8).map(|_| msg("user", "and then?")).collect();
        assert_eq!(classify(&m, 100, Some(256), false).difficulty, Difficulty::Complex);
    }

    #[test]
    fn mid_sized_request_is_moderate_and_keeps_todays_priority() {
        let m = vec![msg("user", "a somewhat longer question")];
        let shape = classify(&m, 2_000, Some(1024), false);
        assert_eq!(shape.difficulty, Difficulty::Moderate);
        assert!(matches!(shape.priority(), RoutingPriority::Balanced));
    }

    #[test]
    fn system_prompt_does_not_disqualify_simple() {
        let m = vec![msg("system", "You are terse."), msg("user", "2+2?")];
        assert_eq!(classify(&m, 40, Some(64), false).difficulty, Difficulty::Simple);
    }

    #[test]
    fn image_request_is_never_simple() {
        let m = vec![msg("user", "what is this?")];
        assert_ne!(classify(&m, 20, Some(64), true).difficulty, Difficulty::Simple);
    }

    #[test]
    fn code_prompts_classify_as_code() {
        let fenced = vec![msg("user", "fix this:\n```\nlet x = 1\n```")];
        assert_eq!(classify(&fenced, 40, Some(256), false).task, TaskKind::Code);

        let traceback = vec![msg("user", "got a Traceback from my script")];
        assert_eq!(classify(&traceback, 40, Some(256), false).task, TaskKind::Code);

        let prose = vec![msg("user", "Who won the 1998 world cup?")];
        assert_eq!(classify(&prose, 40, Some(256), false).task, TaskKind::General);
    }

    #[test]
    fn simple_general_routes_within_the_cheap_tier() {
        let e = RouteEngine::new();
        let shape = RequestShape { task: TaskKind::General, difficulty: Difficulty::Simple };
        let d = e.select_for_shape(shape, 300, 256, None).unwrap();

        let blended = d.model.cost_per_1m_input + d.model.cost_per_1m_output;
        assert!(
            blended <= SIMPLE_MAX_BLENDED_COST_PER_1M,
            "picked {} at {} blended", d.model.api_id, blended
        );
        assert!(d.model.quality_score >= SIMPLE_MIN_QUALITY);
        assert!(d.reason.contains("difficulty=Simple"), "reason was: {}", d.reason);
    }

    #[test]
    fn simple_path_is_cheaper_than_the_priority_alone_would_pick() {
        // Regression guard for the reason the price ceiling exists at all:
        // s_cost saturates on small requests, so Cost weighting on its own
        // ranks by quality and picks a mid-tier model. If someone deletes
        // the ceiling believing the priority is sufficient, this fails.
        let e = RouteEngine::new();
        let shape = RequestShape { task: TaskKind::General, difficulty: Difficulty::Simple };
        let with_limits = e.select_for_shape(shape, 300, 256, None).unwrap();
        let priority_only = e
            .select_reachable(300, 256, RoutingPriority::Cost, None)
            .unwrap();

        let blended = |m: &ModelConfig| m.cost_per_1m_input + m.cost_per_1m_output;
        assert!(
            blended(&with_limits.model) < blended(&priority_only.model),
            "limits picked {} ({}) but Cost priority alone picked {} ({})",
            with_limits.model.api_id, blended(&with_limits.model),
            priority_only.model.api_id, blended(&priority_only.model),
        );
    }

    #[test]
    fn simple_code_prefers_a_code_tuned_model() {
        let e = RouteEngine::new();
        let shape = RequestShape { task: TaskKind::Code, difficulty: Difficulty::Simple };
        let d = e.select_for_shape(shape, 300, 256, None).unwrap();
        assert!(
            SIMPLE_CODE_MODELS.contains(&d.model.api_id.as_str()),
            "expected a code-tuned model, got {}", d.model.api_id
        );
    }

    #[test]
    fn simple_code_falls_back_when_no_code_model_is_reachable() {
        let e = RouteEngine::new();
        // grok-code-fast-1 is xAI and codestral-2508 is Mistral — allow neither.
        let mut only_gemini = HashSet::new();
        only_gemini.insert(Provider::Gemini);
        let shape = RequestShape { task: TaskKind::Code, difficulty: Difficulty::Simple };
        let d = e.select_for_shape(shape, 300, 256, Some(&only_gemini)).unwrap();
        assert_eq!(d.model.provider, Provider::Gemini);
    }

    #[test]
    fn non_simple_shapes_route_identically_to_the_old_auto_default() {
        // The guarantee that adding classification cannot make any
        // previously-served request more expensive.
        let e = RouteEngine::new();
        for difficulty in [Difficulty::Moderate, Difficulty::Complex] {
            for task in [TaskKind::General, TaskKind::Code] {
                let shape = RequestShape { task, difficulty };
                let via_shape = e.select_for_shape(shape, 50_000, 1024, None).unwrap();
                let via_old = e
                    .select_reachable(50_000, 1024, RoutingPriority::Balanced, None)
                    .unwrap();
                assert_eq!(
                    via_shape.model.api_id, via_old.model.api_id,
                    "{difficulty:?}/{task:?} diverged from the Balanced default"
                );
            }
        }
    }

    #[test]
    fn min_quality_floor_excludes_weak_models() {
        let e = RouteEngine::new();
        let limits = SelectionLimits { min_quality: Some(0.95), ..Default::default() };
        let d = e
            .select_filtered(1_000, 256, RoutingPriority::Cost, None, limits)
            .unwrap();
        assert!(d.model.quality_score >= 0.95, "picked {}", d.model.api_id);

        // Unfloored, the same Cost-weighted query picks something weaker —
        // proving the floor is what changed the outcome.
        let unfloored = e.select_reachable(1_000, 256, RoutingPriority::Cost, None).unwrap();
        assert!(unfloored.model.quality_score < 0.95);
    }

    #[test]
    fn price_ceiling_excludes_expensive_models() {
        let e = RouteEngine::new();
        let limits = SelectionLimits { max_blended_cost_per_1m: Some(60.0), ..Default::default() };
        let d = e
            .select_filtered(300, 256, RoutingPriority::Quality, None, limits)
            .unwrap();
        assert!(d.model.cost_per_1m_input + d.model.cost_per_1m_output <= 60.0);

        // Quality priority uncapped goes straight to a flagship.
        let uncapped = e.select_reachable(300, 256, RoutingPriority::Quality, None).unwrap();
        assert!(uncapped.model.cost_per_1m_input + uncapped.model.cost_per_1m_output > 60.0);
    }

    #[test]
    fn unsatisfiable_limits_still_serve_the_request() {
        // select_filtered itself fails when nothing clears the limits...
        let e = RouteEngine::new();
        let impossible = SelectionLimits { min_quality: Some(1.5), ..Default::default() };
        assert!(e
            .select_filtered(1_000, 256, RoutingPriority::Cost, None, impossible)
            .is_err());

        // ...but the simple path must never fail for that reason. Restrict
        // to a provider whose cheapest model sits below SIMPLE_MIN_QUALITY
        // and confirm a decision still comes back.
        let mut only_mistral = HashSet::new();
        only_mistral.insert(Provider::Mistral);
        let shape = RequestShape { task: TaskKind::General, difficulty: Difficulty::Simple };
        assert!(e.select_for_shape(shape, 300, 256, Some(&only_mistral)).is_ok());
    }

    #[test]
    fn normalize_model_id_strips_claude_code_bracket_suffix() {
        // The case that motivated this: Claude Code's long-context variant.
        assert_eq!(normalize_model_id("claude-opus-5[1m]"), "claude-opus-5");
        assert_eq!(normalize_model_id("claude-sonnet-5[1m]"), "claude-sonnet-5");
        // Whitespace before the bracket is tolerated.
        assert_eq!(normalize_model_id("claude-opus-5 [1m]"), "claude-opus-5");
        // Untouched when there is no bracket — including ids that contain
        // dots and dashes, which is all of them.
        assert_eq!(normalize_model_id("claude-fable-5-1"), "claude-fable-5-1");
        assert_eq!(normalize_model_id("gemini-3.8-flash"), "gemini-3.8-flash");
        assert_eq!(normalize_model_id("auto"), "auto");
        assert_eq!(normalize_model_id(""), "");
    }

    #[test]
    fn bracketed_model_ids_resolve_against_the_registry() {
        // The actual failure this prevents: find() misses the bracketed id,
        // but resolves once normalized.
        let e = RouteEngine::new();
        assert!(e.find("claude-opus-5[1m]").is_err());
        let m = e.find(normalize_model_id("claude-opus-5[1m]")).unwrap();
        assert_eq!(m.api_id, "claude-opus-5");
        assert_eq!(m.provider, Provider::Anthropic);
    }

    #[test]
    fn newly_added_google_and_zhipu_models_are_registered() {
        let e = RouteEngine::new();

        // (api_id, cost_in, cost_out, context)
        let expected = [
            ("gemini-3.8-flash", 75.0, 375.0, 1_048_576u32),
            ("gemini-3.7-flash", 75.0, 375.0, 1_048_576),
            ("gemini-3.6-flash", 75.0, 375.0, 1_048_576),
            ("gemini-3.5-flash-lite", 30.0, 250.0, 1_048_576),
            ("gemini-2.5-flash", 30.0, 250.0, 1_048_576),
        ];
        for (id, cin, cout, ctx) in expected {
            let m = e.find(id).unwrap_or_else(|_| panic!("{id} missing"));
            assert_eq!(m.provider, Provider::Gemini, "{id}");
            assert_eq!(e.get_pricing(id).unwrap(), (cin, cout), "{id}");
            assert_eq!(m.context_window, ctx, "{id}");
            assert!(m.supports_vision, "{id} should accept image input");
        }

        let glm = e.find("glm-5.3").unwrap();
        assert_eq!(glm.provider, Provider::Zhipu);
        assert_eq!(e.get_pricing("glm-5.3").unwrap(), (140.0, 440.0));
        assert!(!glm.supports_vision, "vision ships as separate -v variants");
    }

    #[test]
    fn corrected_gemini_prices_do_not_undercut_their_successors() {
        // 3.5 Flash used to be entered at 75/450, cheaper-looking than the
        // 3.6/3.7/3.8 Flash models that actually replaced it at 75/375.
        let e = RouteEngine::new();
        let blended = |id: &str| {
            let m = e.find(id).unwrap();
            m.cost_per_1m_input + m.cost_per_1m_output
        };
        assert!(blended("gemini-3.5-flash") > blended("gemini-3.8-flash"));
        // And 3.1 Flash-Lite must no longer share 2.5 Flash-Lite's price.
        assert!(blended("gemini-3.1-flash-lite") > blended("gemini-2.5-flash-lite"));
    }

    #[test]
    fn newly_added_models_need_no_openrouter_slug_override() {
        // Their ids already produce the correct "{prefix}/{model}" slug, so
        // an override entry would be dead weight. Verified against
        // OpenRouter's live catalog.
        for id in [
            "gemini-3.8-flash",
            "gemini-3.7-flash",
            "gemini-3.6-flash",
            "gemini-3.5-flash-lite",
            "gemini-2.5-flash",
            "glm-5.3",
        ] {
            assert_eq!(openrouter_slug_override(id), None, "{id}");
        }
    }

    #[test]
    fn astra_is_registered_but_not_auto_selectable() {
        let e = RouteEngine::new();

        // Nameable and priceable: a Trusted Access client can request it.
        let m = e.find("gpt-6-astra").unwrap();
        assert_eq!(m.provider, Provider::OpenAI);
        // Corrected from 1_100_000 — OpenAI publishes 1,050,000, the same
        // window the gpt-5.6 family already carries here.
        assert_eq!(m.context_window, 1_050_000);
        assert!(m.supports_vision);
        assert_eq!(e.get_pricing("gpt-6-astra").unwrap(), (1000.0, 5000.0));
        assert_eq!(
            e.select_provider("gpt-6-astra", false, false).unwrap(),
            Provider::OpenAI
        );

        // But disabled, so it is neither advertised nor auto-selected while
        // general API access is still rolling out.
        assert!(!m.enabled);
        assert!(!e.list_enabled().iter().any(|m| m.api_id == "gpt-6-astra"));

        // Even a 1M-token Quality-priority request must not land on it.
        let d = e
            .select_reachable(900_000, 4096, RoutingPriority::Quality, None)
            .unwrap();
        assert_ne!(d.model.api_id, "gpt-6-astra");
    }

    #[test]
    fn anthropic_dotted_minor_version_needs_a_slug_override() {
        // The "{prefix}/{model}" formula would guess
        // "anthropic/claude-fable-5-1", which OpenRouter does not carry.
        assert_eq!(
            openrouter_slug_override("claude-fable-5-1"),
            Some("anthropic/claude-fable-5.1")
        );

        // These two the formula already gets right, so they must stay
        // un-overridden — an entry here would be dead weight at best and
        // wrong the day OpenRouter renames something.
        assert_eq!(openrouter_slug_override("grok-4.6"), None);
        assert_eq!(openrouter_slug_override("claude-fable-5"), None);
    }

    #[test]
    fn astra_long_context_tier_applies_only_above_the_threshold() {
        let e = RouteEngine::new();

        // Base rates below and at the threshold — the vendor's wording is
        // "more than 272K", so 272_000 exactly is still short-context.
        assert_eq!(e.get_pricing_for("gpt-6-astra", 0).unwrap(), (1000.0, 5000.0));
        assert_eq!(e.get_pricing_for("gpt-6-astra", 272_000).unwrap(), (1000.0, 5000.0));

        // One token over: 2x input, 1.5x output, for the whole request.
        assert_eq!(
            e.get_pricing_for("gpt-6-astra", 272_001).unwrap(),
            (2000.0, 7500.0)
        );

        // get_pricing stays on the base rates regardless — it is the
        // display/reporting accessor and has no input size to reason about.
        assert_eq!(e.get_pricing("gpt-6-astra").unwrap(), (1000.0, 5000.0));
    }

    #[test]
    fn models_without_a_tier_price_identically_either_way() {
        // The property that lets callers use get_pricing_for
        // unconditionally instead of branching on whether the selected
        // model happens to have tiered pricing.
        let e = RouteEngine::new();
        for m in e.list_enabled() {
            if m.long_context_tier.is_some() {
                continue;
            }
            assert_eq!(
                e.get_pricing_for(&m.api_id, 5_000_000).unwrap(),
                e.get_pricing(&m.api_id).unwrap(),
                "{} has no tier but priced differently at 5M input", m.api_id
            );
        }
    }

    #[test]
    fn gemini_preview_suffixed_ids_are_registered_under_their_real_names() {
        // Both were sending an id generativelanguage 404s on. Google
        // publishes the "-preview" form for each; the bare ids must be gone
        // so a stale caller fails loudly at routing rather than silently at
        // the provider.
        let e = RouteEngine::new();

        for id in ["gemini-3.1-pro-preview", "gemini-3-flash-preview"] {
            let m = e.find(id).unwrap_or_else(|_| panic!("{id} missing"));
            assert_eq!(m.provider, Provider::Gemini, "{id}");
            assert!(m.enabled, "{id}");
        }
        assert!(e.find("gemini-3.1-pro").is_err());
        assert!(e.find("gemini-3-flash").is_err());

        // And with the ids corrected, the "{prefix}/{model}" formula reaches
        // OpenRouter's real slugs unaided — so neither needs an override.
        // The gemini-3-flash entry that used to live there is deleted.
        assert_eq!(openrouter_slug_override("gemini-3-flash-preview"), None);
        assert_eq!(openrouter_slug_override("gemini-3.1-pro-preview"), None);
        assert_eq!(openrouter_slug_override("gemini-3-flash"), None);
    }

    #[test]
    fn task_routing_answer_question_still_resolves_after_the_gemini_rename() {
        // select_for_task looks its preferred model up by literal id, so a
        // registry rename silently demotes the task to score-based fallback
        // unless the table is updated with it.
        let e = RouteEngine::new();
        let d = e.select_for_task(MeetingTask::AnswerQuestion, 5_000, None).unwrap();
        assert_eq!(d.model.api_id, "gemini-3-flash-preview");
    }

    #[test]
    fn retired_and_unlisted_models_are_not_routable() {
        // kimi-k2.5 was retired 2026-08-31 (404s); grok-4.1-fast's family
        // was retired 2026-05-15; grok-code-fast-1 is absent from xAI's
        // current list. All three were `enabled: true`, so auto/task routing
        // could pick a model that cannot answer.
        let e = RouteEngine::new();
        for id in ["kimi-k2.5", "grok-4.1-fast", "grok-code-fast-1"] {
            // Still findable, so historical request_logs rows resolve for
            // pricing and display.
            assert!(e.find(id).is_ok(), "{id} should remain in the registry");
            assert!(!e.find(id).unwrap().enabled, "{id} must be disabled");
            assert!(
                !e.list_enabled().iter().any(|m| m.api_id == id),
                "{id} must not be advertised by /v1/models"
            );
        }
    }

    #[test]
    fn simple_code_survives_grok_code_fast_1_being_disabled() {
        // SIMPLE_CODE_MODELS still lists grok-code-fast-1 first; the loop
        // gates on `enabled`, so this must fall through to codestral-2508
        // rather than returning a dead model or erroring.
        let e = RouteEngine::new();
        let shape = RequestShape { task: TaskKind::Code, difficulty: Difficulty::Simple };
        let d = e.select_for_shape(shape, 300, 256, None).unwrap();
        assert_eq!(d.model.api_id, "codestral-2508");
    }

    #[test]
    fn mistral_ids_are_dated_not_bare_product_names() {
        // Mistral's API keys models by date code; the bare names were
        // product labels and 404'd on the direct path.
        let e = RouteEngine::new();
        let expected = [
            ("mistral-large-2512", "mistral-large-3"),
            ("mistral-small-2603", "mistral-small-4"),
            ("codestral-2508", "codestral-2"),
            ("ministral-8b-2512", "ministral-8b"),
        ];
        for (dated, bare) in expected {
            let m = e.find(dated).unwrap_or_else(|_| panic!("{dated} missing"));
            assert_eq!(m.provider, Provider::Mistral, "{dated}");
            assert!(m.enabled, "{dated}");
            assert!(e.find(bare).is_err(), "{bare} should no longer resolve");
        }

        // Dated ids, not "-latest" aliases: every entry carries hand-tuned
        // figures for one release, which a floating alias would silently
        // invalidate.
        for alias in [
            "mistral-large-latest",
            "mistral-small-latest",
            "codestral-latest",
            "ministral-8b-latest",
        ] {
            assert!(e.find(alias).is_err(), "{alias} must not be registered");
        }
    }

    #[test]
    fn simple_code_preference_survives_the_codestral_rename() {
        // SIMPLE_CODE_MODELS resolves preferred models by literal id, so a
        // registry rename that misses it silently drops Simple+Code routing
        // to the score-based fallback instead of erroring.
        let e = RouteEngine::new();
        for id in SIMPLE_CODE_MODELS {
            assert!(e.find(id).is_ok(), "{id} in SIMPLE_CODE_MODELS does not resolve");
        }
    }

    #[test]
    fn xai_context_windows_match_published_figures() {
        // 4.5 and 4.20 were entered at 2M, a figure xAI publishes for
        // neither — the effect was admitting 1M-2M requests xAI rejects.
        let e = RouteEngine::new();
        let ctx = |id: &str| e.find(id).unwrap().context_window;
        assert_eq!(ctx("grok-4.6"), 500_000);
        assert_eq!(ctx("grok-4.5"), 500_000);
        assert_eq!(ctx("grok-4.3"), 1_000_000);
        assert_eq!(ctx("grok-4.20"), 1_000_000);

        // No enabled Grok claims more than 1M, so nothing in the family
        // absorbs an overflow from the others.
        for m in e.list_enabled().iter().filter(|m| m.provider == Provider::XAI) {
            assert!(m.context_window <= 1_000_000, "{} claims {}", m.api_id, m.context_window);
        }
    }
}
