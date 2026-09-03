// =============================================================================
// src/supercompress.rs — RouterFuel
//
// Reduces input token cost by trimming the prompt before it reaches the
// target model. This file implements TIER 1 ONLY: non-LLM, lossless
// transformations that cost nothing and add no latency.
//
//   1. Whitespace normalization — outside fenced code blocks only
//   2. Exact-duplicate message dedup — the real win on agent traffic that
//      resends the same large tool output every turn
//
// Tier 2 (LLM summarization of older context) is deliberately NOT here.
// The economics have to be checked per request first, because RouterFuel is
// pure BYOK: a compression call is billed to the *client's* key, so it only
// pays when
//
//     (1 - r) * P_target_in  >  P_comp_in + r * P_comp_out
//
// where r is the compressed/original ratio. The compressor's *output* price
// dominates, which rules out most same-provider options — e.g. compressing
// a 100k prompt to 50k for claude-opus-5 using claude-haiku-4-5 costs 35c
// to save 25c, a net loss of 10c. Cheap cross-provider compressors clear
// the bar, but sending the prompt to a second vendor is a data-egress
// change no gateway should make on a client's behalf without being asked.
//
// Hence the audit surface below: every request gets a CompressionReport
// recording what Tier 1 actually saved and whether the request was even a
// Tier 2 candidate, so the decision to build Tier 2 can be made from real
// traffic instead of guesswork.
//
// SAFETY RAILS (why this is genuinely lossless, not just cheap):
//   - Indentation is never touched. It is semantic in Python and YAML, and
//     in Markdown it distinguishes nested lists and indented code blocks.
//   - Anything inside a ``` fence is copied through byte-for-byte.
//   - Multimodal messages are never rewritten or deduped — as_text() drops
//     image parts, so two messages with identical text but different
//     images would otherwise dedup into one and silently lose an image.
//   - The final message is never dropped. It is the actual request.
//   - Dedup only fires inside a run of same-role messages. Removing one
//     element of a strictly alternating conversation always leaves its
//     neighbours sharing a role, which the Anthropic Messages API rejects.
//     So on ordinary alternating chat traffic dedup deliberately does
//     nothing; it earns its keep on the agent pattern, where the same large
//     tool result arrives as several consecutive messages. Expect the audit
//     records to show dedup firing on a narrow slice of traffic — that is
//     the design, not a bug.
// =============================================================================

use crate::connectors::ChatMessage;
use crate::vision::MessageContent;
use lazy_static::lazy_static;
use serde::Deserialize;
use std::collections::HashSet;
use tracing::{info, warn};

/// Input-token size above which a request would be worth considering for
/// Tier 2. Not a gate on Tier 1 — Tier 1 is free and instant, so it runs on
/// everything. This only classifies audit records, so the Tier 2 decision
/// can be made against the share of traffic that could actually profit.
const TIER2_MIN_INPUT_TOKENS: u32 = 5_000;

/// Minimum message length before dedup will drop a duplicate. Collapsing
/// two identical short messages saves nothing measurable while still
/// changing the conversation the model sees, so it isn't worth doing. The
/// case worth catching is a large repeated tool output.
const DEDUP_MIN_CHARS: usize = 200;

/// Operator-level switch, read once from `ROUTERFUEL_SUPERCOMPRESS_MODE`.
///
/// `audit` is the **default**: it computes and logs the full report but
/// sends the original messages untouched, so a new deployment measures the
/// effect on its own traffic before changing a single request. Flip to `on`
/// once the audit records look right. `off` disables the feature entirely.
///
/// Defaulting to `audit` rather than `on` is deliberate even though Tier 1
/// is lossless: "lossless" is a claim about this code, and the audit log is
/// how an operator verifies that claim against their own traffic instead of
/// taking our word for it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ServerMode {
    Off,
    Audit,
    On,
}

lazy_static! {
    static ref SERVER_MODE: ServerMode = {
        match std::env::var("ROUTERFUEL_SUPERCOMPRESS_MODE") {
            Ok(v) => match v.trim().to_lowercase().as_str() {
                "off" => ServerMode::Off,
                "audit" => ServerMode::Audit,
                "on" => ServerMode::On,
                other => {
                    warn!(
                        value = other,
                        "unrecognized ROUTERFUEL_SUPERCOMPRESS_MODE — expected off|audit|on, \
                         falling back to audit (measure, change nothing)"
                    );
                    ServerMode::Audit
                }
            },
            Err(_) => ServerMode::Audit,
        }
    };
}

/// The configured mode. `compress` takes the mode as an argument rather
/// than reading this itself, so it stays a pure function — a global read
/// inside it would make the applied path untestable, since `lazy_static`
/// resolves once per process and test order is nondeterministic.
pub fn server_mode() -> ServerMode {
    *SERVER_MODE
}

/// Per-request `supercompress` object on the chat-completions body.
///
/// An object rather than a bare bool so the individual levers stay
/// addressable as Tier 2 arrives — a client will want to accept lossless
/// trimming while declining anything that rewrites meaning.
///
/// `deny_unknown_fields` is deliberate: a client that misspells a lever
/// should get a clear 400 rather than silently not getting the behaviour
/// they asked for. That matters more than forward-compatibility for a
/// feature whose whole job is changing what the model sees.
#[derive(Debug, Clone, Copy, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SupercompressOptions {
    /// Master switch for this request.
    #[serde(default = "enabled_by_default")]
    pub enabled: bool,
    /// Collapse blank-line runs and strip trailing whitespace outside fences.
    #[serde(default = "enabled_by_default")]
    pub whitespace: bool,
    /// Drop later exact-duplicate messages.
    #[serde(default = "enabled_by_default")]
    pub dedup: bool,
    /// Compute and log the report, but send the original messages.
    #[serde(default)]
    pub audit_only: bool,
}

fn enabled_by_default() -> bool {
    true
}

impl Default for SupercompressOptions {
    /// What a request with no `supercompress` field gets. Tier 1 is
    /// lossless, so it is on by default; nothing lossy exists yet to opt
    /// into.
    fn default() -> Self {
        Self {
            enabled: true,
            whitespace: true,
            dedup: true,
            audit_only: false,
        }
    }
}

#[derive(Debug, Clone, Copy, Default)]
pub struct CompressionReport {
    pub original_tokens: u32,
    pub compressed_tokens: u32,
    pub messages_before: usize,
    pub messages_dropped: usize,
    pub whitespace_chars_removed: usize,
    /// False when the report was produced in audit mode — the savings below
    /// are what *would* have been saved, and the original prompt was sent.
    pub applied: bool,
    /// Whether this request is large enough that Tier 2 could plausibly pay.
    pub tier2_candidate: bool,
}

impl CompressionReport {
    pub fn saved_tokens(&self) -> u32 {
        self.original_tokens.saturating_sub(self.compressed_tokens)
    }

    pub fn saved_pct(&self) -> f64 {
        if self.original_tokens == 0 {
            return 0.0;
        }
        (self.saved_tokens() as f64 / self.original_tokens as f64) * 100.0
    }

    /// Input-side saving in cents, at the target model's own input rate.
    pub fn saved_cents(&self, cost_per_1m_input: f64) -> f64 {
        (self.saved_tokens() as f64 / 1_000_000.0) * cost_per_1m_input
    }

    pub fn changed_anything(&self) -> bool {
        self.messages_dropped > 0 || self.whitespace_chars_removed > 0
    }
}

/// Applies Tier 1 compression.
///
/// Returns `Some(messages)` when the caller should send those instead of the
/// original, and `None` when the original should be sent unchanged (nothing
/// to do, disabled, or audit mode). The report is always returned so every
/// request produces an audit record either way.
pub fn compress(
    messages: &[ChatMessage],
    opts: SupercompressOptions,
    original_tokens: u32,
    mode: ServerMode,
) -> (Option<Vec<ChatMessage>>, CompressionReport) {
    let mut report = CompressionReport {
        original_tokens,
        compressed_tokens: original_tokens,
        messages_before: messages.len(),
        tier2_candidate: original_tokens >= TIER2_MIN_INPUT_TOKENS,
        ..Default::default()
    };

    if mode == ServerMode::Off || !opts.enabled {
        return (None, report);
    }

    let mut working = messages.to_vec();

    if opts.whitespace {
        for m in working.iter_mut() {
            // Only plain-text messages. A Parts message's text lives in
            // individual parts alongside image parts; rewriting it through
            // as_text() would flatten that structure and drop the images.
            let (original_len, normalized) = match &m.content {
                MessageContent::Text(s) => (s.len(), normalize_whitespace(s)),
                MessageContent::Parts(_) => continue,
            };
            if normalized.len() < original_len {
                report.whitespace_chars_removed += original_len - normalized.len();
                m.content = MessageContent::Text(normalized);
            }
        }
    }

    if opts.dedup {
        let (deduped, dropped) = dedup_exact(&working);
        report.messages_dropped = dropped;
        working = deduped;
    }

    if !report.changed_anything() {
        return (None, report);
    }

    // Recount so the spend reservation and audit record reflect what is
    // actually being sent, not the pre-compression estimate.
    report.compressed_tokens =
        crate::tokens::count_request_tokens(&working, "").unwrap_or(original_tokens);

    let audit_only = opts.audit_only || mode == ServerMode::Audit;
    report.applied = !audit_only;

    if audit_only {
        (None, report)
    } else {
        (Some(working), report)
    }
}

/// Normalizes whitespace outside fenced code blocks.
///
/// Only transformations that cannot change meaning:
///   - collapse runs of 3+ newlines down to 2 (one blank line)
///   - strip trailing spaces and tabs from each line
///   - drop leading blank lines and all trailing whitespace
///
/// Leading indentation is never stripped, and fenced regions pass through
/// untouched.
fn normalize_whitespace(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut in_fence = false;
    let mut blank_run = 0usize;
    let mut first = true;

    for line in text.split('\n') {
        let is_fence_marker = line.trim_start().starts_with("```");

        if is_fence_marker {
            in_fence = !in_fence;
            blank_run = 0;
        } else if in_fence {
            // Inside a fence: byte-for-byte passthrough.
        } else {
            let trimmed = line.trim_end_matches([' ', '\t']);
            if trimmed.is_empty() {
                blank_run += 1;
                // Keep at most one blank line, i.e. collapse 3+ newlines
                // into 2. Also drops leading blank lines, since `first` is
                // still true and nothing has been emitted yet.
                if blank_run > 1 || first {
                    continue;
                }
            } else {
                blank_run = 0;
            }
            if !first {
                out.push('\n');
            }
            out.push_str(trimmed);
            first = false;
            continue;
        }

        if !first {
            out.push('\n');
        }
        out.push_str(line);
        first = false;
    }

    // Trailing whitespace only — trimming the start would eat the first
    // line's indentation, which may be semantic.
    out.trim_end().to_string()
}

/// Drops later exact `(role, text)` duplicates.
///
/// Skips messages that are short, multimodal, final, or whose removal would
/// place two same-role messages back to back. Returns the surviving
/// messages and how many were dropped.
fn dedup_exact(messages: &[ChatMessage]) -> (Vec<ChatMessage>, usize) {
    let mut out: Vec<ChatMessage> = Vec::with_capacity(messages.len());
    let mut seen: HashSet<(String, String)> = HashSet::new();
    let mut dropped = 0usize;

    for (i, m) in messages.iter().enumerate() {
        let is_final = i + 1 == messages.len();

        // Never dedup a multimodal message: as_text() drops its image
        // parts, so two different images with the same caption would look
        // identical here.
        if m.content.has_image() {
            out.push(m.clone());
            continue;
        }

        let text = m.content.as_text();
        let big_enough = text.len() >= DEDUP_MIN_CHARS;
        let already_seen = !seen.insert((m.role.clone(), text));

        // A duplicate may only be dropped if it sits inside a run of
        // same-role messages.
        //
        // Removing one element of a strictly alternating conversation
        // always leaves its two neighbours sharing a role, and the
        // Anthropic Messages API requires alternating roles — so a
        // "lossless" dedup there would turn a working request into a 400.
        // That means dedup correctly never fires on well-formed chat
        // traffic. Inside a same-role run it is safe, because the run
        // already exists and shortening it introduces no new adjacency —
        // and that run is exactly the case worth catching: an agent loop
        // resending the same large tool result as consecutive messages.
        let prev_same_role = i > 0 && messages[i - 1].role == m.role;
        let next_same_role = messages.get(i + 1).map_or(false, |n| n.role == m.role);
        let in_same_role_run = prev_same_role || next_same_role;

        if already_seen && big_enough && !is_final && in_same_role_run {
            dropped += 1;
            continue;
        }

        out.push(m.clone());
    }

    (out, dropped)
}

/// Emits the audit record. One line per request — this is the whole point
/// of shipping Tier 1 with measurement attached: it produces the traffic
/// data needed to decide whether Tier 2 is worth building, at zero cost.
pub fn log_audit(
    request_id: &str,
    target_model: &str,
    cost_per_1m_input: f64,
    report: &CompressionReport,
    mode: ServerMode,
) {
    info!(
        request_id = %request_id,
        target_model = %target_model,
        mode = ?mode,
        applied = report.applied,
        original_tokens = report.original_tokens,
        compressed_tokens = report.compressed_tokens,
        saved_tokens = report.saved_tokens(),
        saved_pct = format!("{:.2}", report.saved_pct()),
        saved_cents = format!("{:.5}", report.saved_cents(cost_per_1m_input)),
        messages_before = report.messages_before,
        messages_dropped = report.messages_dropped,
        whitespace_chars_removed = report.whitespace_chars_removed,
        tier2_candidate = report.tier2_candidate,
        "supercompress audit"
    );
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vision::{ContentPart, ImageUrl, PartKind};

    fn msg(role: &str, text: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: MessageContent::Text(text.to_string()),
        }
    }

    fn image_msg(role: &str, caption: &str, url: &str) -> ChatMessage {
        ChatMessage {
            role: role.to_string(),
            content: MessageContent::Parts(vec![
                ContentPart {
                    part_type: "text".to_string(),
                    kind: PartKind::Text { text: caption.to_string() },
                },
                ContentPart {
                    part_type: "image_url".to_string(),
                    kind: PartKind::ImageUrl {
                        image_url: ImageUrl { url: url.to_string(), detail: None },
                    },
                },
            ]),
        }
    }

    fn big(tag: &str) -> String {
        format!("{tag} {}", "x".repeat(DEDUP_MIN_CHARS))
    }

    /// The shape dedup actually targets: an agent loop delivering the same
    /// large tool result as consecutive same-role messages.
    fn agent_convo(dup: &str) -> Vec<ChatMessage> {
        vec![
            msg("user", "run the tool"),
            msg("assistant", "calling it"),
            msg("user", dup),
            msg("user", dup),
            msg("user", "what now?"),
        ]
    }

    // ---- whitespace ----

    #[test]
    fn collapses_blank_line_runs() {
        assert_eq!(normalize_whitespace("a\n\n\n\n\nb"), "a\n\nb");
    }

    #[test]
    fn strips_trailing_whitespace_per_line() {
        assert_eq!(normalize_whitespace("a   \nb\t\t"), "a\nb");
    }

    #[test]
    fn preserves_leading_indentation() {
        // Semantic in Python and YAML — must survive verbatim.
        let src = "def f():\n    if x:\n        return 1";
        assert_eq!(normalize_whitespace(src), src);
    }

    #[test]
    fn fenced_code_passes_through_byte_for_byte() {
        let src = "text\n\n\n```python\ndef f():\n    x = 1\n\n\n\n    return x   \n```\n\n\ndone";
        let got = normalize_whitespace(src);
        // The fence body keeps its blank-line run and its trailing spaces.
        assert!(got.contains("    x = 1\n\n\n\n    return x   \n"), "got: {got:?}");
        // Prose outside the fence is still collapsed.
        assert!(got.starts_with("text\n\n```python"), "got: {got:?}");
        assert!(got.ends_with("```\n\ndone"), "got: {got:?}");
    }

    #[test]
    fn drops_leading_blank_lines_but_not_indentation() {
        assert_eq!(normalize_whitespace("\n\n    indented"), "    indented");
    }

    #[test]
    fn whitespace_normalization_is_idempotent() {
        let src = "a   \n\n\n\nb\n```\n  code  \n```\n\n";
        let once = normalize_whitespace(src);
        assert_eq!(normalize_whitespace(&once), once);
    }

    // ---- dedup ----

    #[test]
    fn drops_a_large_duplicate_inside_a_same_role_run() {
        let m = agent_convo(&big("tool-output"));
        let (out, dropped) = dedup_exact(&m);
        assert_eq!(dropped, 1);
        assert_eq!(out.len(), 4);
        // The surviving copy is still there, and the final message is intact.
        assert!(out[2].content.as_text().starts_with("tool-output"));
        assert_eq!(out[3].content.as_text(), "what now?");
    }

    #[test]
    fn alternating_conversation_is_never_deduped() {
        // Documents the real constraint: dropping any single message from a
        // strictly alternating conversation leaves two same-role messages
        // adjacent, which Anthropic rejects. Dedup must decline here.
        let dup = big("ctx");
        let m = vec![
            msg("user", &dup),
            msg("assistant", "a1"),
            msg("user", &dup),
            msg("assistant", "a2"),
            msg("user", "go"),
        ];
        let (out, dropped) = dedup_exact(&m);
        assert_eq!(dropped, 0);
        let roles: Vec<_> = out.iter().map(|m| m.role.as_str()).collect();
        assert_eq!(roles, vec!["user", "assistant", "user", "assistant", "user"]);
    }

    #[test]
    fn keeps_short_duplicates() {
        // In a same-role run, so only DEDUP_MIN_CHARS stops this — no
        // measurable saving, so leave the conversation structure alone.
        let m = vec![msg("user", "hi"), msg("user", "hi"), msg("user", "final")];
        let (out, dropped) = dedup_exact(&m);
        assert_eq!(dropped, 0);
        assert_eq!(out.len(), m.len());
    }

    #[test]
    fn never_drops_the_final_message() {
        // A same-role run whose duplicate *is* the last message: every
        // other condition to drop it holds, so only the final-message rail
        // saves it.
        let dup = big("same");
        let m = vec![msg("user", "setup"), msg("user", &dup), msg("user", &dup)];
        let (out, dropped) = dedup_exact(&m);
        assert_eq!(dropped, 0, "the final message is the actual request");
        assert_eq!(out.len(), 3);
    }

    #[test]
    fn never_dedups_multimodal_messages() {
        // Identical captions, different images, in a same-role run — every
        // other condition to drop holds, so only the multimodal rail stops
        // an image being silently lost.
        let m = vec![
            msg("user", "compare these"),
            msg("assistant", "ok"),
            image_msg("user", &big("look"), "https://example.com/a.png"),
            image_msg("user", &big("look"), "https://example.com/b.png"),
            msg("user", "which?"),
        ];
        let (out, dropped) = dedup_exact(&m);
        assert_eq!(dropped, 0, "deduping these would silently lose an image");
        assert_eq!(out.len(), m.len());
    }

    // ---- compress() orchestration ----

    #[test]
    fn compress_reports_savings_and_returns_new_messages() {
        let m = agent_convo(&big("payload"));
        let original = crate::tokens::count_request_tokens(&m, "").unwrap();
        let (out, report) = compress(&m, SupercompressOptions::default(), original, ServerMode::On);

        let out = out.expect("a dropped duplicate should produce new messages");
        assert_eq!(out.len(), 4);
        assert_eq!(report.messages_dropped, 1);
        assert!(report.saved_tokens() > 0);
        assert!(report.applied);
        assert!(report.compressed_tokens < report.original_tokens);
    }

    #[test]
    fn audit_only_reports_savings_but_sends_the_original() {
        let m = agent_convo(&big("payload"));
        let original = crate::tokens::count_request_tokens(&m, "").unwrap();
        let opts = SupercompressOptions { audit_only: true, ..Default::default() };
        let (out, report) = compress(&m, opts, original, ServerMode::On);

        assert!(out.is_none(), "audit mode must not change what is sent");
        assert!(report.saved_tokens() > 0, "but it must still report the saving");
        assert!(!report.applied);
    }

    #[test]
    fn disabled_does_nothing() {
        let m = agent_convo(&big("payload"));
        let opts = SupercompressOptions { enabled: false, ..Default::default() };
        let (out, report) = compress(&m, opts, 100, ServerMode::On);
        assert!(out.is_none());
        assert_eq!(report.saved_tokens(), 0);
        assert!(!report.changed_anything());
    }

    #[test]
    fn dedup_can_be_declined_while_whitespace_still_applies() {
        // The reason the flag is an object rather than a bare bool.
        let m = vec![
            msg("user", "run the tool"),
            msg("user", &format!("{}   \n\n\n\ntail", big("payload"))),
            msg("user", &format!("{}   \n\n\n\ntail", big("payload"))),
            msg("user", "what now?"),
        ];
        let original = crate::tokens::count_request_tokens(&m, "").unwrap();
        let opts = SupercompressOptions { dedup: false, ..Default::default() };
        let (out, report) = compress(&m, opts, original, ServerMode::On);

        assert_eq!(report.messages_dropped, 0, "dedup was declined");
        assert!(report.whitespace_chars_removed > 0, "whitespace still ran");
        assert_eq!(out.expect("whitespace changed something").len(), 4);
    }

    #[test]
    fn nothing_to_do_returns_none_without_claiming_savings() {
        let m = vec![msg("user", "already tight")];
        let original = crate::tokens::count_request_tokens(&m, "").unwrap();
        let (out, report) = compress(&m, SupercompressOptions::default(), original, ServerMode::On);
        assert!(out.is_none());
        assert_eq!(report.saved_tokens(), 0);
        assert_eq!(report.compressed_tokens, report.original_tokens);
    }

    #[test]
    fn tier2_candidate_reflects_the_token_threshold() {
        let m = vec![msg("user", "small")];
        let (_, small) = compress(&m, SupercompressOptions::default(), 100, ServerMode::On);
        assert!(!small.tier2_candidate);

        let (_, large) = compress(&m, SupercompressOptions::default(), TIER2_MIN_INPUT_TOKENS, ServerMode::On);
        assert!(large.tier2_candidate);
    }

    #[test]
    fn saved_cents_uses_the_target_input_rate() {
        let r = CompressionReport {
            original_tokens: 1_000_000,
            compressed_tokens: 500_000,
            ..Default::default()
        };
        assert_eq!(r.saved_tokens(), 500_000);
        assert_eq!(r.saved_pct(), 50.0);
        // 500k tokens saved at 500 cents/1M input = 250 cents.
        assert_eq!(r.saved_cents(500.0), 250.0);
    }

    #[test]
    fn options_reject_unknown_fields() {
        // A misspelled lever must fail loudly rather than silently not
        // applying — this feature changes what the model sees.
        let ok: Result<SupercompressOptions, _> =
            serde_json::from_str(r#"{"dedup": false}"#);
        assert!(ok.is_ok());
        let typo: Result<SupercompressOptions, _> =
            serde_json::from_str(r#"{"dedupe": false}"#);
        assert!(typo.is_err(), "unknown field should be rejected");
    }

    #[test]
    fn shipped_default_mode_is_audit() {
        // Pins the shipped default: measure first, change nothing until an
        // operator opts in. Skipped if the running shell overrides it.
        if std::env::var("ROUTERFUEL_SUPERCOMPRESS_MODE").is_ok() {
            return;
        }
        assert_eq!(server_mode(), ServerMode::Audit);
    }

    #[test]
    fn audit_mode_measures_but_sends_the_original() {
        // Same as the per-request audit_only flag, but operator-wide — and
        // it overrides default per-request options rather than deferring to
        // them.
        let m = agent_convo(&big("payload"));
        let original = crate::tokens::count_request_tokens(&m, "").unwrap();
        let (out, report) =
            compress(&m, SupercompressOptions::default(), original, ServerMode::Audit);

        assert!(out.is_none(), "audit mode must not change what is sent");
        assert!(report.saved_tokens() > 0, "but it must still measure");
        assert!(!report.applied);
    }

    #[test]
    fn off_mode_does_not_even_measure() {
        let m = agent_convo(&big("payload"));
        let original = crate::tokens::count_request_tokens(&m, "").unwrap();
        let (out, report) =
            compress(&m, SupercompressOptions::default(), original, ServerMode::Off);

        assert!(out.is_none());
        assert_eq!(report.saved_tokens(), 0);
        assert!(!report.changed_anything());
    }

    #[test]
    fn absent_options_default_to_tier1_on() {
        let o: SupercompressOptions = serde_json::from_str("{}").unwrap();
        assert!(o.enabled && o.whitespace && o.dedup && !o.audit_only);
    }
}
