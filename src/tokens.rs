// =============================================================================
// src/tokens.rs  — RouterFuel v0.6
//
// Precision token counting using OpenAI's tiktoken tokenizer (cl100k_base).
// Used for:
//   - Pre-request: count input tokens for routing score and cost estimation
//   - Post-response: verify output tokens match what the API reported
//   - Semantic cache: normalize prompt text for embedding
//
// FIX (this revision): count_tokens() used to treat a poisoned TOKENIZER
// mutex as a permanent hard failure (`.lock().map_err(...)?`). Since this
// is called on literally every chat-completion request (streaming and
// non-streaming both call count_request_tokens), a single panic anywhere
// inside tiktoken's encode_ordinary() for any one request would poison the
// mutex forever and take down 100% of subsequent traffic with
// InternalError until the process was restarted — a much bigger blast
// radius than the equivalent bug already fixed in embedder.rs (which only
// affected semantic caching). Recovers via `poisoned.into_inner()` instead,
// same pattern as embedder.rs::embed().
// =============================================================================

use lazy_static::lazy_static;
use std::sync::Mutex;
use thiserror::Error;
use tiktoken_rs::cl100k_base;
use tracing::{debug, warn};

#[derive(Error, Debug)]
pub enum TokenError {
    #[error("Tokenizer encode failed: {0}")]
    EncodeFailed(String),
}

// Initialized once at startup — cl100k_base is used by GPT-4, Claude, DeepSeek, Gemini
lazy_static! {
    static ref TOKENIZER: Mutex<tiktoken_rs::CoreBPE> = {
        match cl100k_base() {
            Ok(bpe) => Mutex::new(bpe),
            Err(e)  => panic!("Failed to load tiktoken cl100k_base: {e}"),
        }
    };
}

/// Count tokens in a plain string
pub fn count_tokens(text: &str) -> Result<u32, TokenError> {
    // FIX: recover from a poisoned lock instead of permanently failing
    // every future call. The underlying CoreBPE is still perfectly usable
    // after one panicking call — only the lock's poison flag needs
    // clearing, exactly like embedder.rs::embed()'s Mutex<Session>.
    let tok = match TOKENIZER.lock() {
        Ok(guard) => guard,
        Err(poisoned) => {
            warn!(
                "tokenizer mutex was poisoned by a previous panic — recovering and \
                 continuing rather than permanently failing every request"
            );
            poisoned.into_inner()
        }
    };

    let tokens = tok.encode_ordinary(text);

    let n = tokens.len() as u32;
    debug!(text_len = text.len(), token_count = n, "Counted tokens");
    Ok(n)
}

/// Count tokens for a chat message including role + formatting overhead.
/// OpenAI adds ~4 tokens per message for <|im_start|>role\ncontent<|im_end|>\n
pub fn count_message_tokens(role: &str, content: &str) -> Result<u32, TokenError> {
    let role_tokens     = count_tokens(role)?;
    let content_tokens = count_tokens(content)?;
    Ok(role_tokens + content_tokens + 4)   // 4 = formatting overhead
}

/// Count total input tokens for a full message array
pub fn count_request_tokens(
    messages: &[crate::connectors::ChatMessage],
    _model:   &str,
) -> Result<u32, TokenError> {
    let mut total = 3u32;  // per-request overhead

    for msg in messages {
        // content is now text-or-multimodal (see src/vision.rs) — as_text()
        // extracts just the text segments for counting. This undercounts
        // true token cost when images are present (image tokenization is
        // provider-specific and not modeled here), which only affects the
        // routing/cost-estimate signal, not what the client is actually
        // billed by the provider.
        total = total.saturating_add(
            count_message_tokens(&msg.role, &msg.content.as_text())?
        );
    }

    debug!(
        message_count = messages.len(),
        total_tokens  = total,
        "Counted request tokens"
    );

    Ok(total)
}

/// Estimate output tokens from max_tokens param or model default
pub fn estimate_output_tokens(max_tokens: Option<u32>, model: &str) -> u32 {
    max_tokens.unwrap_or_else(|| {
        if model.contains("opus") || model.contains("5.5") { 2048 }
        else if model.contains("sonnet") || model.contains("gpt-5") { 1024 }
        else { 512 }
    })
}

/// Verify output token count.
/// Returns (counted, matches) — allows ±5 token variance for model differences.
pub fn verify_output_tokens(
    text:             &str,
    reported_tokens:  u32,
) -> Result<(u32, bool), TokenError> {
    let counted  = count_tokens(text)?;
    let variance = (counted as i32 - reported_tokens as i32).abs();
    let matches  = variance <= 5;
    debug!(counted, reported_tokens, variance, matches, "Verified output tokens");
    Ok((counted, matches))
}

#[derive(Debug, Clone, Default)]
pub struct TokenCostBreakdown {
    pub input_tokens: u32,
    pub output_tokens: u32,
    pub total_tokens: u32,
    pub cost_input_cents: f64,
    pub cost_output_cents: f64,
    pub total_cost_cents: f64,
}

impl TokenCostBreakdown {
    pub fn new(
        input_tokens: u32,
        output_tokens: u32,
        cost_per_1m_input: f64,
        cost_per_1m_output: f64,
    ) -> Self {
        let total_tokens = input_tokens + output_tokens;
        let cost_input_cents = (input_tokens as f64 / 1_000_000.0) * cost_per_1m_input;
        let cost_output_cents = (output_tokens as f64 / 1_000_000.0) * cost_per_1m_output;
        let total_cost_cents = cost_input_cents + cost_output_cents;

        Self {
            input_tokens,
            output_tokens,
            total_tokens,
            cost_input_cents,
            cost_output_cents,
            total_cost_cents,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test] fn counts_simple_string() {
        let n = count_tokens("Hello world").unwrap();
        assert!(n > 0);
    }

    #[test] fn message_has_overhead() {
        let n = count_message_tokens("user", "Hello").unwrap();
        assert!(n >= 6);  // at least role + content + 4
    }

    #[test] fn estimate_uses_max_tokens() {
        assert_eq!(estimate_output_tokens(Some(300), "any"), 300);
    }

    #[test] fn estimate_defaults_by_model() {
        assert_eq!(estimate_output_tokens(None, "claude-opus-4-7"), 2048);
        assert_eq!(estimate_output_tokens(None, "gemini-3-flash-preview"),  512);
    }

    #[test] fn survives_a_poisoned_lock() {
        // Poison the mutex deliberately, then confirm count_tokens still
        // works afterward instead of returning Err forever.
        let _ = std::panic::catch_unwind(|| {
            let _guard = TOKENIZER.lock().unwrap();
            panic!("deliberate poison for test");
        });
        let n = count_tokens("still works after a poison").unwrap();
        assert!(n > 0);
    }
}
