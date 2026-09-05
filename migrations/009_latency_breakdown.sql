-- =============================================================================
-- RouterFuel  —  Latency breakdown: separate provider time from gateway time
-- Run automatically by sqlx::migrate! on startup
--
-- WHY: request_logs.latency_ms could not answer "how much of this request was
-- RouterFuel versus the provider", because no column ever held total
-- wall-clock. Worse, latency_ms itself carried three different meanings
-- depending on which code path wrote the row:
--
--   handle_non_streaming success   -> provider round-trip only
--   handle_non_streaming cache hit -> handler wall-clock
--   handle_non_streaming error     -> handler wall-clock
--   streaming.rs                   -> stream_handler wall-clock, which
--                                     includes the whole generation
--
-- So AVG(latency_ms) on the admin dashboard was averaging incompatible
-- quantities. This migration adds the missing column and the writers are
-- updated to define each one exactly once:
--
--   latency_ms        = provider round-trip only
--   total_latency_ms  = full handler wall-clock
--   overhead          = total_latency_ms - latency_ms  (computed at read time,
--                       deliberately not stored — it would go stale if either
--                       input were ever corrected)
--
-- NULLABLE ON PURPOSE, not NOT NULL DEFAULT 0. A stored 0 would read as a
-- genuine measurement of zero and drag AVG() down; NULL honestly means "this
-- row predates the column" and AVG() skips it. Any query comparing gateway
-- overhead across a range spanning this migration must therefore expect NULLs
-- rather than assume every row is populated.
--
-- KNOWN GAP, tracked as a follow-up: the streaming path
-- (src/streaming.rs) still writes a conflated latency_ms, because separating
-- its provider call from its generation time needs a second Instant inside
-- stream_handler and was deliberately scoped out of this change. Until that
-- lands, filter streaming rows out of provider-latency analysis, or read only
-- total_latency_ms for them.
-- =============================================================================

ALTER TABLE request_logs
    ADD COLUMN IF NOT EXISTS total_latency_ms INTEGER;

COMMENT ON COLUMN request_logs.latency_ms       IS 'Provider round-trip only, in ms. NOTE: rows written by the streaming path still conflate this with generation time — see migration 009.';
COMMENT ON COLUMN request_logs.total_latency_ms IS 'Full handler wall-clock in ms, including auth, token counting, cache lookup and spend reservation. NULL for rows written before migration 009. Gateway overhead = total_latency_ms - latency_ms.';
