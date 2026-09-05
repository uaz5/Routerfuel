-- =============================================================================
-- RouterFuel  —  Time-to-first-byte: close migration 009's streaming gap
-- Run automatically by sqlx::migrate! on startup
--
-- WHY: migration 009 split provider time from gateway time and defined
-- latency_ms as "provider round-trip only", but left one path unfixed — the
-- streaming path wrote stream_handler wall-clock into latency_ms, so a long
-- generation read as huge "provider latency" and AVG(latency_ms) was still
-- averaging incompatible quantities for any range containing streaming rows.
--
-- The obvious fix — put a second Instant around the upstream call and write
-- that into latency_ms — does NOT work, and the reason is worth recording
-- because it is the whole justification for this migration existing:
--
--   On a stream there is no moment corresponding to "provider round-trip".
--   The response is not complete until generation ends, and that instant is
--   already what total_latency_ms measures. The only provider-only figure a
--   stream can yield is time-to-headers. That is a genuinely DIFFERENT
--   quantity from the non-streaming latency_ms (which includes generation,
--   because the connector reads the whole body before stopping its clock).
--   Writing time-to-headers into latency_ms would have made the column mean
--   two things again — exactly the defect 009 set out to remove.
--
-- So time-to-first-byte gets its own column, and streaming rows now leave
-- latency_ms NULL rather than filling it with a number that is not the thing
-- the column claims to hold:
--
--   latency_ms        = provider round-trip incl. generation. NULL on
--                       streaming paths, where no such figure exists.
--   ttfb_ms           = time from sending the upstream request to receiving
--                       response headers. Populated on streaming paths only.
--   total_latency_ms  = full handler wall-clock (unchanged from 009).
--
-- latency_ms therefore has to lose its NOT NULL DEFAULT 0. Same reasoning as
-- 009's nullable column: a stored 0 reads as a genuine measurement of zero
-- and drags AVG() down, while NULL honestly means "not measurable on this
-- path" and AVG() skips it. No backfill is needed or wanted — every existing
-- row was written by a caller that passed a real value, and the streaming
-- rows written between 009 and this migration hold a conflated figure that
-- cannot be retroactively separated. Treat streaming latency_ms values
-- predating this migration as unusable, not as provider time.
--
-- Consequence for readers: admin.rs's avg_overhead_ms is
-- AVG(total_latency_ms - latency_ms), which is NULL for a streaming row and
-- so skipped by AVG. That is correct — gateway overhead is not computable
-- without a comparable provider figure — but it does mean the overhead stat
-- now describes non-streaming traffic only. Streaming rows are still fully
-- represented by total_latency_ms and ttfb_ms.
--
-- Both /v1/chat/completions streaming (src/streaming.rs) and the native
-- /v1/messages passthrough (src/anthropic_passthrough.rs) write ttfb_ms; the
-- passthrough previously put its time-to-headers figure in latency_ms, which
-- this migration's writers correct so the column is honest on every path
-- rather than all but one.
-- =============================================================================

ALTER TABLE request_logs
    ADD COLUMN IF NOT EXISTS ttfb_ms INTEGER;

ALTER TABLE request_logs
    ALTER COLUMN latency_ms DROP DEFAULT;

ALTER TABLE request_logs
    ALTER COLUMN latency_ms DROP NOT NULL;

COMMENT ON COLUMN request_logs.latency_ms IS 'Provider round-trip in ms, including generation. NULL on streaming paths, where no equivalent figure exists — read ttfb_ms and total_latency_ms for those rows. Streaming rows written before migration 010 hold a conflated value; treat them as unusable rather than as provider time.';
COMMENT ON COLUMN request_logs.ttfb_ms IS 'Time-to-first-byte in ms: upstream request sent -> response headers received. Populated on streaming paths only (NULL elsewhere, and NULL for every row predating migration 010). Excludes time spent waiting on the gateway concurrency limiter, which counts as gateway overhead.';
