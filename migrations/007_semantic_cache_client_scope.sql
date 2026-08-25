-- =============================================================================
-- migrations/007_semantic_cache_client_scope.sql  — RouterFuel v0.8
--
-- FIX: semantic_cache had no client/tenant scoping at all — any client
-- could be served another client's cached response on an identical or
-- semantically-similar prompt (cross-tenant data leak). This adds
-- client_id and re-scopes the uniqueness constraint to (prompt_hash,
-- client_id) instead of prompt_hash alone.
--
-- Existing rows have no way to know which client they belonged to (the old
-- hash didn't include client_id), so this truncates the cache rather than
-- guessing — same call made in 005_local_embeddings.sql when the embedding
-- dimension changed. Cached *responses* live nowhere else as the source of
-- truth; this table is purely a performance cache, safe to empty. It will
-- simply repopulate (now correctly scoped) as traffic comes in.
-- =============================================================================

TRUNCATE TABLE semantic_cache;

ALTER TABLE semantic_cache
    ADD COLUMN IF NOT EXISTS client_id VARCHAR(64) NOT NULL DEFAULT '__unscoped__';

-- Drop the default now that existing (truncated) rows don't need a
-- placeholder — every future INSERT must supply a real client_id.
ALTER TABLE semantic_cache
    ALTER COLUMN client_id DROP DEFAULT;

-- The old constraint made prompt_hash globally unique, which is what
-- allowed cross-tenant collisions in the first place. Replace it with a
-- composite constraint scoped per client.
ALTER TABLE semantic_cache
    DROP CONSTRAINT IF EXISTS semantic_cache_prompt_hash_key;

ALTER TABLE semantic_cache
    ADD CONSTRAINT semantic_cache_prompt_hash_client_id_key UNIQUE (prompt_hash, client_id);

CREATE INDEX IF NOT EXISTS idx_semantic_cache_client_id
    ON semantic_cache (client_id);

COMMENT ON COLUMN semantic_cache.client_id IS
    'SHA-256(raw_api_key) of the client this cache entry belongs to — same convention as client_tiers.client_id. Cache entries are never shared across clients.';
