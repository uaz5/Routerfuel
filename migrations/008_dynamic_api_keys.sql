-- =============================================================================
-- migrations/008_dynamic_api_keys.sql  — RouterFuel v0.9
--
-- Makes `client_tiers` the live source of truth for BOTH authentication and
-- tier assignment, so an external provisioning app (e.g. a Clerk-based
-- Next.js frontend on the same Postgres instance) can create and re-tier API
-- keys by writing rows — no env var edit, no redeploy, no restart.
--
-- Extends the table created in 003_client_tiers.sql rather than replacing it:
-- existing rows, the primary key, and the client_tiers_ts trigger that
-- maintains updated_at all survive untouched.
--
-- What 003 already gave us:  client_id (PK), tier, notes, created_at, updated_at
-- What this adds:            client_name, active
--
-- client_id is the SHA-256 hex digest of the raw API key — the same value
-- auth.rs hashes an incoming X-API-Key down to, and the same convention used
-- by request_logs.client_id and semantic_cache.client_id. It is NOT a Clerk
-- user id; store that in `notes` (or add your own column) if you need to map
-- back to your user table.
-- =============================================================================

-- ── client_name ─────────────────────────────────────────────────────────────
-- Previously the client's display name lived ONLY in the ROUTERFUEL_API_KEYS
-- env var, which is precisely why auth couldn't be driven from the DB. With
-- the name here, a row is fully self-sufficient for authentication.
--
-- DEFAULT 'unnamed' backfills the rows 003 may already have; the default is
-- kept (not dropped) so a provisioning app can insert client_id + tier alone
-- and still produce a valid, usable key.
ALTER TABLE client_tiers
    ADD COLUMN IF NOT EXISTS client_name VARCHAR(255) NOT NULL DEFAULT 'unnamed';

-- ── active ──────────────────────────────────────────────────────────────────
-- The revocation switch. Flipping this to FALSE stops the key authenticating
-- within one sync interval (ROUTERFUEL_CLIENT_SYNC_SECS, default 30s) —
-- preferred over DELETE so the row stays available for audit and so
-- request_logs.client_id keeps resolving to a name.
ALTER TABLE client_tiers
    ADD COLUMN IF NOT EXISTS active BOOLEAN NOT NULL DEFAULT TRUE;

-- The background sync reads only live keys, so index for exactly that.
CREATE INDEX IF NOT EXISTS idx_client_tiers_active
    ON client_tiers (client_id)
    WHERE active;

-- ── Documentation for the external provisioning app ─────────────────────────
COMMENT ON TABLE client_tiers IS
    'Live source of truth for API key auth AND rate-limit tiers. Read by '
    'RouterFuel every ROUTERFUEL_CLIENT_SYNC_SECS (default 30s) — INSERT/UPDATE '
    'here takes effect without a restart. Safe to write from an external app on '
    'the same Postgres instance.';

COMMENT ON COLUMN client_tiers.client_id IS
    'SHA-256 hex digest (64 lowercase chars) of the raw API key. This is the '
    '"key hash" — the raw key is NEVER stored here. Generate with: '
    'echo -n "rf_live_yourkey" | sha256sum. Same convention as '
    'request_logs.client_id and semantic_cache.client_id.';

COMMENT ON COLUMN client_tiers.client_name IS
    'Human-readable client/org display name. Used in logs and the admin '
    'dashboard only — never for authorization decisions.';

COMMENT ON COLUMN client_tiers.tier IS
    'free=10 rps | pro=100 rps | enterprise=1000 rps (see rate_limiter.rs '
    'TierConfig). UPDATE takes effect within one sync interval.';

COMMENT ON COLUMN client_tiers.active IS
    'FALSE revokes the key: authentication starts failing with 401 within one '
    'sync interval. Prefer this over DELETE so audit history survives.';

COMMENT ON COLUMN client_tiers.notes IS
    'Free-form operator notes. Handy for storing the external user id (e.g. a '
    'Clerk user_id) that this key was provisioned for.';

-- ── How the provisioning app uses this ──────────────────────────────────────
--
-- Provision a new free-tier key (hash the raw key in your app; never send the
-- raw key to Postgres):
--
--   INSERT INTO client_tiers (client_id, client_name, tier, active, notes)
--   VALUES ($1, $2, 'free', TRUE, $3)
--   ON CONFLICT (client_id) DO NOTHING;
--
-- Upgrade a key's tier (updated_at is maintained by the 003 trigger):
--
--   UPDATE client_tiers SET tier = 'pro' WHERE client_id = $1;
--
-- Revoke a key:
--
--   UPDATE client_tiers SET active = FALSE WHERE client_id = $1;
