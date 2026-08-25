// =============================================================================
// src/client_registry.rs  — RouterFuel v0.9
//
// Per-client provisioning: which API keys exist, and what tier each one is
// on. Sourced from either:
//
//   A) Environment variable ROUTERFUEL_CLIENT_TIERS (fast, no DB needed)
//      Format:  "raw_key_1:pro,raw_key_2:enterprise,raw_key_3:free"
//      Keys are hashed to match ApiKeyStore's client_id convention.
//      Applied once at startup — it's static config.
//
//   B) Postgres table `client_tiers` — the live source of truth, re-read on a
//      timer by `spawn_client_sync_task` (see migrations/003_client_tiers.sql
//      for the table and 008_dynamic_api_keys.sql for the client_name/active
//      columns that let it drive authentication too).
//
// FIXED (was: "tier changes made only in the DB take effect on the next
// restart"): `spawn_client_sync_task` re-reads the table every
// ROUTERFUEL_CLIENT_SYNC_SECS (default 30) and pushes the result into BOTH
// the ApiKeyStore's DB layer and the RateLimiter. So an INSERT provisions a
// new key and an UPDATE re-tiers an existing one, both without a restart.
//
// One sync feeds both consumers deliberately: auth and tiering read the same
// row, so splitting them into two queries would let a key authenticate at a
// tier the same row disagrees with.
// =============================================================================

use crate::auth::{ApiKeyStore, DbKeyRecord};
use crate::rate_limiter::{RateLimiter, TierConfig};
use sha2::{Digest, Sha256};
use sqlx::postgres::PgPool;
use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;
use tracing::{error, info, warn};

/// How often to re-read `client_tiers` when the interval isn't configured.
/// Short enough that a signup flow feels instant, long enough that the query
/// is negligible next to normal request traffic.
pub const DEFAULT_SYNC_INTERVAL_SECS: u64 = 30;

// =============================================================================
// Tier parsing
// =============================================================================

pub fn parse_tier(s: &str) -> TierConfig {
    match s.trim().to_lowercase().as_str() {
        "free"       => TierConfig::FREE,
        "pro"        => TierConfig::PRO,
        "enterprise" => TierConfig::ENTERPRISE,
        other => {
            warn!("Unknown tier '{}' — defaulting to Pro", other);
            TierConfig::PRO
        }
    }
}

// =============================================================================
// Load from environment variable
//
// ROUTERFUEL_CLIENT_TIERS format:
//   "raw_key_1:pro,raw_key_2:enterprise,raw_key_3:free"
//
// Keys are stored as SHA-256 hashes in ApiKeyStore (see auth.rs) — here we
// accept the raw key so the same secret works for both auth and tier
// assignment, and hash it the same way ApiKeyStore does.
// =============================================================================

pub fn load_tiers_from_env(raw: &str, rate_limiter: &Arc<RateLimiter>) -> usize {
    let mut count = 0;

    for entry in raw.split(',') {
        let entry = entry.trim();
        if entry.is_empty() { continue; }

        match entry.split_once(':') {
            Some((key, tier_str)) => {
                let client_id = sha256_hex(key.trim());
                let tier = parse_tier(tier_str);

                rate_limiter.register(&client_id, tier);

                info!(
                    client_id = &client_id[..8],
                    tier = tier_str.trim(),
                    "Registered client tier from env"
                );
                count += 1;
            }
            None => {
                error!(
                    "Bad ROUTERFUEL_CLIENT_TIERS entry '{}' — format: raw_key:tier",
                    entry
                );
            }
        }
    }

    count
}

// =============================================================================
// Sync from Postgres — one query, two consumers (ApiKeyStore + RateLimiter).
//
// See migrations/003_client_tiers.sql (table) and 008_dynamic_api_keys.sql
// (client_name + active columns).
// =============================================================================

#[derive(Debug, Default, Clone, Copy)]
pub struct SyncStats {
    pub active: usize,
    pub inactive: usize,
    pub tier_changes: usize,
}

pub async fn sync_clients_from_db(
    pool: &PgPool,
    api_key_store: &Arc<ApiKeyStore>,
    rate_limiter: &Arc<RateLimiter>,
) -> Result<SyncStats, sqlx::Error> {
    let table_exists: bool = sqlx::query_scalar(
        "SELECT EXISTS (
            SELECT FROM information_schema.tables
            WHERE table_name = 'client_tiers'
         )"
    )
    .fetch_one(pool)
    .await?;

    if !table_exists {
        info!("client_tiers table not found — skipping DB client sync");
        return Ok(SyncStats::default());
    }

    // Inactive rows are fetched too, not filtered out in SQL: ApiKeyStore
    // needs them to shadow the env layer so `active = false` is a real
    // revocation even for a key also listed in ROUTERFUEL_API_KEYS.
    let rows = sqlx::query_as::<_, (String, String, String, bool)>(
        "SELECT client_id, client_name, tier, active FROM client_tiers"
    )
    .fetch_all(pool)
    .await?;

    let mut stats = SyncStats::default();
    let mut snapshot: HashMap<String, DbKeyRecord> = HashMap::with_capacity(rows.len());

    for (client_id, client_name, tier_str, active) in rows {
        if active {
            let tier = parse_tier(&tier_str);

            // Only re-register when the tier actually changed. `register()`
            // replaces the client's governor limiter, which resets its token
            // bucket — calling it unconditionally every sync interval would
            // hand every client a fresh full burst every 30s and effectively
            // void the rate limit.
            let current = rate_limiter.status(&client_id).map(|s| s.tier_name);
            if current != Some(tier.name) {
                rate_limiter.register(&client_id, tier);
                stats.tier_changes += 1;
                info!(
                    client_id = &client_id[..client_id.len().min(8)],
                    client = %client_name,
                    from = current.unwrap_or("<unregistered>"),
                    to = tier.name,
                    "Client tier applied from DB"
                );
            }
            stats.active += 1;
        } else {
            stats.inactive += 1;
        }

        // The tier isn't kept in the snapshot — it was applied to the
        // RateLimiter above, which is what the request path reads.
        snapshot.insert(client_id, DbKeyRecord { client_name, active });
    }

    // Swapped in only on success — a failed query above returns early and
    // leaves the previous snapshot intact rather than revoking everyone.
    api_key_store.replace_db_keys(snapshot);

    Ok(stats)
}

// =============================================================================
// Background sync task — this is what removes the restart requirement.
// =============================================================================

pub fn spawn_client_sync_task(
    pool: PgPool,
    api_key_store: Arc<ApiKeyStore>,
    rate_limiter: Arc<RateLimiter>,
    interval: Duration,
) {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(interval);
        // The first tick fires immediately; the caller has already run one
        // sync synchronously at startup, so skip straight to the next.
        ticker.tick().await;

        loop {
            ticker.tick().await;

            match sync_clients_from_db(&pool, &api_key_store, &rate_limiter).await {
                Ok(stats) => {
                    if stats.tier_changes > 0 {
                        info!(
                            active = stats.active,
                            inactive = stats.inactive,
                            tier_changes = stats.tier_changes,
                            "client_tiers sync applied changes"
                        );
                    } else {
                        tracing::debug!(
                            active = stats.active,
                            inactive = stats.inactive,
                            "client_tiers sync — no changes"
                        );
                    }
                }
                Err(e) => {
                    // Keep the last good snapshot and try again next tick;
                    // a DB blip must not lock every client out.
                    warn!(
                        "client_tiers sync failed ({}) — keeping the previous snapshot, \
                         retrying in {}s",
                        e,
                        interval.as_secs()
                    );
                }
            }
        }
    });
}

// =============================================================================
// Combined startup loader — env first, then DB (DB is applied after, so if the
// same client_id appears in both, the DB value is what's active — intentional,
// since the DB is the one you can change without a redeploy).
//
// Runs one sync synchronously so the server never starts serving with an
// empty auth store, then hands off to `spawn_client_sync_task` for the
// recurring refresh.
// =============================================================================

pub async fn load_all_tiers(
    pool: &PgPool,
    api_key_store: &Arc<ApiKeyStore>,
    rate_limiter: &Arc<RateLimiter>,
    env_tiers_raw: &str,
    default_tier: TierConfig,
) {
    rate_limiter.set_default_tier(default_tier);

    let mut total = 0;

    if !env_tiers_raw.is_empty() {
        let n = load_tiers_from_env(env_tiers_raw, rate_limiter);
        total += n;
        info!("Loaded {} client tiers from ROUTERFUEL_CLIENT_TIERS", n);
    }

    match sync_clients_from_db(pool, api_key_store, rate_limiter).await {
        Ok(stats) => {
            total += stats.active;
            info!(
                active = stats.active,
                inactive = stats.inactive,
                "Loaded client keys/tiers from the client_tiers table"
            );
        }
        Err(e) => {
            warn!(
                "Could not load clients from DB ({}) — running on ROUTERFUEL_API_KEYS / \
                 ROUTERFUEL_CLIENT_TIERS only until the next sync succeeds",
                e
            );
        }
    }

    if total == 0 {
        warn!(
            "No client tiers configured — every client will get the '{}' tier ({} req/s) \
             until explicitly registered. Set ROUTERFUEL_CLIENT_TIERS or add rows to \
             client_tiers.",
            default_tier.name, default_tier.capacity
        );
    }

    info!("Client tier registry ready ({} entries, default = {})", total, default_tier.name);
}

// =============================================================================
// Helper
// =============================================================================

fn sha256_hex(input: &str) -> String {
    let mut h = Sha256::new();
    h.update(input.as_bytes());
    format!("{:x}", h.finalize())
}

// =============================================================================
// Tests
// =============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_known_tiers() {
        assert_eq!(parse_tier("free").capacity, TierConfig::FREE.capacity);
        assert_eq!(parse_tier("pro").capacity, TierConfig::PRO.capacity);
        assert_eq!(parse_tier("enterprise").capacity, TierConfig::ENTERPRISE.capacity);
    }

    #[test]
    fn parse_unknown_defaults_to_pro() {
        let t = parse_tier("gold");
        assert_eq!(t.capacity, TierConfig::PRO.capacity);
    }

    #[test]
    fn load_from_env_registers_clients() {
        let rl = Arc::new(RateLimiter::new());
        let raw = "rf_live_key1:pro,rf_live_key2:free,rf_live_key3:enterprise";
        let n = load_tiers_from_env(raw, &rl);
        assert_eq!(n, 3);

        let id1 = sha256_hex("rf_live_key1");
        assert!(rl.status(&id1).is_some());
        assert_eq!(rl.status(&id1).unwrap().tier_name, "pro");
    }

    #[test]
    fn empty_env_string_registers_nothing() {
        let rl = Arc::new(RateLimiter::new());
        let n = load_tiers_from_env("", &rl);
        assert_eq!(n, 0);
    }
}
