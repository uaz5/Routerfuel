// =============================================================================
// src/admin.rs  — RouterFuel v0.7 — dashboard backend
//
// Admin Dashboard API
//
// Endpoints (all require X-Admin-Key — see admin_key_middleware below):
//   GET /admin/overview          — total spend, savings, request count
//   GET /admin/cache             — cache hit rate, top cached prompts
//   GET /admin/models/expensive  — top 5 most expensive models by total cost
//   GET /admin/models/usage      — all models ranked by request count
//   GET /admin/clients           — per-client spend breakdown
//   GET /admin/timeline          — hourly request/cost timeline (last 24h)
//   GET /admin/rate-limits       — every registered client's tier + capacity
//   GET /admin/shadow            — shadow-mode A/B comparison stats
//   GET /audit/daily             — daily cost/savings report (moved here —
//                                  see note below)
//
// This file previously documented "all require X-API-Key with admin scope"
// but never actually implemented that check — every query below was wide
// open to anyone who found the route. `admin_key_middleware` closes that.
//
// FIX (this revision): `/audit/daily` used to live in main.rs's
// `public_routes`, next to `/health` and `/v1/models`, with NO auth at all —
// even though `get_daily_report` returns the same class of aggregate
// spend/token data as `/admin/overview`, which has always required
// X-Admin-Key. That was a real gap, not a docs mismatch: anyone who found
// the route could pull cross-client spend data with zero credentials.
// Moved here so it's covered by the same admin_key_middleware as everything
// else in this file. `AdminState` now also carries `cost_tracker` so this
// handler has what it needs.
//
// Column note: earlier revisions of this file queried `model_api_id`, but
// migrations/001_request_logs.sql declares the column `model_name` (that's
// what cost_tracker.rs has always written to) — fixed throughout.
// =============================================================================
 
use subtle::ConstantTimeEq;

use axum::{
    extract::{Query, Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Json, Response},
};
use serde::{Deserialize, Serialize};
use sqlx::postgres::PgPool;
use sqlx::Row;
use std::sync::Arc;
use tracing::{error, instrument, warn};
 
use crate::cost_tracker::CostTracker;
use crate::rate_limiter::RateLimiter;
 
// =============================================================================
// Shared state for admin handlers
// =============================================================================
 
#[derive(Clone)]
pub struct AdminState {
    pub pool: Arc<PgPool>,
    pub rate_limiter: Arc<RateLimiter>,
    /// Added so `/audit/daily` can be served from here — see the FIX note
    /// at the top of this file for why it moved out of main.rs's public
    /// routes.
    pub cost_tracker: Arc<CostTracker>,
}
 
// =============================================================================
// Admin auth — a single shared secret (ROUTERFUEL_ADMIN_KEY), not the
// per-client BYOK/auth keys in auth.rs. Deliberately separate: a client key
// should never be able to see every other client's spend.
// =============================================================================
 
pub async fn admin_key_middleware(
    State(admin_key): State<Arc<String>>,
    request: Request,
    next: Next,
) -> Response {
    if admin_key.is_empty() {
        warn!("ROUTERFUEL_ADMIN_KEY is not set — admin routes are disabled");
        return (
            StatusCode::SERVICE_UNAVAILABLE,
            Json(serde_json::json!({
                "error": "Admin dashboard is disabled: set ROUTERFUEL_ADMIN_KEY to enable it."
            })),
        )
            .into_response();
    }
 
    let supplied = request
        .headers()
        .get("x-admin-key")
        .and_then(|v| v.to_str().ok());
 
    // FIX: was a plain `==` on the raw secret, which short-circuits on the
    // first mismatching byte — a timing side channel on ROUTERFUEL_ADMIN_KEY.
    // Compare in constant time instead, same spirit as auth.rs hashing
    // client keys before lookup.
    let matches = supplied
        .map(|k| {
            k.len() == admin_key.len() && k.as_bytes().ct_eq(admin_key.as_bytes()).into()
        })
        .unwrap_or(false);

    if matches {
        next.run(request).await
    } else {
        (
            StatusCode::UNAUTHORIZED,
            Json(serde_json::json!({ "error": "Missing or invalid X-Admin-Key header" })),
        )
            .into_response()
    }
}
 
// =============================================================================
// Query params
// =============================================================================
 
#[derive(Debug, Deserialize)]
pub struct DateRangeQuery {
    /// ISO date string e.g. "2026-06-01"
    #[serde(default = "thirty_days_ago")]
    pub from: String,
    /// ISO date string e.g. "2026-07-01"
    #[serde(default = "today")]
    pub to: String,
}
 
fn today() -> String {
    chrono::Utc::now().format("%Y-%m-%d").to_string()
}
 
fn thirty_days_ago() -> String {
    (chrono::Utc::now() - chrono::Duration::days(30))
        .format("%Y-%m-%d")
        .to_string()
}
 
// =============================================================================
// Response types
// =============================================================================
 
#[derive(Debug, Serialize)]
pub struct OverviewResponse {
    pub period_from: String,
    pub period_to: String,
    pub total_requests: i64,
    pub successful_requests: i64,
    pub failed_requests: i64,
    pub total_spend_usd: f64,
    pub total_saved_usd: f64,
    pub total_saved_pct: f64,
    /// Provider round-trip including generation. Since migration 010 this
    /// is NULL on streaming rows, so AVG covers NON-STREAMING TRAFFIC ONLY.
    /// `latency_sample_count` says how many rows that actually is.
    pub avg_latency_ms: f64,
    pub avg_routing_ms: f64,
    /// Full handler wall-clock. NULL in rows predating migration 009, and
    /// AVG skips those — so on a range spanning the deploy this reflects
    /// only the instrumented rows.
    pub avg_total_latency_ms: f64,
    /// RouterFuel's own overhead: avg(total_latency_ms - latency_ms).
    /// NULL-and-therefore-skipped for streaming rows since migration 010,
    /// because overhead is not computable without a comparable provider
    /// figure — so like `avg_latency_ms` this describes non-streaming
    /// traffic only.
    pub avg_overhead_ms: f64,
    /// Time-to-first-byte, STREAMING TRAFFIC ONLY (NULL elsewhere). Added
    /// with migration 010.
    ///
    /// Deliberately paired with its own count: this and `avg_latency_ms` are
    /// averages over DISJOINT row sets, not two measurements of the same
    /// population. Rendering them side by side without the counts invites
    /// reading one as faster than the other, when a mostly-streaming period
    /// can show a confident-looking provider latency drawn from a handful of
    /// rows.
    pub avg_ttfb_ms: f64,
    /// Rows contributing to `avg_latency_ms` / `avg_overhead_ms`.
    pub latency_sample_count: i64,
    /// Rows contributing to `avg_ttfb_ms`.
    pub ttfb_sample_count: i64,
    pub cache_hits: i64,
    pub cache_hit_rate_pct: f64,
    pub byok_requests: i64,
}
 
#[derive(Debug, Serialize)]
pub struct CacheStatsResponse {
    pub total_entries: i64,
    pub total_hits: i64,
    pub hit_rate_pct: f64,
    pub estimated_saved_usd: f64,
    pub top_cached: Vec<TopCachedEntry>,
}
 
#[derive(Debug, Serialize)]
pub struct TopCachedEntry {
    pub prompt_preview: String,
    pub model_used: String,
    pub hit_count: i64,
}
 
#[derive(Debug, Serialize)]
pub struct ModelCostEntry {
    pub rank: i32,
    pub model_name: String,
    pub provider: String,
    pub total_requests: i64,
    pub total_spend_usd: f64,
    pub avg_cost_usd: f64,
    pub total_tokens_in: i64,
    pub total_tokens_out: i64,
}
 
#[derive(Debug, Serialize)]
pub struct ClientSpendEntry {
    pub client_id: String,
    pub total_requests: i64,
    pub total_spend_usd: f64,
    pub total_saved_usd: f64,
    pub avg_latency_ms: f64,
}
 
#[derive(Debug, Serialize)]
pub struct TimelineEntry {
    pub hour: String,
    pub request_count: i64,
    pub spend_usd: f64,
    pub saved_usd: f64,
    pub avg_latency_ms: f64,
    pub cache_hits: i64,
}
 
#[derive(Debug, Serialize)]
pub struct RateLimitEntry {
    pub client_id: String,
    pub tier: &'static str,
    pub capacity_rps: u32,
}
 
#[derive(Debug, Serialize)]
pub struct ShadowPairStats {
    pub primary_model: String,
    pub shadow_model: String,
    pub comparisons: i64,
    pub avg_cost_delta_usd: f64,
    pub avg_latency_delta_ms: f64,
    pub shadow_error_rate_pct: f64,
}
 
// =============================================================================
// GET /admin/overview
// =============================================================================
 
#[instrument(skip(state))]
pub async fn overview_handler(
    State(state): State<AdminState>,
    Query(q): Query<DateRangeQuery>,
) -> impl IntoResponse {
    let row = sqlx::query(
        r#"
        SELECT
            COUNT(*)                                          AS total_requests,
            COUNT(*) FILTER (WHERE status = 'success')       AS successful_requests,
            COUNT(*) FILTER (WHERE status != 'success')       AS failed_requests,
            COALESCE(SUM(cost_cents)  / 100.0, 0)::float8   AS total_spend_usd,
            COALESCE(SUM(cost_saved_cents) / 100.0, 0)::float8 AS total_saved_usd,
            COALESCE(AVG(latency_ms), 0)::float8             AS avg_latency_ms,
            COALESCE(AVG(routing_decision_ms), 0)::float8    AS avg_routing_ms,
            COALESCE(AVG(total_latency_ms), 0)::float8       AS avg_total_latency_ms,
            COALESCE(AVG(total_latency_ms - latency_ms), 0)::float8 AS avg_overhead_ms,
            COALESCE(AVG(ttfb_ms), 0)::float8                AS avg_ttfb_ms,
            -- COUNT(col) counts non-NULLs, which is the point: these report
            -- how many rows each average is actually built from. The two are
            -- disjoint by construction (migration 010), so a caller can tell
            -- a meaningful figure from one row's worth of noise.
            COUNT(latency_ms)                                AS latency_sample_count,
            COUNT(ttfb_ms)                                   AS ttfb_sample_count,
            COUNT(*) FILTER (WHERE from_cache = TRUE)        AS cache_hits,
            COUNT(*) FILTER (WHERE is_byok = TRUE)           AS byok_requests
        FROM request_logs
        WHERE DATE(created_at) BETWEEN $1::DATE AND $2::DATE
        "#,
    )
    .bind(&q.from)
    .bind(&q.to)
    .fetch_one(state.pool.as_ref())
    .await;
 
    match row {
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
        Ok(r) => {
            let total: i64 = r.get("total_requests");
            let hits: i64 = r.get("cache_hits");
            let spend: f64 = r.get("total_spend_usd");
            let saved: f64 = r.get("total_saved_usd");
            let hit_pct: f64 = if total > 0 { hits as f64 / total as f64 * 100.0 } else { 0.0 };
            let saved_pct: f64 = if (spend + saved) > 0.0 {
                saved / (spend + saved) * 100.0
            } else {
                0.0
            };
 
            Json(OverviewResponse {
                period_from: q.from,
                period_to: q.to,
                total_requests: total,
                successful_requests: r.get("successful_requests"),
                failed_requests: r.get("failed_requests"),
                total_spend_usd: spend,
                total_saved_usd: saved,
                total_saved_pct: saved_pct,
                avg_latency_ms: r.get("avg_latency_ms"),
                avg_routing_ms: r.get("avg_routing_ms"),
                avg_total_latency_ms: r.get("avg_total_latency_ms"),
                avg_overhead_ms: r.get("avg_overhead_ms"),
                avg_ttfb_ms: r.get("avg_ttfb_ms"),
                latency_sample_count: r.get("latency_sample_count"),
                ttfb_sample_count: r.get("ttfb_sample_count"),
                cache_hits: hits,
                cache_hit_rate_pct: hit_pct,
                byok_requests: r.get("byok_requests"),
            })
            .into_response()
        }
    }
}
 
// =============================================================================
// GET /admin/cache
// =============================================================================
 
#[instrument(skip(state))]
pub async fn cache_stats_handler(State(state): State<AdminState>) -> impl IntoResponse {
    let summary = sqlx::query(
        r#"
        SELECT
            COUNT(*)           AS total_entries,
            COALESCE(SUM(hit_count), 0) AS total_hits
        FROM semantic_cache
        "#,
    )
    .fetch_one(state.pool.as_ref())
    .await;
 
    let top_rows = sqlx::query(
        r#"
        SELECT prompt_preview, model_used, hit_count
        FROM semantic_cache
        ORDER BY hit_count DESC
        LIMIT 10
        "#,
    )
    .fetch_all(state.pool.as_ref())
    .await
    .unwrap_or_default();
 
    let cache_hit_count: i64 =
        sqlx::query_scalar("SELECT COUNT(*) FROM request_logs WHERE from_cache = TRUE")
            .fetch_one(state.pool.as_ref())
            .await
            .unwrap_or(0);
 
    let total_req: i64 = sqlx::query_scalar("SELECT COUNT(*) FROM request_logs")
        .fetch_one(state.pool.as_ref())
        .await
        .unwrap_or(0);
 
    // Estimated savings: avg cost per non-cached request × cache hits
    let avg_cost: f64 = sqlx::query_scalar(
        r#"
        SELECT COALESCE(AVG(cost_cents) / 100.0, 0)::float8
        FROM request_logs
        WHERE from_cache = FALSE AND status = 'success'
        "#,
    )
    .fetch_one(state.pool.as_ref())
    .await
    .unwrap_or(0.0);
 
    let hit_rate = if total_req > 0 {
        cache_hit_count as f64 / total_req as f64 * 100.0
    } else {
        0.0
    };
 
    let top_cached: Vec<TopCachedEntry> = top_rows
        .iter()
        .map(|r| TopCachedEntry {
            prompt_preview: r.get("prompt_preview"),
            model_used: r.get("model_used"),
            hit_count: r.get::<i64, _>("hit_count"),
        })
        .collect();
 
    let (total_entries, total_hits) = match summary {
        Ok(r) => (r.get::<i64, _>("total_entries"), r.get::<i64, _>("total_hits")),
        Err(_) => (0, 0),
    };
 
    Json(CacheStatsResponse {
        total_entries,
        total_hits,
        hit_rate_pct: hit_rate,
        estimated_saved_usd: avg_cost * cache_hit_count as f64,
        top_cached,
    })
    .into_response()
}
 
// =============================================================================
// GET /admin/models/expensive  — Top 5 most expensive models by total cost
// =============================================================================
 
#[instrument(skip(state))]
pub async fn top_expensive_models_handler(
    State(state): State<AdminState>,
    Query(q): Query<DateRangeQuery>,
) -> impl IntoResponse {
    let rows = sqlx::query(
        r#"
        SELECT
            ROW_NUMBER() OVER (ORDER BY SUM(cost_cents) DESC)::INT AS rank,
            model_name,
            provider,
            COUNT(*)                                     AS total_requests,
            (SUM(cost_cents)       / 100.0)::float8      AS total_spend_usd,
            (AVG(cost_cents)       / 100.0)::float8      AS avg_cost_usd,
            COALESCE(SUM(tokens_in),  0)                 AS total_tokens_in,
            COALESCE(SUM(tokens_out), 0)                 AS total_tokens_out
        FROM request_logs
        WHERE DATE(created_at) BETWEEN $1::DATE AND $2::DATE
          AND status = 'success'
        GROUP BY model_name, provider
        ORDER BY SUM(cost_cents) DESC
        LIMIT 5
        "#,
    )
    .bind(&q.from)
    .bind(&q.to)
    .fetch_all(state.pool.as_ref())
    .await;
 
    match rows {
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
        Ok(rows) => {
            let models: Vec<ModelCostEntry> = rows
                .iter()
                .map(|r| ModelCostEntry {
                    rank: r.get("rank"),
                    model_name: r.get("model_name"),
                    provider: r.get("provider"),
                    total_requests: r.get("total_requests"),
                    total_spend_usd: r.get("total_spend_usd"),
                    avg_cost_usd: r.get("avg_cost_usd"),
                    total_tokens_in: r.get("total_tokens_in"),
                    total_tokens_out: r.get("total_tokens_out"),
                })
                .collect();
            Json(models).into_response()
        }
    }
}
 
// =============================================================================
// GET /admin/models/usage  — All models ranked by request count
// =============================================================================
 
#[instrument(skip(state))]
pub async fn model_usage_handler(
    State(state): State<AdminState>,
    Query(q): Query<DateRangeQuery>,
) -> impl IntoResponse {
    let rows = sqlx::query(
        r#"
        SELECT
            ROW_NUMBER() OVER (ORDER BY COUNT(*) DESC)::INT AS rank,
            model_name,
            provider,
            COUNT(*)                                     AS total_requests,
            (SUM(cost_cents)       / 100.0)::float8      AS total_spend_usd,
            (AVG(cost_cents)       / 100.0)::float8      AS avg_cost_usd,
            COALESCE(SUM(tokens_in),  0)                 AS total_tokens_in,
            COALESCE(SUM(tokens_out), 0)                 AS total_tokens_out
        FROM request_logs
        WHERE DATE(created_at) BETWEEN $1::DATE AND $2::DATE
          AND status = 'success'
        GROUP BY model_name, provider
        ORDER BY COUNT(*) DESC
        "#,
    )
    .bind(&q.from)
    .bind(&q.to)
    .fetch_all(state.pool.as_ref())
    .await;
 
    match rows {
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
        Ok(rows) => {
            let models: Vec<ModelCostEntry> = rows
                .iter()
                .map(|r| ModelCostEntry {
                    rank: r.get("rank"),
                    model_name: r.get("model_name"),
                    provider: r.get("provider"),
                    total_requests: r.get("total_requests"),
                    total_spend_usd: r.get("total_spend_usd"),
                    avg_cost_usd: r.get("avg_cost_usd"),
                    total_tokens_in: r.get("total_tokens_in"),
                    total_tokens_out: r.get("total_tokens_out"),
                })
                .collect();
            Json(models).into_response()
        }
    }
}
 
// =============================================================================
// GET /admin/clients  — Per-client spend breakdown
// =============================================================================
 
#[instrument(skip(state))]
pub async fn client_spend_handler(
    State(state): State<AdminState>,
    Query(q): Query<DateRangeQuery>,
) -> impl IntoResponse {
    let rows = sqlx::query(
        r#"
        SELECT
            COALESCE(client_id, 'anonymous')             AS client_id,
            COUNT(*)                                     AS total_requests,
            (SUM(cost_cents)       / 100.0)::float8      AS total_spend_usd,
            (SUM(cost_saved_cents) / 100.0)::float8      AS total_saved_usd,
            AVG(latency_ms)::float8                      AS avg_latency_ms
        FROM request_logs
        WHERE DATE(created_at) BETWEEN $1::DATE AND $2::DATE
        GROUP BY client_id
        ORDER BY SUM(cost_cents) DESC
        "#,
    )
    .bind(&q.from)
    .bind(&q.to)
    .fetch_all(state.pool.as_ref())
    .await;
 
    match rows {
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
        Ok(rows) => {
            let clients: Vec<ClientSpendEntry> = rows
                .iter()
                .map(|r| ClientSpendEntry {
                    client_id: r.get("client_id"),
                    total_requests: r.get("total_requests"),
                    total_spend_usd: r.get("total_spend_usd"),
                    total_saved_usd: r.get("total_saved_usd"),
                    avg_latency_ms: r.get("avg_latency_ms"),
                })
                .collect();
            Json(clients).into_response()
        }
    }
}
 
// =============================================================================
// GET /admin/timeline  — Hourly breakdown for last 24 hours
// =============================================================================
 
#[instrument(skip(state))]
pub async fn timeline_handler(State(state): State<AdminState>) -> impl IntoResponse {
    let rows = sqlx::query(
        r#"
        SELECT
            TO_CHAR(DATE_TRUNC('hour', created_at), 'YYYY-MM-DD HH24:00') AS hour,
            COUNT(*)                                     AS request_count,
            COALESCE(SUM(cost_cents)       / 100.0, 0)::float8 AS spend_usd,
            COALESCE(SUM(cost_saved_cents) / 100.0, 0)::float8 AS saved_usd,
            COALESCE(AVG(latency_ms),       0)::float8   AS avg_latency_ms,
            COUNT(*) FILTER (WHERE from_cache = TRUE)    AS cache_hits
        FROM request_logs
        WHERE created_at >= NOW() - INTERVAL '24 hours'
        GROUP BY DATE_TRUNC('hour', created_at)
        ORDER BY DATE_TRUNC('hour', created_at) ASC
        "#,
    )
    .fetch_all(state.pool.as_ref())
    .await;
 
    match rows {
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
        Ok(rows) => {
            let timeline: Vec<TimelineEntry> = rows
                .iter()
                .map(|r| TimelineEntry {
                    hour: r.get("hour"),
                    request_count: r.get("request_count"),
                    spend_usd: r.get("spend_usd"),
                    saved_usd: r.get("saved_usd"),
                    avg_latency_ms: r.get("avg_latency_ms"),
                    cache_hits: r.get("cache_hits"),
                })
                .collect();
            Json(timeline).into_response()
        }
    }
}
 
// =============================================================================
// GET /admin/rate-limits  — every registered client's tier + capacity.
// Doesn't exist without a DB query in the original file — the rate limiter
// registry (rate_limiter.rs) only lived in memory with no way to inspect it
// from outside the process. Since it's in-memory, this only lists clients
// that have made at least one request (or were registered at startup) on
// *this* replica.
// =============================================================================
 
#[instrument(skip(state))]
pub async fn rate_limits_handler(State(state): State<AdminState>) -> impl IntoResponse {
    // We don't have a full-scan API on RateLimiter (by design — it's a hot
    // path structure, not meant for iteration) so this reports registered
    // clients we can find via request_logs, cross-referenced with their
    // live tier status.
    let client_ids: Vec<String> = sqlx::query_scalar(
        "SELECT DISTINCT client_id FROM request_logs WHERE client_id IS NOT NULL LIMIT 500",
    )
    .fetch_all(state.pool.as_ref())
    .await
    .unwrap_or_default();
 
    let entries: Vec<RateLimitEntry> = client_ids
        .into_iter()
        .filter_map(|client_id| {
            state.rate_limiter.status(&client_id).map(|status| RateLimitEntry {
                client_id,
                tier: status.tier_name,
                capacity_rps: status.capacity_rps,
            })
        })
        .collect();
 
    Json(entries).into_response()
}
 
// =============================================================================
// GET /admin/shadow  — shadow-mode A/B comparison summary, grouped by
// (primary_model, shadow_model) pair. See migrations/006_shadow_comparisons.sql
// and main.rs::maybe_fire_shadow_request.
// =============================================================================
 
#[instrument(skip(state))]
pub async fn shadow_stats_handler(
    State(state): State<AdminState>,
    Query(q): Query<DateRangeQuery>,
) -> impl IntoResponse {
    let rows = sqlx::query(
        r#"
        SELECT
            primary_model,
            shadow_model,
            COUNT(*)                                                        AS comparisons,
            COALESCE(AVG(cost_delta_cents) / 100.0, 0)::float8             AS avg_cost_delta_usd,
            COALESCE(AVG(latency_delta_ms), 0)::float8                     AS avg_latency_delta_ms,
            (COUNT(*) FILTER (WHERE shadow_error IS NOT NULL))::FLOAT8
                / GREATEST(COUNT(*), 1)::FLOAT8 * 100.0                     AS shadow_error_rate_pct
        FROM shadow_comparisons
        WHERE DATE(created_at) BETWEEN $1::DATE AND $2::DATE
        GROUP BY primary_model, shadow_model
        ORDER BY COUNT(*) DESC
        "#,
    )
    .bind(&q.from)
    .bind(&q.to)
    .fetch_all(state.pool.as_ref())
    .await;
 
    match rows {
        Err(e) => (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(serde_json::json!({ "error": e.to_string() })),
        )
            .into_response(),
        Ok(rows) => {
            let pairs: Vec<ShadowPairStats> = rows
                .iter()
                .map(|r| ShadowPairStats {
                    primary_model: r.get("primary_model"),
                    shadow_model: r.get("shadow_model"),
                    comparisons: r.get("comparisons"),
                    avg_cost_delta_usd: r.get("avg_cost_delta_usd"),
                    avg_latency_delta_ms: r.get("avg_latency_delta_ms"),
                    shadow_error_rate_pct: r.get("shadow_error_rate_pct"),
                })
                .collect();
            Json(pairs).into_response()
        }
    }
}

// =============================================================================
// GET /audit/daily  — moved here from main.rs's public_routes. Was
// previously reachable with NO authentication at all even though it returns
// the same class of aggregate cost/spend data as /admin/overview. Now
// behind the same admin_key_middleware as every other handler in this file.
// =============================================================================

#[instrument(skip(state))]
pub async fn audit_daily_handler(
    State(state): State<AdminState>,
    Query(params): Query<std::collections::HashMap<String, String>>,
) -> Response {
    let date = match params.get("date") {
        Some(d) => d,
        None => {
            return (
                StatusCode::BAD_REQUEST,
                Json(serde_json::json!({ "error": "date parameter required" })),
            )
                .into_response()
        }
    };

    match state.cost_tracker.get_daily_report(date).await {
        Ok(report) => Json(report).into_response(),
        Err(e) => {
            error!("Failed to generate report: {}", e);
            (
                StatusCode::INTERNAL_SERVER_ERROR,
                Json(serde_json::json!({ "error": "Report generation failed" })),
            )
                .into_response()
        }
    }
}
