# RouterFuel

[![License: AGPL v3](https://img.shields.io/badge/License-AGPLv3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0)

A BYOK (Bring Your Own Key) AI gateway written in Rust. RouterFuel sits between your app and the LLM providers you already have keys for — Anthropic, OpenAI, Gemini, DeepSeek, xAI, Mistral, Qwen, Moonshot, Zhipu, Meta, Azure OpenAI, AWS Bedrock, and OpenRouter as a universal fallback — and adds the routing, cost tracking, caching, and safety nets you'd otherwise have to build yourself.

RouterFuel never holds a billable key of its own. Every request is billed to *your* provider account, using *your* key. RouterFuel's job is just to route it well, cache it when it can, and tell you what it cost.

## Features

- **Smart routing** — pick a model by name, let RouterFuel auto-select on cost/latency/quality, or route by task type (`task:summarize`, `task:extract_action_items`, `task:draft_response`, `task:answer_question`, `task:classify`)
- **BYOK across 13 providers** — supply your own key per provider via request headers; OpenRouter acts as a universal fallback if that's the only key you have
- **Azure OpenAI** — bring your own Azure OpenAI deployment; supply an endpoint + API key (or managed identity) via the `X-Azure-OpenAI-Connection` header. Models are fetched dynamically from your Azure Foundry deployments list at startup
- **AWS Bedrock** — bring your own AWS Bedrock access; supply region + IAM credentials via the `X-Bedrock-Connection` header. Available foundation models are fetched dynamically from the Bedrock `ListFoundationModels` API at startup
- **Vision support** — send images (URL or base64) to any vision-capable model in the registry
- **Semantic caching** — a local ONNX embedding model (no external API cost) matches semantically similar prompts and serves cached responses instead of re-calling a provider. Cached entries are scoped per client — two different clients sending the same prompt never share a cache entry
- **Cost tracking & audit trail** — every request is logged with token counts, cost, latency, and savings vs. a GPT-4o baseline
- **Circuit breaker** — automatically stops sending traffic to a provider that's returning errors, and probes it back into rotation once it recovers
- **Rate limiting & tiers** — per-client rate limits (free / pro / enterprise), configurable via env var or a Postgres table; tier changes take effect on the **next server restart** (tiers are loaded once at startup, not watched live)
- **Concurrency limiting** — bounds in-flight provider calls so a traffic spike doesn't get you rate-limited or IP-blocked upstream
- **Guardrails** — LoopGuard flags a client stuck retrying the same prompt; SpendGuard hard-caps per-client spend in a rolling window
- **Shadow-mode A/B testing** — fire a second, comparison-only call at a different model alongside the real one, without affecting what the client receives. **Enabled by default** — any client can trigger it by sending `shadow_model` on a request, and it bills a second real call to their BYOK key
- **Streaming** — full SSE streaming support for Anthropic, Gemini, Azure OpenAI, Bedrock, and every OpenAI-compatible provider
- **Admin dashboard** — a self-hosted, no-build-step web UI at `/admin/dashboard` visualizing spend, cache performance, per-model and per-client cost, the request timeline, rate-limit tiers, and shadow-mode comparisons — reads the `/admin/*` endpoints below in real time. The dashboard *page* itself is public; the data endpoints it calls each require `X-Admin-Key`
- **Cursor integration** — point Cursor's custom OpenAI-compatible model settings straight at RouterFuel and route your editor's requests through your own provider keys

## Requirements

- Rust (2021 edition)
- PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension installed
- A local ONNX sentence-embedding model + tokenizer (e.g. [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)) for semantic caching — download the `.onnx` and `tokenizer.json` files and point RouterFuel at them (see below)

## Quickstart (Docker)

```bash
git clone https://github.com/uaz5/Routerfuel.git
cd Routerfuel
cp env.example .env         # then fill in ROUTERFUEL_ADMIN_KEY at minimum
./scripts/generate-key.sh "MyFirstClient"   # copy the hash line into .env's ROUTERFUEL_API_KEYS
docker compose up
```

This starts Postgres with `pgvector` pre-installed and runs migrations automatically on first boot — no manual database setup. RouterFuel listens on `http://localhost:3000`.

No `--build` needed: the `app` service pulls a prebuilt image (`nayilumair/routerfuel:0.6.2`), so first run takes seconds instead of the 10-15 minutes a from-scratch Rust compile costs.

Semantic caching (local ONNX embeddings) is on out of the box — the model and tokenizer are committed to this repo under `./models/`, and compose mounts them into the container, so there's nothing to download or convert. If those files are missing or unreadable the gateway still runs normally with semantic caching disabled; look for `Local ONNX embedding model loaded — semantic cache active` in the startup log to confirm it's active. Note the models are mounted by compose rather than baked into the image, so running the image on its own leaves caching off.

### Building from source (contributors)

If you're changing the Rust code, layer on the build override to compile locally instead of pulling:

```bash
docker compose -f docker-compose.yml -f docker-compose.build.yml up --build
```

That builds the `app` service from the local `Dockerfile` and tags it `routerfuel-local:dev`, leaving the published image alone. Everything else — Postgres, env vars, volumes, ports — is inherited from the base compose file.

**Heads up if you're building this yourself:** the container build needs network access (the `ort` crate downloads ONNX Runtime binaries during compilation), and the ONNX shared library path is the one part of this setup that's genuinely a little fragile across environments — see the comments at the top of the `Dockerfile` if `docker compose up` starts fine but logs a warning that the embedding model didn't load. The gateway itself runs correctly either way; only semantic caching is affected.

## Manual Setup (no Docker)

**1. Clone and build**

```
git clone https://github.com/uaz5/Routerfuel.git
cd Routerfuel
cargo build --release
```

**2. Set up the database**

Create a Postgres database with the `vector` extension available, then run the migrations in `migrations/` in order (001 through 007). If you're using `sqlx-cli`:

```
sqlx migrate run
```

Migrations run automatically on startup too, via `sqlx::migrate!` in `main.rs`.

**3. Set environment variables**

| Variable                        | Required | Default       | Purpose                                                             |
| -------------------------------- | -------- | ------------- | -------------------------------------------------------------------- |
| `DATABASE_URL`                  | yes      | —             | Postgres connection string                                          |
| `ROUTERFUEL_API_KEYS`           | no       | empty         | Fallback/override client API keys, format `sha256hex:ClientName,...`. The `client_tiers` Postgres table is the primary source of keys and tiers |
| `ROUTERFUEL_CLIENT_TIERS`       | no       | empty         | Fallback per-client tiers, format `raw_key:pro,raw_key:enterprise`. Applied once at startup; `client_tiers` rows override |
| `ROUTERFUEL_CLIENT_SYNC_SECS`   | no       | 30            | How often to re-read the `client_tiers` table for new keys and tier changes |
| `ROUTERFUEL_ADMIN_KEY`          | no       | empty         | Key required to access `/admin/*` endpoints (`X-Admin-Key` header)  |
| `ROUTERFUEL_SUPERCOMPRESS_MODE` | no       | `audit`       | Prompt compression before the request is sent: `audit` (measure and log only, the default), `on` (apply), `off`. Tier 1 is lossless — whitespace normalization outside code fences plus exact-duplicate message dedup — with no LLM call and no extra vendor. Per-request override via the `supercompress` object |
| `EMBEDDING_MODEL_PATH`          | no       | `./models/embedding.onnx` | ONNX embedding model path (ships with the repo; enables semantic cache) |
| `EMBEDDING_TOKENIZER_PATH`      | no       | `./models/tokenizer.json` | Matching tokenizer path (ships with the repo)              |
| `LOOP_GUARD_REPEAT_THRESHOLD`   | no       | 4             | Repeats of an identical prompt before it's flagged as a loop        |
| `LOOP_GUARD_WINDOW_SECS`        | no       | 60            | Window LoopGuard checks over                                        |
| `MAX_SPEND_CENTS_PER_CLIENT`    | no       | 5000          | Per-client spend cap (cents) per window                             |
| `SPEND_GUARD_WINDOW_SECS`       | no       | 3600          | SpendGuard rolling window, in seconds                               |
| `MAX_CONCURRENT_PROVIDER_CALLS` | no       | 200           | Caps simultaneous in-flight provider calls                          |
| `ENABLE_SHADOW_MODE`            | no       | **true**      | Enables shadow-mode A/B comparison calls — on by default; set to `false` to disable |
| `TELEMETRY_OUTPUT_DIR`          | no       | `./telemetry` | Where telemetry JSONL files are written                             |
| `TELEMETRY_BUFFER_SIZE`         | no       | 500           | Records buffered before a telemetry flush                           |
| `HOST`                          | no       | `0.0.0.0`     | Bind address                                                        |
| `PORT`                          | no       | `3000`        | Bind port                                                           |

To generate an API key hash for `ROUTERFUEL_API_KEYS`:

```
echo -n "rf_live_yoursecretkey" | sha256sum | awk '{print $1}'
```

**4. Run it**

```
cargo run --release
```

RouterFuel is now listening on `http://localhost:3000` (or whatever `HOST`/`PORT` you set).

See [USAGE.md](https://github.com/uaz5/Routerfuel/blob/main/USAGE.md) for how to actually call it, including the admin dashboard UI and Cursor setup.

## BYOK Provider Headers

RouterFuel is pure BYOK — you supply your own keys per provider via request headers. Here are the headers for each supported provider:

| Provider       | Header                          | Value Format                                                                 |
| -------------- | ------------------------------- | ---------------------------------------------------------------------------- |
| OpenAI         | `X-OpenAI-Api-Key`              | `sk-proj-...` (standard OpenAI API key)                                      |
| Anthropic      | `X-Anthropic-Api-Key`           | `sk-ant-...` (standard Anthropic API key)                                    |
| Gemini         | `X-Gemini-Api-Key`              | Your Google AI Studio API key                                                |
| DeepSeek       | `X-DeepSeek-Api-Key`            | Your DeepSeek API key                                                        |
| Mistral        | `X-Mistral-Api-Key`             | Your Mistral API key                                                         |
| xAI (Grok)     | `X-XAI-Api-Key`                 | Your xAI API key                                                             |
| Qwen           | `X-Qwen-Api-Key`                | Your Alibaba DashScope API key                                               |
| Moonshot (Kimi)| `X-Moonshot-Api-Key`            | Your Moonshot API key                                                        |
| Zhipu (GLM)    | `X-Zhipu-Api-Key`               | Your Zhipu API key                                                           |
| Meta (Llama)   | `X-Meta-Api-Key`                | Your Meta Llama API key                                                      |
| OpenRouter     | `X-OpenRouter-Api-Key`          | `sk-or-...` (standard OpenRouter API key) — acts as universal fallback       |
| Azure OpenAI   | `X-Azure-OpenAI-Connection`     | `endpoint=https://my-resource.openai.azure.com;key=abc123` or `endpoint=...;identity=managed` |
| AWS Bedrock    | `X-Bedrock-Connection`          | `region=us-east-1;access_key=AKIA...;secret_key=...`                         |

**OpenRouter fallback:** If you only supply an `X-OpenRouter-Api-Key` (no direct provider keys), RouterFuel routes *any* model through OpenRouter automatically — you don't need a separate key for each provider.

**Azure OpenAI:** Supply your Azure OpenAI endpoint and either an API key or `identity=managed` for managed identity auth. RouterFuel fetches your available deployments from the Azure Foundry deployments list endpoint at startup, so models appear automatically in the registry. That list isn't a restriction, though: whenever the connection header is present, *any* model name is accepted and routed straight to your Azure deployment — no name prefix and no pre-registration required, since the header itself is proof you can pay for the call.

**AWS Bedrock:** Supply your AWS region and IAM credentials (access key + secret key). RouterFuel fetches available foundation models from the Bedrock `ListFoundationModels` API at startup. In production, proper AWS SigV4 signing is used; for testing, credentials can be passed as headers. As with Azure, that list isn't a restriction — with the connection header present, *any* model name is accepted and routed to Bedrock, with no prefix or pre-registration needed.

## Project structure

```
src/
  main.rs                 — HTTP server, routing glue, request handlers
  connectors.rs            — per-provider HTTP clients (Anthropic, Gemini, Azure OpenAI, Bedrock, OpenAI-compatible)
  route_engine.rs           — model registry + routing decisions
  auth.rs                   — API key validation, BYOK header extraction, Cursor composite-key bridge
  rate_limiter.rs           — per-client tiered rate limiting
  client_registry.rs        — loads client tiers from env/Postgres
  circuit_breaker.rs        — per-provider health tracking
  concurrency.rs            — bounds in-flight provider calls
  guardrails.rs             — LoopGuard + SpendGuard
  semantic_cache.rs         — pgvector-backed semantic cache, scoped per client
  embedder.rs               — local ONNX embedding model
  vision.rs                 — multimodal message types + per-provider image formatting
  tokens.rs                 — tiktoken-based token counting
  cost_tracker.rs           — request logging + cost/savings reports
  telemetry.rs              — JSONL telemetry + ROI reports
  streaming.rs              — SSE streaming handler
  admin.rs                  — /admin/* dashboard data endpoints, incl. /audit/daily
  openrouter_catalog.rs     — pulls OpenRouter's public model list into the registry
  bedrock_catalog.rs        — pulls AWS Bedrock's foundation model list into the registry
static/
  dashboard.html            — self-contained admin dashboard UI, served at /admin/dashboard
migrations/                — Postgres schema, run in numeric order (001–007)
scripts/
  generate-key.sh           — generates a client API key + its SHA-256 hash
Dockerfile                  — multi-stage build (see Quickstart above)
docker-compose.yml          — RouterFuel + Postgres/pgvector wired together
docker-compose.build.yml    — override to build from source instead of pulling
env.example                 — copy to .env before `docker compose up`
```

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0) - see the [LICENSE](https://github.com/uaz5/Routerfuel/blob/main/LICENSE) file for details.
