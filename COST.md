# Cost — 전체 비용

## What is exact: billed LLM $ by model (all benchmark runs)

From `system.billing.usage` (MODEL_SERVING SKUs), workspace fevm-newjeans-ontos.

| Model | list price (in/out per 1M) | LLM $ billed (all runs) |
|---|---|---|
| Opus 5 | $5 / $25 | **$23.22** |
| GPT-5.6-sol | $5 / $30 | **$5.17** |
| GLM 5.2 | $1.40 / $4.40 | **$3.10** |

Plus serverless **compute** ≈ $23.29 across the 30 core runs. Whole core
benchmark: **≈ $54.78**.

## The gap: per-task cost is NOT separable in v1

Two reasons, both real:

1. **Runs were concurrent** — many jobs hit the gateway in the same window, so a
   time-window split can't attribute tokens to one run.
2. **The OpenAI-compat route logs no tokens.** `system.ai_gateway.usage` records
   input/output tokens only on the **Anthropic** route; for `sol`/`glm` (mlflow
   route) the token columns are null.

So per-task LLM cost columns in this repo show the **model total**, clearly
labeled — not a per-task figure.

## v2 fix

- **`request_tags`** — the gateway usage schema has a `request_tags` map. Tagging
  each run (`kmle_run_id`) makes per-run token attribution exact, concurrency and
  route notwithstanding.
- **omni per-session $** — the team's session cost view is the independent
  cross-check; reconcile the tagged per-run totals against it.
