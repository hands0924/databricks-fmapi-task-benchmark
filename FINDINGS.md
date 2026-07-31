# Findings

## The model verdict — three profiles, not one winner

| | Opus 5 | GPT-5.6-sol | GLM 5.2 |
|---|---|---|---|
| task wins (of 5) | **8** | 2 | 1 |
| profile | quality leader | efficiency frontier | budget contender |
| LLM $ (all runs) | $23.22 | $5.17 | $3.10 |
| median time / task | ~69 min | ~10 min | ~49 min |
| reliability | 5/5 valid | 5/5 valid | 4/5 (1 DNF) |

- **Opus 5** wins 3/5, decisively on calibration-heavy tasks (작가판별 log-loss
  0.258 vs 0.36+). It costs ~4.5× GPT and ~7.5× GLM, and runs longest.
- **GPT-5.6-sol** lands within ~1% of the best on three tasks, is fastest
  everywhere, and is the cheapest per unit of quality. The default when latency
  and cost matter.
- **GLM 5.2** posts the best forecasting result (자전거 RMSE 210) at 1/7th Opus's
  cost — with the only DNF (감성분석). Capable; watch reliability.
- All three operated **fully in Korean** with no instruction-following failures.

## Do we need to fix the harness?  → **Yes — measured.**

Team TODO: *"Harness 에 대한 고정도 필요할지?"* We ran the same models two ways —
**fixed harness** (one pinned opencode scaffold, model swapped) and **native
stacks** (each model in its own agent CLI). Same model, different scaffold,
materially different score:

| Model · task | native stack | fixed harness | Δ |
|---|---|---|---|
| GPT-5.6-sol · 뉴스분류 (F1 ↑) | 0.824 (Codex) | **0.848** (opencode) | +0.024 |
| Opus 5 · 작가판별 (log-loss ↓) | 0.319 (Claude Code) | **0.258** (opencode) | −0.061 |
| Opus 5 · 감성분석 (acc ↑) | **0.880** (Claude Code) | 0.870 (opencode) | −0.010 |

The stack even changes *who wins* a task. **Conclusion:** a model-comparison
claim requires a fixed harness; otherwise you are benchmarking the agent product,
not the model. Native-stack numbers are still worth reporting — as "what a
customer deploys" — but they are a separate question.

## Platform notes (reusable)

- Claude streams reasoning as **structured content blocks**; a single
  OpenAI-compatible wire for all models fails — use per-model native drivers
  (Anthropic driver for Claude, OpenAI driver for GPT/GLM) against the gateway.
- The **OpenAI-compat gateway route logs no tokens** in `system.ai_gateway.usage`
  (only the Anthropic route does) — so per-run cost attribution needs
  `request_tags`, not the usage table. See COST.md.
- Of the Databricks-hosted **OSS** models, only Qwen 3.5 122B drove the agent
  reliably; Llama-4-Maverick rejected the tool schema and gpt-oss-120b emitted
  malformed tool calls — tool-use conformance gates OSS agentic work.
