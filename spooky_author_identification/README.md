# spooky_author_identification

Korean ML-engineering task. Metric: **multiclass log loss** (lower is better). Full spec in [`TASK_DESCRIPTION.md`](./TASK_DESCRIPTION.md); standardized prompt in [`PROMPT.md`](./PROMPT.md).

## Results — Opus 5 vs GPT-5.6-sol vs GLM 5.2

3 models, one fixed harness (ucode → Databricks AI Gateway), model swapped.

| Model | 결과물 퀄리티 (multiclass log loss) | 소요시간 | LLM 비용 |
|---|---|---|---|
| Opus 5 | **0.2582** | 94 min | $23.22 |
| GPT-5.6-sol | 0.3625 | 15 min | $5.17 |
| GLM 5.2 | 0.3712 | 105 min | $3.10 |

Each model folder (`opus/` · `sol/` · `glm/`) holds that model's `submission.csv`, the agent-written `solution/` code, and `metrics.json`.

> **퀄리티**: scored vs a hidden test split the agent never saw; bold = best; n=1 run. **소요시간**: wall-clock, exact. **LLM 비용**: model total across *all* benchmark runs (per-task not separable in v1 — see [`../COST.md`](../COST.md)).
