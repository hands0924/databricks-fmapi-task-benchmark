# databricks-fmapi-task-benchmark

Benchmarking **coding agents on Korean ML-engineering tasks**, run entirely on
Databricks via [ucode](https://github.com/databricks/ucode) → Unity AI Gateway.
Three Foundation-Model-API models compared head-to-head under one fixed harness:

**Claude Opus 5 · GPT-5.6-sol · GLM 5.2**

## Results — 11 Korean tasks

| Task | metric | opus | sol | glm |
|---|---|---|---|---|
| [pubg_placement_prediction](./pubg_placement_prediction) | MAE ↓ | **0.02315** | 0.02334 | 0.0591 |
| [spooky_author_identification](./spooky_author_identification) | multiclass log loss ↓ | **0.2582** | 0.3625 | 0.3712 |
| [klue_ynat_news_topic](./klue_ynat_news_topic) | macro-F1 ↑ | **0.8492** | 0.8483 | 0.8452 |
| [nsmc_movie_sentiment](./nsmc_movie_sentiment) | accuracy ↑ | 0.8699 | **0.8758** | DNF |
| [seoul_bike_demand](./seoul_bike_demand) | RMSE ↓ | 299.4 | 219.6 | **209.8** |
| [klue_nli_inference](./klue_nli_inference) | accuracy ↑ | 0.8734 | **0.8986** | 0.4356 |
| [klue_sts_similarity](./klue_sts_similarity) | Pearson ↑ | **0.9585** | 0.9474 | 0.9429 |
| [beep_hate_speech](./beep_hate_speech) | macro-F1 ↑ | **0.5822** | 0.5623 | 0.5599 |
| [korquad_reading_comprehension](./korquad_reading_comprehension) | char-F1 ↑ | **0.4691** | 0.412 | DNF |
| [kornli_inference](./kornli_inference) | accuracy ↑ | **0.6462** | 0.6265 | 0.5345 |
| [korsts_similarity](./korsts_similarity) | Pearson ↑ | **0.8024** | 0.7502 | 0.7495 |
| **task wins** | | **8** | **2** | **1** |

Bold = best per task. Scored against a **hidden test split** the agent never saw
(leakage controlled by Unity Catalog ACL). n=1 run per cell.

## Three metrics (per the team spec)

| | 지표 | Source | Granularity |
|---|---|---|---|
| 1 | **결과물 퀄리티** | grader vs hidden split | exact, per run |
| 2 | **소요시간** | job wall-clock | exact, per run |
| 3 | **전체 비용** | `system.billing.usage` | **per model** (see [COST.md](./COST.md)) |

## Layout

```
<task>/
├── README.md            # per-task result table (quality · time · cost)
├── TASK_DESCRIPTION.md  # the standardized task spec (Korean)
├── PROMPT.md            # the standardized kickoff prompt (identical across models)
├── opus/                # Opus 5:       submission.csv · solution/ · metrics.json
├── sol/                 # GPT-5.6-sol:  submission.csv · solution/ · metrics.json
└── glm/                 # GLM 5.2:      submission.csv · solution/ · metrics.json
```

## Read next

- **[FINDINGS.md](./FINDINGS.md)** — model verdict + the answer to *"do we need to
  fix the harness?"* (measured: yes).
- **[METHODOLOGY.md](./METHODOLOGY.md)** — fixed-harness design, grading, gateway routing.
- **[COST.md](./COST.md)** — billed cost by model + the per-task-attribution gap.

> Methodology ported from OpenAI's [MLE-bench](https://github.com/openai/mle-bench).
> Data is **not** redistributed here (submissions and agent code only); source datasets
> are public (KLUE · NSMC · UCI · Kaggle). Run on Databricks workspace fevm-newjeans-ontos.
