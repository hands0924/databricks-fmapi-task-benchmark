# Methodology

Ported from OpenAI **MLE-bench**: agents write and run code against real tasks;
submissions are graded on a hidden split; fresh splits keep answer keys out of
the training path.

## The loop (entirely on Databricks)

```
UC Volume: task pack            Serverless Job: agent          Serverless Job: grader
  spec(KR)·train·HIDDEN test  →   ucode → AI Gateway,       →   scores vs hidden split
  ·grader                         reads spec+train only,        (ACL'd away from agent)
                                  writes submission.csv     →   MLflow score + system-table cost
```

## Fixed harness (this repo's numbers)

- One pinned **opencode** scaffold, byte-identical config and Korean kickoff
  prompt; only the **model** changes: `databricks-claude-opus-5` /
  `databricks-gpt-5-6-sol` / `databricks-glm-5-2`.
- Per-model native drivers: Anthropic driver → `/ai-gateway/anthropic/v1`;
  OpenAI driver → `/ai-gateway/mlflow/v1`.
- Runs as serverless Databricks Jobs (same compute class for every model).

## Grading

- Hidden test lives in a separate UC volume, ACL'd away from the run principal —
  leakage control by **permission**, not convention.
- Deterministic graders; re-runs reproduce identically. Format-invalid
  submission ⇒ **DNF**.

## Standardized prompt

Every model receives the identical kickoff (`PROMPT.md`) plus the task's
`TASK_DESCRIPTION.md`. No model-specific hints. 2-hour wall-clock cap, n=1.
