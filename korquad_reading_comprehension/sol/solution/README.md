# KorQuAD classical extractive QA

This solution uses only `train.csv` and packages available in the environment. It
generates short answer spans from question-relevant sentences and trains a
gradient-boosted ranker directly on character F1.

Run from the task root:

```bash
python solution/train.py
```

The deterministic output is written to `outputs/submission.csv`. Use
`python solution/train.py --validate --max-train-rows 4000` for a quick
document-grouped local validation run.
