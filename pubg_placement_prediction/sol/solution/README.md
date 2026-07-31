# PUBG placement solution

Run from the workspace root:

```bash
python solution/train.py
```

This reads `train.csv` and `test.csv` and writes `outputs/submission.csv`.
Only NumPy, pandas, and scikit-learn are required. The random seed and model
configuration are fixed in `train.py`.

The model operates at the team level. It aggregates player statistics by team,
adds within-match percentile features, and trains a histogram gradient boosting
regressor with team size as sample weight. Predictions are ordered within each
match, blended with the calibrated raw prediction, and rounded to the valid
`maxPlace` placement grid.

To reproduce the match-level holdout comparison used for model selection:

```bash
python solution/train.py --validate-all
```
