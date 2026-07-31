# YNAT solution

The final model is a character-boundary TF-IDF (`1` to `5`-grams) classifier
with a linear SVM. Mild inverse-frequency class weighting is used because the
evaluation metric is macro F1.

Run from any directory with:

```bash
python solution/train_predict.py
```

The script reads the CSV files from the workspace root by default and writes
`outputs/submission.csv`. Paths can be overridden with `--train`, `--test`,
`--sample-submission`, and `--output`.

Only NumPy, pandas, and scikit-learn are required. No external data or model
weights are used.
