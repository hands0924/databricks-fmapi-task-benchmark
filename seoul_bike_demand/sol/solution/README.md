# Reproducible solution

Run from any directory with:

```bash
python solution/train_predict.py
```

The script reads the three CSV files from the workspace root, trains three
deterministic histogram gradient boosting regressors, and writes
`outputs/submission.csv`. It uses only the supplied training data. Calendar
features are computed from `date`; non-functioning hours are set to zero after
the ensemble prediction, matching every such observation in the training set.

The model settings were selected with chronological rolling holdouts rather
than random validation. Required packages are listed in `requirements.txt`.
