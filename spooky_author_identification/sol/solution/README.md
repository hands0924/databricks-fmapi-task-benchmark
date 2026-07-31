# Solution

Run from the workspace root:

```bash
python solution/train.py
```

The script trains word and character TF-IDF logistic-regression models on all
of `train.csv`, blends their probabilities, validates the submission schema,
and writes `outputs/submission.csv`.

Model parameters and the blend were selected with two fixed stratified
holdouts. The corresponding experiment is reproducible with:

```bash
python solution/experiment.py
```

Required packages: Python 3, NumPy, pandas, and scikit-learn.
