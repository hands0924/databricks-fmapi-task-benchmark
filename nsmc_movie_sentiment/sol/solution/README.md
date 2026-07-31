# NSMC solution

The final model is an ensemble of three linear SVM classifiers trained on
TF-IDF features. It combines unrestricted character n-grams (60%),
word-boundary character n-grams (35%), and word n-grams (5%). All components
use only `train.csv`.

Run from the workspace root:

```bash
python solution/train.py
```

This deterministically creates `outputs/submission.csv`. The implementation
requires Python 3 with pandas, NumPy, and scikit-learn. `validate.py` reproduces
the fixed, stratified holdout model comparison used to choose the ensemble.
