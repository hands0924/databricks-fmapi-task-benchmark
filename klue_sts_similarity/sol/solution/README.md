# KLUE-STS solution

The model uses only the provided CSV files. It combines character/word TF-IDF
cosine similarities and surface overlap statistics with Extra Trees and
histogram gradient boosting. A low-weight sparse Ridge model supplies lexical
and domain information that is complementary to the similarity features.

Run from the task root:

```bash
python solution/train.py
```

This reports deterministic 5-fold validation scores, trains on all rows, and
writes `outputs/submission.csv`. To skip validation and only train the final
model:

```bash
python solution/train.py --no-cv
```

Test text is included only when fitting unsupervised TF-IDF vocabularies and
IDF weights; no test labels or external data are used.

Tested with Python 3.11, NumPy 1.26, pandas 2.2, SciPy 1.12, and
scikit-learn 1.4.
