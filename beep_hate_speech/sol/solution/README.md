# Reproducible solution

Run from any directory with:

```bash
python solution/train_predict.py
```

The script trains a class-weighted linear SVM on character-boundary TF-IDF
1-5 grams. This representation handles Korean slang, misspellings, and spacing
variation without external tokenizers or data. It writes and validates
`outputs/submission.csv` against `sample_submission.csv`.

Tested with Python 3.12, pandas 1.5.3, and scikit-learn 1.4.2.
