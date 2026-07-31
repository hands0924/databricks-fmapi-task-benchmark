# KorNLI solution

Run from the task workspace root:

```bash
python solution/train_predict.py
```

The script uses only `train.csv` and local Python packages. It combines a
character TF-IDF Linear SVM over the hypothesis with binned lexical-overlap,
length, negation, and pair-interaction features. For repeated premises, it
penalizes labels already observed for that premise in the training set.

A fixed stratified 80/20 split with seed 2026 was used during development.
The selected model reached 0.5795 accuracy before and 0.6052 after applying
the repeated-premise constraint.
