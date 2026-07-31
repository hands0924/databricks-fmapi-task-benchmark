# KLUE-NLI solution

This solution uses only `train.csv`. It combines character/word TF-IDF,
sentence-pair overlap features, a linear SVM, and joint decoding of examples
that share a premise. A stratified holdout on complete premise groups reached
0.907 accuracy with the default decoding strength.

Run from the task root:

```bash
python solution/train.py
```

Optional local validation:

```bash
python solution/train.py --validate
```
