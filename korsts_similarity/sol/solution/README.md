# KorSTS solution

Run from the task directory:

```bash
python solution/train.py
```

The script uses only `train.csv` to fit its text transforms and models. It
combines character/word TF-IDF similarity, surface overlap and latent semantic
features with histogram gradient boosting, extremely randomized trees and a
sparse ridge model. Five-fold out-of-fold predictions fit a non-negative linear
blend. Exact sentence-pair repeats use the mean available training annotation.

The fixed random seed is `20260731`. On the fixed five-fold split used during
development, the blend obtained approximately 0.739 Pearson before exact-pair
replacement. The generated file is `outputs/submission.csv`.
