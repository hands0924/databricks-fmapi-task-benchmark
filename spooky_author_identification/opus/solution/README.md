# t2_spooky — Spooky Author Identification (EAP / HPL / MWS)

Metric: multi-class log loss (lower is better). No external data, no pretrained weights;
only `train.csv` is used for fitting.

## How to reproduce

Run from the task root (the directory holding `train.csv`, `test.csv`):

```bash
python solution/run.py                 # full pipeline (~50-70 min on 4 cores)
python solution/run.py --ensemble-only # re-do only level-2 using cached OOF in work/oof
```

Outputs `outputs/submission.csv` with columns `id,EAP,HPL,MWS`, one row per test id.
All intermediate artefacts are cached in `work/` (features in `work/cache`,
per-model OOF/test predictions in `work/oof`), so the run is resumable.

## Approach

**Level 1 — a zoo of 147 linear / naive-Bayes text models** (`solution/models.py`),
each evaluated with the *same* 5-fold stratified split (seed 42) so that their
out-of-fold predictions are stackable. Feature blocks (`solution/features.py`):

| block      | description |
|------------|-------------|
| `word`     | word TF-IDF 1-2 grams, `min_df=2`, sublinear |
| `word1`    | word TF-IDF unigrams, `min_df=1` |
| `word3`    | word TF-IDF 1-3 grams, `min_df=3` |
| `char`     | `char_wb` TF-IDF 2-6 grams, 300k features |
| `charfull` | `char` (cross-word) TF-IDF 2-5 grams, 300k features |
| `charcase` | `char_wb` TF-IDF 2-5 grams with **case preserved** (capitalisation habits) |
| `pos`      | TF-IDF over a hand-rolled token-shape sequence (function words kept verbatim, content words mapped to suffix/case classes) — captures syntax/style without any external tagger |
| `stopw`    | function-word-only stream (content words collapsed to `X`, punctuation kept) — the classic authorship-attribution view; weak alone (~0.66) but very decorrelated |
| `wc`,`cc`  | raw counts (word 1-2 grams / char_wb 2-5 grams), plus binarised variants |
| `all`, `allpos`, `wordchar` | horizontal concatenations |
| `hand`     | 26 dense stylometric features (length, punctuation rates, type/token ratio, sentence stats, suffix rates) |
| `svd`      | 180 TruncatedSVD components of `word`+`char` |

Model families: `MultinomialNB` (very strong here with a *tiny* alpha on sublinear
TF-IDF — best single model family, ~0.379), `ComplementNB`, `BernoulliNB`,
`LogisticRegression` (needs large `C` on sublinear TF-IDF; `lr_all_C12` ≈ 0.384),
NBSVM (`NBFeatures` log-count-ratio reweighting + LR), and `LinearSVC` whose
decision function is converted to calibrated probabilities by a temperature-scaled
softmax fitted on an internal holdout (`SoftmaxDecision`, `svc_all_C0.3` ≈ 0.372).

**Pruning** — models with OOF log loss > 1.2 and near-duplicates
(prediction correlation > 0.9995) are dropped: 147 → 133 kept.

**Level 2 — two independent combiner families:**

1. **Bagged LR stack**: `LogisticRegression` on the centred log-probabilities of
   every kept model (399 meta features), 5-fold CV, averaged over 3 different
   fold seeds to damp fold noise. `C ∈ {0.05, 0.15, 0.5, 1.5}`, and two variants
   that additionally see the 26 dense stylometric features (`solution/stack.py`).
2. **Greedy geometric blend**: Caruana forward selection with replacement over a
   weighted geometric mean (weighted average in log space + softmax
   re-normalisation), then continuous Powell refinement on the selected support
   (`solution/blend.py`).

**Level 3** — the same greedy geometric procedure over the six level-2 candidates
plus the level-1 greedy blend. It picked
`0.59 × lrh0.3 + 0.21 × l1greedy + 0.20 × lr1.5`.

Geometric averaging is preferred over arithmetic because log loss is a log-space
penalty, and it stops the ensemble being dragged towards over-confident members.
Final probabilities are clipped at `2e-6` and row-normalised, which caps the
worst-case per-row penalty at ~13 nats instead of unbounded.

## Results (5-fold OOF log loss)

| model | OOF |
|---|---|
| plain "word+char TF-IDF → LR" baseline | 0.4255 |
| best single NB (`mnb_word_a0.03`) | 0.3790 |
| best single SVM (`svc_all_C0.3`) | 0.3719 |
| best single LR (`lr_all_C12`) | 0.3840 |
| level-1 greedy geometric blend | 0.2844 |
| best single level-2 stack (`lrh0.3`) | 0.2660 |
| **level-3 blend (submitted)** | **0.2642** |

Exact weights and per-candidate scores are dumped to `work/final_weights.txt`.

## Notes / caveats

- The level-2 and level-3 weights are chosen on the same OOF predictions they are
  scored on, so 0.2642 is mildly optimistic; expect test log loss a little above
  it. The level-2 stacks themselves are properly cross-validated, and bagging over
  fold seeds plus keeping the level-3 search short (few candidates) limits the
  optimism.
- Level-1 OOF all comes from one shared 5-fold split (seed 42) so predictions are
  mutually stackable; level-2 uses different seeds.
- `MultinomialNB` on *sublinear* TF-IDF with a very small `alpha` (~0.003–0.03) is
  the surprise workhorse here — it beats logistic regression on the same features
  and its errors are decorrelated from the margin-based models.
- Logistic regression needs unusually large `C` (12–80) on these sublinear TF-IDF
  blocks; the default `C=1` is heavily under-fit (0.67 vs 0.43 on `word`).
- No external data, no pretrained embeddings, no internet: every vectoriser is
  fitted on `train.csv` + `test.csv` text (transductive vocabulary only, labels
  never touched), and all supervised fitting uses `train.csv` alone.
