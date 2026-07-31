# PUBG winPlacePerc prediction (t1_pubg)

Predicts the final placement percentile (`winPlacePerc`, 0..1) for each player
record in `test.csv`. Metric: MAE (lower is better).

## Approach

`winPlacePerc` is constant within a `groupId` (a team shares its placement), so
group-level information is central.

### Features (`solution/main.py`)
1. **Per-player engineered**: `totalDistance`, `items`, `killsPerWalk`,
   `dmgPerKill`, `headshotRate`.
2. **Match-normalized**: z-score of each numeric column within `matchId`
   (match stats computed on train and applied to test to avoid leakage).
3. **Group-aggregate**: per-`groupId` `mean`/`sum`/`min`/`max` of numeric
   columns + `totalDistance`/`items`, plus team size `grp_size`.

### Model
- `HistGradientBoostingRegressor` (scikit-learn only, no internet needed).
- **5-fold `GroupKFold` on `matchId`** (train/test are split by match, so this
  mirrors the true evaluation split and prevents leakage).
- Ensemble of 2 seeds (42, 2024); out-of-fold + test predictions averaged.

### Post-processing
- Per-player predictions are **averaged within each `groupId`** so all members
  of a team share one prediction (target is group-constant).
- Clipped to `[0, 1]`.

### Result
- OOF MAE (player): ~0.0395
- OOF MAE (group):  ~0.0394
- Runtime: ~7-8 min on 4 CPU cores.

## Reproduce

```bash
python solution/main.py
```
Writes `outputs/submission.csv` with columns `Id,winPlacePerc`, matching
`sample_submission.csv` exactly (all `test.csv` ids, in sample order).
