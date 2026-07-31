# t1_pubg — PUBG 최종 순위 백분위 예측

## 결과 요약
- 모델: LightGBM (objective=MAE), **그룹(팀) 단위** 회귀
- 검증: `matchId` 기준 GroupKFold(5) — train/test가 매치로 분리되어 있으므로 동일 구조
- 후처리: 매치 내 그룹 랭킹 재배치 + `1/(maxPlace-1)` 격자 스냅

### OOF 플레이어 단위 MAE (실제 채점과 동일한 가중 방식)
| 후처리 | MAE | 개선 |
|---|---|---|
| `raw` (클리핑만) | 0.028292 | — |
| `grid` (격자 스냅) | 0.027282 | -3.6% |
| `rank` (매치 내 랭킹 재배치) | 0.025428 | -10.1% |
| `rank_grid` | 0.024413 | -13.7% |
| **`blend_grid` (최종, α=0.72)** | **0.023978** | **-15.2%** |

참고 상한: 랭킹이 완벽할 때(oracle)의 `rank+grid` MAE는 0.00643 →
남은 오차는 거의 전부 **그룹 간 순위 예측 오차**에서 온다.
비교용 단순 베이스라인: 전부 0.5 예측 시 MAE 0.2690.

## 데이터 관찰 (핵심 인사이트)
1. `winPlacePerc`는 `(matchId, groupId)` 안에서 **항상 동일**하다
   (101,242개 그룹 전부 nunique==1). → 플레이어 단위가 아닌 **그룹 단위로 학습**하면
   노이즈가 줄고 데이터가 222,869 → 101,242행으로 줄어 학습이 빨라진다.
2. `winPlacePerc * (maxPlace - 1)`은 **100%가 정수**다. 즉 타깃은
   `1/(maxPlace-1)` 격자 위에만 존재한다. → 예측을 격자에 스냅하면 MAE가 개선된다.
3. 같은 매치 안의 그룹들은 서로 다른 순위를 가진다. → 매치 내 상대적 랭킹
   (`rank(pct=True)`)이 절대값보다 훨씬 강한 신호다.

## 파이프라인
### 1) `features.py`
- **플레이어 단위 파생 피처** (`add_player_features`)
  - `totalDistance`, `healsBoosts`, `items`, `killPlacePerc`,
    `headshotRate`, `damagePerKill`, `walkPerDuration`,
    `killsPerDistance`, `damagePerDistance`, `itemsPerDistance`,
    `longestKillPerKill`, `killsNoMove`(치터/AFK 신호), `teamworkScore`,
    `maxPlaceOverNumGroups` 등
  - `rankPoints == -1`, `killPoints/winPoints == 0`은 "미사용" 센티널이므로 NaN 처리
- **그룹 집계** (`build_group_features`): 행동 피처에 대해 `mean/max/min/sum/std`
- **매치 상대 피처**
  - 모든 `*_mean|max|min|sum`에 대해 매치 내 `rank(pct=True)` → `*_rank`
  - 그룹 평균의 매치 평균 대비 상대 편차 → `*_mdev`
  - `groupSize`, `playersInMatch`, `groupSizeRatio`, `numGroupsRatio`
- `matchType`은 solo/duo/squad + fpp + normal 플래그로 인코딩
- 최종 461개 피처

### 2) `train.py`
- LightGBM: `objective=mae, num_leaves=255, lr=0.07, feature_fraction=0.6,
  bagging_fraction=0.85, lambda_l2=1.0`, early stopping(100), 최대 1600 rounds
- GroupKFold(5) on `matchId`, 5개 fold 모델의 테스트 예측 평균
- 검증 지표는 **플레이어 단위 MAE** = 그룹 예측을 `groupSize`로 가중한 MAE
  (실제 채점과 동일)

### 3) 후처리 (`postprocess`)
5가지 모드를 OOF로 비교하여 최적 모드를 자동 선택:
| 모드 | 설명 |
|---|---|
| `raw` | [0,1] 클리핑만 |
| `grid` | `1/(maxPlace-1)` 격자로 스냅 |
| `rank` | 매치 내 그룹 예측 순위를 `(rank-1)/(numGroups-1)`로 재배치 |
| `rank_grid` | `rank` 후 격자 스냅 |
| `blend_grid` | `α*rank + (1-α)*raw` 후 격자 스냅 (**채택**, α=0.72) |

`blend_grid`의 근거: `rank` 재배치는 순위 정보를 최대한 쓰지만 `numGroups < maxPlace`인
매치에서 "빈 순위"를 무시해 편향이 생긴다. 반대로 `raw`는 값 자체는 잘 보정되어 있으나
매치 내 등간격 구조를 못 쓴다. 두 값은 매치 안에서 **모두 raw 예측의 단조 함수**라
블렌딩해도 순서가 보존되며, α는 OOF에서 0.65–0.78 구간이 평탄한 최적점이다
(`tune_postprocess.py` 참조). matchType별/numGroups 구간별 α는 추가 이득이
0.00002 수준이라 과적합 위험을 피해 전역 α를 사용했다.

엣지 케이스: `maxPlace==0 → 0`, `maxPlace==1 → 1`, 그룹이 1개인 매치는 raw 유지.

또한 `slot_assign`(매치 내 그룹을 서로 다른 격자 슬롯에 순서 보존 배정)도 실험했으나
0.0347로 크게 악화되어 폐기했다.

## 재현 방법
```bash
python solution/train.py              # 전체 학습 + outputs/submission.csv 생성
python solution/train.py --cv-only    # CV 진단만
python solution/train.py --from-cache # 저장된 예측으로 후처리만 재실행
python solution/tune_postprocess.py   # 후처리 α 튜닝 실험 재현
```
피처는 `solution/.cache/`에 pickle로 캐시된다 (삭제하면 재계산).
런타임: 피처 생성 ~10초, 학습 ~9.5분/fold (4 threads), 총 약 47분.
평균 best_iteration = 1573 (early stopping 100, 최대 1600).

## 제출 형식 검증
`write_submission`에서 다음을 assert한다.
- `test.csv`의 모든 `Id`가 정확히 한 번 포함
- `sample_submission.csv`와 행 수/Id 집합 일치
- 결측 없음, 값은 [0, 1]
