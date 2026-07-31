# 배틀그라운드 최종 순위 예측 (t1_pubg)

## 배경
PUBG(배틀그라운드, 크래프톤) 매치 로그를 바탕으로 각 플레이어의 **최종 순위
백분위**를 예측하는 과제입니다. 이동 거리, 킬, 아이템 사용 등 게임 내 행동
데이터로 승부 결과를 모델링합니다.

## 데이터
- `train.csv` — 플레이어 단위 레코드 (타깃 포함)
- `test.csv` — 타깃 제외
- `sample_submission.csv` — 제출 형식 예시

주요 컬럼: `Id`(플레이어 레코드), `groupId`(팀), `matchId`(매치),
`kills`, `damageDealt`, `walkDistance`, `boosts`, `heals`, `matchType` 등.
타깃 `winPlacePerc` — 0(꼴찌)~1(우승)의 순위 백분위.

**주의:** train과 test는 **서로 다른 매치**로 분리되어 있습니다(matchId 기준).
같은 매치 내 그룹 구조(`groupId`)를 활용한 피처 엔지니어링이 유효합니다.

## 목표
`test.csv`의 각 `Id`에 대해 `winPlacePerc`를 예측하십시오.

## 평가 지표
**MAE** (Mean Absolute Error). 낮을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 헤더 포함, 컬럼 2개:
```
Id,winPlacePerc
7f96b2f878858a,0.4444
```
- `test.csv`의 모든 `Id`가 정확히 한 번씩 포함되어야 합니다.
- 예측값은 숫자여야 합니다 (0~1 범위 권장).

## 규칙
- 외부 데이터 사용 금지. 인터넷 사용 금지.
- 제공된 `train.csv`만을 학습에 사용하십시오.
- 시간 예산: 2시간. 재현 가능한 코드를 `solution/` 아래에 남기십시오.
