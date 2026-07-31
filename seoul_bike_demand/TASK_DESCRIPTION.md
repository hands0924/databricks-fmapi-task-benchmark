# 서울시 공공자전거 수요 예측 (t5_bike)

## 배경
서울시 공공자전거(따릉이형 시스템)의 **시간 단위 대여량**을 예측하는 과제입니다.
날씨·계절·공휴일 정보를 활용한 시계열/회귀 모델링 능력을 평가합니다.

## 데이터
- `train.csv` — 과거 구간의 시간별 데이터 (타깃 포함)
- `test.csv` — 미래 구간의 시간별 데이터 (타깃 제외)
- `sample_submission.csv` — 제출 형식 예시

주요 컬럼: `id`(YYYYMMDD_HH), `date`, `hour`, `temperature_c`, `humidity_pct`,
`wind_speed_ms`, `visibility_10m`, `dew_point_c`, `solar_radiation_mj`,
`rainfall_mm`, `snowfall_cm`, `seasons`, `holiday`, `functioning_day`,
그리고 train에만 있는 타깃 `rented_bike_count`.

**주의:** 시간순 분할입니다 — test 구간은 train 구간 이후의 기간입니다.
미래 정보 누출(leakage)이 없도록 검증 전략을 설계하십시오.

## 목표
test 구간의 각 시간대별 `rented_bike_count`를 예측하십시오.

## 평가 지표
**RMSE** (Root Mean Squared Error). 낮을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 헤더 포함, 컬럼 2개:
```
id,rented_bike_count
20181005_00,312
20181005_01,201
```
- `test.csv`의 모든 `id`가 정확히 한 번씩 포함되어야 합니다.
- 예측값은 숫자여야 합니다 (정수/실수 무관).

## 규칙
- 외부 데이터(실제 기상 기록 등) 사용 금지. 인터넷 사용 금지.
- 제공된 `train.csv`만을 학습에 사용하십시오.
- 시간 예산: 2시간. 재현 가능한 코드를 `solution/` 아래에 남기십시오.
