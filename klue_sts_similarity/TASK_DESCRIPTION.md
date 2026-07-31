# 문장 유사도 (t7_klue_sts)

## 배경
두 한국어 문장의 의미적 유사도를 0~5 실수 점수로 예측하는 과제입니다 (KLUE-STS).

## 데이터
- `train.csv` — 컬럼: `id`, `sentence1`, `sentence2`, `score` (0~5)
- `test.csv` — 컬럼: `id`, `sentence1`, `sentence2`

## 목표 / 평가 지표
test의 각 쌍에 대해 유사도 점수를 예측하십시오.
**Pearson 상관계수**, 높을수록 좋습니다. (상수 예측은 0점 처리)

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,score`; test의 모든 id를 정확히 한 번씩,
score는 숫자 (범위 제한 없음, 0~5 권장).

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 ·
재현 가능한 코드를 `solution/`에 저장.
