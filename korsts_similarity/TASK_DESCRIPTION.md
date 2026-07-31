# 문장 유사도 — 번역체 (t11_korsts)

## 배경
STS-B를 한국어로 번역한 KorSTS 데이터로, 두 문장의 의미 유사도를 0~5 실수로
예측합니다.

## 데이터
- `train.csv` — 컬럼: `id`, `sentence1`, `sentence2`, `score` (0~5)
- `test.csv` — 컬럼: `id`, `sentence1`, `sentence2`

## 목표 / 평가 지표
**Pearson 상관계수**, 높을수록 좋습니다. (상수 예측은 0점 처리)

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,score`; test의 모든 id를 정확히 한 번씩.

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 ·
재현 가능한 코드를 `solution/`에 저장.
