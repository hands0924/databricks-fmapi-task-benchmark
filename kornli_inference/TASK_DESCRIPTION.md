# 자연어 추론 — 번역체 (t10_kornli)

## 배경
MultiNLI를 한국어로 번역한 KorNLI 데이터로, 전제-가설 쌍의 논리 관계를
판별합니다. t6과 동일한 과제이지만 번역체 문장(다양한 장르)이라는 점이
다릅니다.

## 데이터
- `train.csv` — 컬럼: `id`, `sentence1`(전제), `sentence2`(가설), `label`
- `test.csv` — 컬럼: `id`, `sentence1`, `sentence2`
- 레이블: `entailment`, `neutral`, `contradiction`

## 목표 / 평가 지표
**Accuracy**, 높을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,label`; test의 모든 id를 정확히 한 번씩.

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 ·
재현 가능한 코드를 `solution/`에 저장.
