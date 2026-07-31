# 자연어 추론 (t6_klue_nli)

## 배경
전제(premise) 문장과 가설(hypothesis) 문장의 논리적 관계를 판별하는 한국어
자연어 추론(NLI) 과제입니다.

## 데이터
- `train.csv` — 컬럼: `id`, `premise`, `hypothesis`, `label`
- `test.csv` — 컬럼: `id`, `premise`, `hypothesis`
- 레이블: `entailment`(함의), `neutral`(중립), `contradiction`(모순)

## 목표 / 평가 지표
test의 각 쌍에 대해 레이블을 예측하십시오. **Accuracy**, 높을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,label`; test의 모든 id를 정확히 한 번씩,
label은 위 3개 값 중 하나.

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 ·
재현 가능한 코드를 `solution/`에 저장.
