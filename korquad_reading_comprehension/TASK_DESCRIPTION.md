# 기계 독해 (t9_korquad)

## 배경
한국어 위키백과 문단(context)과 질문(question)이 주어지면, 문단에서 답을
찾아내는 기계 독해 과제입니다 (KorQuAD 1.0). 답은 항상 문단 내에 존재합니다.

## 데이터
- `train.csv` — 컬럼: `id`, `context`, `question`, `answer`
- `test.csv` — 컬럼: `id`, `context`, `question`
- train과 test는 서로 다른 위키 문서에서 추출되었습니다.

## 목표 / 평가 지표
test의 각 질문에 대한 답 문자열을 예측하십시오.
**문자 단위 F1** (공백 제거 후 문자 중복집합 F1의 평균), 높을수록 좋습니다.
짧고 정확한 답(대개 문단 내 연속 구간)이 유리합니다.

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,answer`; test의 모든 id를 정확히 한 번씩.

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 ·
재현 가능한 코드를 `solution/`에 저장.
