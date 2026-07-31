# 혐오 표현 분류 (t8_beep)

## 배경
한국어 연예 뉴스 댓글의 혐오 표현 수준을 분류하는 과제입니다 (BEEP! 데이터셋).
구어체·은어·변형 표기가 많은 실사용 텍스트입니다.

## 데이터
- `train.csv` — 컬럼: `id`, `comment`, `label`
- `test.csv` — 컬럼: `id`, `comment`
- 레이블: `none`(해당없음), `offensive`(공격적), `hate`(혐오)

## 목표 / 평가 지표
test의 각 댓글에 대해 레이블을 예측하십시오. **Macro F1**, 높을수록 좋습니다.

## 제출 형식
`outputs/submission.csv` — 컬럼 `id,label`; test의 모든 id를 정확히 한 번씩,
label은 위 3개 값 중 하나.

## 규칙
외부 데이터/인터넷 금지 · train.csv만 사용 · 시간 예산 2시간 ·
재현 가능한 코드를 `solution/`에 저장.
