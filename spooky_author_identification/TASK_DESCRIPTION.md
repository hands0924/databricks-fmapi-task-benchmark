# 괴기 소설 작가 판별 (t2_spooky)

## 배경
영문 괴기 소설 문장이 세 작가 중 누구의 글인지 판별하는 과제입니다.
- `EAP` — 에드거 앨런 포 (Edgar Allan Poe)
- `HPL` — H.P. 러브크래프트 (H.P. Lovecraft)
- `MWS` — 메리 셸리 (Mary Wollstonecraft Shelley)

데이터는 영어 텍스트이지만, 본 과제의 지시문과 작업 환경은 한국어입니다.
(MLE-bench lite 수록 과제의 한국어 이식판입니다.)

## 데이터
- `train.csv` — 컬럼: `id`, `text`(문장), `author`(EAP/HPL/MWS)
- `test.csv` — 컬럼: `id`, `text`
- `sample_submission.csv` — 제출 형식 예시

## 목표
`test.csv`의 각 문장에 대해 **세 작가 각각일 확률**을 예측하십시오.

## 평가 지표
**Multi-class Log Loss**. 낮을수록 좋습니다. 확률이 0 또는 1에 과도하게
치우치면 틀렸을 때 큰 벌점을 받으므로 보정(calibration)에 유의하십시오.

## 제출 형식
`outputs/submission.csv` — 헤더 포함, 컬럼 4개:
```
id,EAP,HPL,MWS
id02310,0.72,0.08,0.20
```
- `test.csv`의 모든 `id`가 정확히 한 번씩 포함되어야 합니다.
- 각 행의 확률은 음수가 아니어야 하며, 합이 1에 가깝도록 하십시오
  (채점 시 행 단위로 정규화됩니다).

## 규칙
- 외부 데이터 및 사전학습 자료의 추가 다운로드 금지. 인터넷 사용 금지.
- 제공된 `train.csv`만을 학습에 사용하십시오.
- 시간 예산: 2시간. 재현 가능한 코드를 `solution/` 아래에 남기십시오.
