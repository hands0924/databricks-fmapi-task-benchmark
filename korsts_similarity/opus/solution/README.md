# t11_korsts — 문장 유사도 (KorSTS) 솔루션

## 실행

```bash
python solution/run.py      # 반드시 과제 루트에서 실행 (약 8분, CPU 4코어)
```
`outputs/submission.csv` (`id,score`) 를 생성한다. 중간 산출물은
`solution/cache/` 에 캐싱된다(삭제해도 재생성됨).

## 환경 제약

이 워크스페이스에는 `torch`/`transformers` 가 없고 GPU도 없다(사전학습 가중치
다운로드도 금지). 따라서 사전학습 문장 인코더 없이, `train.csv`+`test.csv`
텍스트만으로 만든 **특징 공학 + 2단계 스태킹** 으로 해결했다.

## 구조

### 1단계 (level-1)

**(a) 밀집 특징 239개** (`feats.py`, `emb.py`, `feats2.py`)
- 다중 tf-idf 공간의 코사인: 문자 n-gram(char/char_wb), **자모 분해** n-gram
  (한국어 어미 변화에 강건), 어절, 어절 접두 2·3자(의사 어간).
- 각 공간별 자카드/포함도, **idf 가중 커버리지**(min/max/조화평균).
- TruncatedSVD(LSA) 잠재공간 코사인·L1·L2 거리.
- BM25(양방향, 3가지 토큰화), soft-cosine measure.
- 토큰 **soft-alignment**: 자모 문자 n-gram 토큰 벡터로 최적 매칭 유사도를 구해
  임계값(0.9/0.75/0.6/0.45)별 idf 질량 커버리지, **미매칭 idf 질량**, 어순
  (정렬 위치의 스피어만 상관).
- difflib opcode 통계(equal/replace/insert/delete 비율), LCS/최장공통부분문자열,
  숫자·라틴문자·부정어(안/않/없/못)·물음표 불일치 등 어휘 특징.
- 과제 코퍼스(train+test 문장, 라벨 미사용)에서 학습한 **PPMI + SVD 단어 임베딩**
  기반 코사인·정렬 특징(문장 내 및 페어 간 공기 행렬 사용).

**(b) 희소 "학습된 유사도" 릿지 7종**
L2 정규화 tf-idf 벡터 a,b 의 상호작용 블록 `[a*b, min(a,b), |a-b|]` 를 입력으로
릿지 회귀 → 어떤 n-gram의 일치/불일치가 점수에 중요한지 직접 학습.
자모 (3,5)/(3,6), 문자 (2,5), 어간 어절 n-gram 등. 단독 CV 0.51~0.71.

**(c) (a)+(b의 OOF 예측)** 을 입력으로 재학습한 밀집 모델 3종(특징 수준 스태킹).

모델: Ridge, SVR(RBF, C=3/8), KernelRidge, HistGradientBoosting×2,
ExtraTrees, MLP(64, 3시드). 모든 1단계 예측은 **5-fold × 2 시드 반복 CV** 로
OOF/테스트 예측을 만든다(테스트 예측은 10개 모델 배깅 평균).

### 2단계 (level-2)
1단계 OOF 17개 열에 대해 **비음수 Ridge**(`positive=True, alpha=1`)로 가중 결합.

### 후처리
정규화된 문장쌍 키가 train에 존재하는 테스트쌍(23/1150)은 예측을
`0.9 * (train 점수) + 0.1 * (모델 예측)` 로 대체. train 내 중복쌍의 점수 편차가
매우 작아(중앙값 0.0, 평균 0.165) 안전한 신호다. 마지막에 [0,5]로 클리핑.

## 검증 결과 (train 4,599행, 5-fold × 2시드 OOF Pearson)

| 모델 | CV Pearson |
|---|---|
| 비지도 char n-gram 코사인 (베이스라인) | 0.619 |
| 희소 학습 유사도 (자모 3-5, prod/min/abs) | 0.709 |
| Ridge / SVR / HGB / ET / MLP (밀집 239 특징) | 0.736 ~ 0.750 |
| + 희소 OOF 특징 추가한 밀집 모델 | 0.769 ~ 0.773 |
| **2단계 스태킹 (최종)** | **0.7815** |

## 파일
- `feats.py` — 정규화/자모 변환, 다중 tf-idf 공간, 어휘·문자열 특징
- `emb.py` — 코퍼스 자체 PPMI+SVD 단어 임베딩 특징
- `feats2.py` — BM25, soft-cosine, 정렬 커버리지, opcode 특징
- `run.py` — 전체 파이프라인 (특징 → 1단계 → 2단계 → 중복쌍 보정 → 제출 저장)
