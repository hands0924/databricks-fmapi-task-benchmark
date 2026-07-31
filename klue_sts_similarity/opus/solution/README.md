# t7_klue_sts — 한국어 문장 유사도 (Pearson)

## 환경 제약
GPU/`torch`/`transformers` 없음, 인터넷 금지 → 사전학습 언어모델을 쓸 수 없으므로
**직접 설계한 유사도 특징 + 고전 ML 앙상블**(numpy/scipy/scikit-learn, CPU 4코어)로 해결.

## 재현 방법
```bash
python solution/run.py plain   # 레벨1 모델 (plain feature)      -> work/level1_v2.npz
python solution/run.py aug     # 레벨1 모델 (문장순서 swap 증강)  -> work/level1_v3aug.npz
python solution/blend.py       # 최종 블렌딩 -> outputs/submission.csv
```
`run.py`는 각각 단독으로도 유효한 `outputs/submission.csv`를 씁니다.
특징 행렬은 `work/feat_v3*.npz`에 캐시됩니다(최초 1회 약 70초/뷰).
총 소요시간 약 15분(4코어 CPU).

## 특징 (solution/features.py, 130개 스칼라 + 256개 임베딩 상호작용)
학습·평가 텍스트만 사용(라벨 미사용, transductive unsupervised 표현학습).

- **TF-IDF 코사인** 7종 표현: `char_wb(2-4)`, `char_wb(3-5)`, `char(2-3)`,
  `word(1-1)`, `word(1-2)`, 어간(prefix stemming) `word(1-1)`,
  **자모 분해(jamo)** `char_wb(3-5)`
- 각 표현별 **IDF 가중 비대칭 커버리지** 4종 (a→b, b→a, min, max)
- **BM25** 점수(word/stem 공간, 양방향 + 대칭화)
- **LSA 단어벡터 기반 soft alignment** (BERTScore식 P/R/F, IDF 가중) 및
  IDF 가중 문장 센트로이드 코사인
- **문자 3-gram Jaccard 기반 토큰 정렬** (한국어 활용/어미 변화에 강건)
- 집합 유사도(Jaccard/Dice/containment): 토큰, 어간, 문자 2·3-gram, 자모 3-gram
- 문자열 유사도: `difflib` ratio(문자/토큰/자모/어간), LCS, 최장공통부분문자열
- 길이/토큰수 및 그 차이·비율
- **숫자 일치/충돌**(예: 금액·기간 불일치), 숫자 상대오차
- 부정 표현(`안/못/없/않`) 개수 차이, 물음표·감탄사, 첫·마지막 토큰 일치
- 미매칭 토큰의 **IDF 질량 비율**(word/stem 공간)
- SVD(LSA) 임베딩 쌍 상호작용 `|u−v|`, `u⊙v` (64차원 × 4블록)

## 모델 (solution/run.py)
5-fold OOF, 동일 fold 시드로 두 세트(plain/aug) 학습:
`HistGradientBoosting`×2, `SVR(RBF)`, `ExtraTrees`, `RidgeCV`(dense),
`Ridge`(280k차원 희소 쌍표현 `[min(a,b), |a−b|]` on char/word/stem TF-IDF).
swap 세트는 문장1↔문장2를 바꾼 특징으로 학습 데이터를 2배 증강하고 예측 시 두 순서를 평균(TTA).

## 블렌딩 (solution/blend.py)
12개 레벨1 예측을 (a) NNLS 가중 평균, (b) dense 특징 + 레벨1 예측을 입력으로 하는
Ridge 스태킹 으로 결합하고, 두 결과의 평균을 5-fold OOF Pearson으로 선택.

## 결과 (5-fold OOF Pearson)
| 모델 | Pearson |
|---|---|
| char_wb TF-IDF 코사인 단독 (baseline) | 0.8374 |
| HistGBM (단일 최강) | 0.9482 |
| 12모델 NNLS 블렌드 | 0.9539 |
| **Ridge 스태킹 + NNLS 평균 (최종 제출)** | **0.9548** |
