"""TXT-7 키프레이즈 추출 (multilabel F1).

이 태스크는 문서에서 핵심 키워드(keyphrase)를 추출하는 다중 레이블 분류 태스크.
- 입력: 논문 제목 + 초록 텍스트
- 출력: 쉼표로 구분된 키프레이즈 목록
- 평가: 정규화된 키프레이즈 집합 간 precision/recall/F1
"""

from __future__ import annotations

import re
from typing import Any

from src.adapters.fmapi import FMAPIClient, build_text_message
from src.datasets_loader import load_registry
from src.scoring.metrics import multilabel_prf
from src.tasks.base import Task, Sample, register
from src.tasks.common import detect_column, load_dataset_rows


def normalize_keyphrase(phrase: str) -> str:
    """키프레이즈 정규화: 소문자, 공백 정리, 특수문자 제거.

    Args:
        phrase: 원본 키프레이즈

    Returns:
        정규화된 키프레이즈
    """
    # 소문자 변환
    phrase = phrase.lower()
    # 양쪽 공백 제거
    phrase = phrase.strip()
    # 여러 공백을 단일 공백으로
    phrase = re.sub(r'\s+', ' ', phrase)
    return phrase


@register
class Txt7Task(Task):
    """키프레이즈 추출 태스크 (TXT-7)."""

    task_id: str = "TXT-7"
    kind: str = "multilabel"
    is_vision: bool = False

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """INSPEC 데이터셋 (memray/inspec 파쿠엣 버전)에서 키프레이즈 샘플 로드.

        각 샘플:
        - inputs: {"document": 제목 + 초록}
        - reference: 정규화된 키프레이즈 집합

        NOTE: 실제 데이터셋이 로드되지 않으면 예외를 발생시킴. 합성 데이터 폴백 없음.
        """
        datasets_map = self.config.get("datasets")  # {en: keyphrase}
        if not datasets_map:
            raise ValueError("config에 datasets 맵이 없음")

        samples = []
        sample_id = 0

        for lang, dataset_key in datasets_map.items():
            # 실제 데이터셋 로드 (실패 시 예외 발생)
            hf_ds = load_dataset_rows(dataset_key, n, seed, default_split="test")

            # 컬럼명 감지
            col_title = self._detect_title_column(hf_ds)
            col_abstract = self._detect_abstract_column(hf_ds)
            col_keyphrases = self._detect_keyphrases_column(hf_ds)

            for idx, row in enumerate(hf_ds):
                title = row.get(col_title, "")
                abstract = row.get(col_abstract, "")
                keyphrases_raw = row.get(col_keyphrases, "")

                # 타입 검증
                if not isinstance(title, str):
                    title = str(title) if title else ""
                if not isinstance(abstract, str):
                    abstract = str(abstract) if abstract else ""

                # 문서 텍스트 구성
                document_text = f"{title}\n\n{abstract}".strip()

                if not document_text:
                    continue

                # 키프레이즈 파싱 (memray/inspec는 세미콜론으로 구분된 문자열)
                keyphrases_set = self._parse_keyphrases(keyphrases_raw)

                if not keyphrases_set:
                    continue

                sample = Sample(
                    sample_id=sample_id,
                    inputs={"document": document_text},
                    reference=keyphrases_set,
                    lang=lang,
                    meta={
                        "dataset": dataset_key,
                        "source_idx": idx,
                        "n_keyphrases": len(keyphrases_set),
                    },
                )
                samples.append(sample)
                sample_id += 1

                if sample_id >= n:
                    break

            if sample_id >= n:
                break

        return samples

    def _parse_keyphrases(self, keyphrases_raw: Any) -> set[str]:
        """키프레이즈 문자열/리스트를 정규화된 집합으로 변환.

        memray/inspec의 'keywords' 필드는 세미콜론으로 구분된 문자열.

        Args:
            keyphrases_raw: 문자열 또는 리스트

        Returns:
            정규화된 키프레이즈 집합
        """
        keyphrases_set = set()

        if isinstance(keyphrases_raw, str):
            # 세미콜론으로 구분
            phrases = [p.strip() for p in keyphrases_raw.split(";")]
            for phrase in phrases:
                if phrase:
                    normalized = normalize_keyphrase(phrase)
                    if normalized:
                        keyphrases_set.add(normalized)
        elif isinstance(keyphrases_raw, list):
            # 리스트 형식
            for kp in keyphrases_raw:
                if isinstance(kp, str) and kp.strip():
                    normalized = normalize_keyphrase(kp)
                    if normalized:
                        keyphrases_set.add(normalized)

        return keyphrases_set

    def _detect_title_column(self, hf_ds: list[dict[str, Any]]) -> str:
        """제목 컬럼명 감지."""
        return detect_column(
            hf_ds, ["title", "document_title", "text_title"], what="제목 컬럼"
        )

    def _detect_abstract_column(self, hf_ds: list[dict[str, Any]]) -> str:
        """초록 컬럼명 감지 (memray/inspec은 'abstract' 또는 'fulltext')."""
        return detect_column(
            hf_ds,
            ["abstract", "fulltext", "text", "document", "content"],
            what="초록 컬럼",
        )

    def _detect_keyphrases_column(self, hf_ds: list[dict[str, Any]]) -> str:
        """키프레이즈 컬럼명 감지 (memray/inspec은 세미콜론 구분 'keywords')."""
        return detect_column(
            hf_ds,
            ["keywords", "keyphrases", "keyphrase", "keyword", "prmu"],
            what="키프레이즈 컬럼",
        )

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """문서를 제시하고 키프레이즈 추출 요청.

        모델에게:
        1. 문서 제시 (제목 + 초록)
        2. 핵심 키프레이즈를 쉼표로 구분된 목록으로 응답 요청
        """
        document = sample.inputs["document"]

        prompt = f"""Extract the key phrases or keywords from the following academic document. List them as a comma-separated list with no additional text.

Title and Abstract:
{document}

Key phrases (comma-separated):"""

        return build_text_message(prompt)

    def parse_output(self, raw_text: str, sample: Sample) -> set[str]:
        """모델 응답을 정규화된 키프레이즈 집합으로 파싱.

        쉼표 또는 다른 구분자로 분리하고 정규화.

        Args:
            raw_text: 모델 응답
            sample: Sample (사용하지 않음, 인터페이스 준수)

        Returns:
            정규화된 키프레이즈 집합
        """
        if not raw_text or not raw_text.strip():
            return set()

        # 여러 구분자(쉼표, 세미콜론, 줄바꿈) 처리
        # 먼저 줄바꿈을 쉼표로 변환
        text = raw_text.replace("\n", ",").replace(";", ",")

        # 쉼표로 분리
        phrases = text.split(",")

        # 각 키프레이즈 정규화
        keyphrases_set = set()
        for phrase in phrases:
            normalized = normalize_keyphrase(phrase)
            if normalized:  # 비어있지 않은 것만
                keyphrases_set.add(normalized)

        return keyphrases_set

    def score(self, parsed: list[set[str]], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 예측 키프레이즈 집합에 대해 precision/recall/F1 계산.

        multilabel_prf 함수 사용 (micro/macro 평균).

        Returns:
            {precision, recall, f1, n_evaluated, per_language: {...}}
        """
        # 예측과 정답 집합 추출
        pred_sets = parsed
        gold_sets = [sample.reference for sample in samples]

        # multilabel_prf 계산
        metrics = multilabel_prf(pred_sets, gold_sets)

        # 언어별 통계
        lang_stats = {}
        for lang in {"en", "ko"}:
            lang_indices = [i for i, s in enumerate(samples) if s.lang == lang]
            if lang_indices:
                lang_pred = [pred_sets[i] for i in lang_indices]
                lang_gold = [gold_sets[i] for i in lang_indices]
                lang_metrics = multilabel_prf(lang_pred, lang_gold)
                lang_stats[lang] = {
                    "precision": lang_metrics["micro_precision"],
                    "recall": lang_metrics["micro_recall"],
                    "f1": lang_metrics["micro_f1"],
                    "n_evaluated": len(lang_indices),
                }

        return {
            "precision": metrics["micro_precision"],
            "recall": metrics["micro_recall"],
            "f1": metrics["micro_f1"],
            "n_evaluated": len(parsed),
            "per_language": lang_stats,
            "macro_precision": metrics["macro_precision"],
            "macro_recall": metrics["macro_recall"],
            "macro_f1": metrics["macro_f1"],
        }


if __name__ == "__main__":
    """End-to-end 테스트: 3샘플로 FMAPI 호출 및 P-R-F1 계산."""
    import sys

    test_config = {
        "datasets": {
            "en": "keyphrase",
        }
    }

    registry = load_registry()
    task = Txt7Task(test_config, registry)

    print("=" * 70)
    print("TXT-7 Keyphrase Extraction - End-to-End Test")
    print("=" * 70)
    print("\nLoading 3 samples...")

    try:
        samples = task.load_samples(n=3, seed=42)
        print(f"✓ Loaded {len(samples)} samples\n")

        for sample in samples:
            doc_preview = sample.inputs["document"][:80].replace("\n", " ")
            kps_preview = list(sample.reference)[:3]
            print(
                f"  [{sample.sample_id}] {len(sample.reference)} keyphrases, "
                f"doc={doc_preview}..., "
                f"kps={kps_preview}"
            )

        print("\n" + "=" * 70)
        print("Calling FMAPIClient (databricks-gpt-5-6-sol)...")
        print("=" * 70)

        with FMAPIClient(
            profile="ai_devtools", timeout_seconds=30, max_retries=3
        ) as client:
            parsed_outputs = []

            for sample in samples:
                messages = task.build_prompt(sample)

                print(f"\n[Sample {sample.sample_id}]")
                doc_preview = sample.inputs["document"][:60].replace("\n", " ")
                print(f"  Document: {doc_preview}...")

                # FMAPI 호출 (sol 모델, reasoning 최소화, max_tokens 작음)
                response = client.chat(
                    endpoint="databricks-gpt-5-6-sol",
                    messages=messages,
                    max_tokens=128,
                    extra_params={"reasoning_effort": "none"},
                )

                kp_preview = response.text[:100].replace("\n", " ")
                print(f"  Generated keyphrases: {kp_preview}...")

                # 파싱
                parsed = task.parse_output(response.text, sample)
                parsed_outputs.append(parsed)

                # 즉각 채점
                gold_set = sample.reference
                tp = len(parsed & gold_set)
                fp = len(parsed - gold_set)
                fn = len(gold_set - parsed)
                p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

                print(
                    f"  P={p:.3f}, R={r:.3f}, F1={f1:.3f} "
                    f"(gold={len(gold_set)}, pred={len(parsed)})"
                )

            # 전체 채점
            print("\n" + "=" * 70)
            print("Computing P-R-F1 Scores...")
            print("=" * 70)

            scores = task.score(parsed_outputs, samples)

            print(f"\nMulti-label PRF Metrics:")
            print(f"  Precision (micro): {scores['precision']:.4f}")
            print(f"  Recall (micro):    {scores['recall']:.4f}")
            print(f"  F1 (micro):        {scores['f1']:.4f}")
            print(f"  Samples evaluated: {scores['n_evaluated']}")

            print(f"\nMacro-averaged:")
            print(f"  Precision (macro): {scores['macro_precision']:.4f}")
            print(f"  Recall (macro):    {scores['macro_recall']:.4f}")
            print(f"  F1 (macro):        {scores['macro_f1']:.4f}")

            if scores["per_language"]:
                print(f"\nPer-language breakdown:")
                for lang, lang_scores in scores["per_language"].items():
                    if lang_scores["n_evaluated"] > 0:
                        print(
                            f"  {lang.upper()}:"
                            f" P={lang_scores['precision']:.4f}"
                            f" R={lang_scores['recall']:.4f}"
                            f" F1={lang_scores['f1']:.4f}"
                            f" (n={lang_scores['n_evaluated']})"
                        )

            print("\n" + "=" * 70)
            print("✓ Test completed successfully")
            print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
