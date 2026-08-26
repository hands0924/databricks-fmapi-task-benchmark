"""TXT-6 감정 분석 (한/영 병행).

이 태스크는 영어와 한국어 리뷰 텍스트의 감정 분석(negative=0, positive=1)을
구현한다. SST-2(영어)와 NSMC(한국어) 데이터셋을 사용하며, 부분 언어별 분할
샘플링을 통해 양 언어를 고르게 평가한다.
"""

from __future__ import annotations

from typing import Any

from src.adapters.fmapi import build_text_message
from src.datasets_loader import load_registry
from src.scoring.metrics import classification_metrics
from src.tasks.base import Task, Sample, register
from src.tasks.common import (
    detect_column,
    language_batches,
    parse_binary_label,
    per_language_stats,
    preds_and_golds,
    valid_indices_of,
)


@register
class Txt6Task(Task):
    """감정 분석 분류 태스크 (TXT-6)."""

    task_id: str = "TXT-6"
    kind: str = "classification"
    is_vision: bool = False

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """영어/한국어 감정 데이터셋에서 seed 고정 subset을 로드.

        n개 샘플을 언어별로 균등 분할: 약 n//2씩 영어와 한국어.
        각 언어별 컬럼명(sentence/document, label)을 자동 감지하고
        binary 0/1 레이블로 정규화한다.
        """
        samples = []
        sample_id = 0

        for batch in language_batches(self.config, n, seed):
            hf_ds = batch.rows

            # 컬럼명 자동 감지: sentiment_en(sentence), sentiment_ko(document)
            col_text = detect_column(
                hf_ds,
                ["sentence", "document", "text"],
                exclude=["label"],
                what="텍스트 컬럼",
            )
            col_label = "label"  # 둘 다 "label"

            for idx, row in enumerate(hf_ds):
                text = row[col_text]
                label_raw = row[col_label]

                # binary 정규화: negative=0, positive=1
                # SST-2: label 0/1이지만 validation에는 -1이 없으므로 그대로 사용
                # NSMC: label 0/1
                label_int = int(label_raw)

                sample = Sample(
                    sample_id=sample_id,
                    inputs={"text": text},
                    reference=label_int,
                    lang=batch.lang,
                    meta={
                        "dataset": batch.dataset_key,
                        "source_idx": idx,
                    },
                )
                samples.append(sample)
                sample_id += 1

        return samples

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """명확한 감정 분류 프롬프트 구성.

        모델에게 "positive" 또는 "negative" 중 정확히 하나의 단어로
        응답하도록 요청한다. 한국어 샘플도 영어 지시로 통일.
        """
        text = sample.inputs["text"]

        prompt = f"""Classify the sentiment of the following text as exactly one word: "positive" or "negative".

Text: {text}

Respond with exactly one word: positive or negative"""

        return build_text_message(prompt)

    def parse_output(self, raw_text: str, sample: Sample) -> int | None:
        """모델 응답을 binary 라벨로 파싱.

        "positive"→1, "negative"→0으로 매핑.
        대소문자 무시, 여러 단어 포함 시 첫 단어만 추출.
        한국어 응답("긍정"/"부정") 처리.
        파싱 불가 시 None 반환.
        """
        return parse_binary_label(
            raw_text,
            positive=("positive", "긍정"),
            negative=("negative", "부정"),
            numeric_positive=r"\b1\b|yes|true",
            numeric_negative=r"\b0\b|no|false",
        )

    def score(self, parsed: list[int | None], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 예측 결과를 집계해 메트릭 계산.

        - None 예측값 제외 (unparseable)
        - classification_metrics로 accuracy/macro_f1 계산
        - 언어별 분석 및 평가 샘플 수 포함
        """
        # None값 필터링
        valid_indices = valid_indices_of(parsed)

        if not valid_indices:
            return {
                "accuracy": 0.0,
                "macro_f1": 0.0,
                "n_evaluated": 0,
                "n_unparsed": len(parsed),
                "per_language": {},
            }

        preds_valid, golds_valid = preds_and_golds(parsed, samples, valid_indices)

        # 메트릭 계산
        metrics = classification_metrics(preds_valid, golds_valid)

        # 언어별 통계
        lang_stats = per_language_stats(
            parsed,
            samples,
            metric_fn=classification_metrics,
            metric_keys=("accuracy",),
        )

        return {
            "accuracy": metrics["accuracy"],
            "macro_f1": metrics["macro_f1"],
            "n_evaluated": len(valid_indices),
            "n_unparsed": len(parsed) - len(valid_indices),
            "per_language": lang_stats,
        }


if __name__ == "__main__":
    """간단한 인라인 테스트: 6샘플(영어 3, 한국어 3)로 end-to-end 실행."""
    import sys

    from src.adapters.fmapi import FMAPIClient

    # 테스트용 작은 태스크 config
    test_config = {
        "datasets": {
            "en": "sentiment_en",
            "ko": "sentiment_ko",
        }
    }

    registry = load_registry()
    task = Txt6Task(test_config, registry)

    # 6샘플 로드 (영어 3, 한국어 3)
    print("Loading 6 samples (3 en, 3 ko)...")
    samples = task.load_samples(n=6, seed=42)
    print(f"Loaded {len(samples)} samples")

    for sample in samples:
        print(f"  [{sample.sample_id}] {sample.lang}: {sample.inputs['text'][:60]}... (ref={sample.reference})")

    # 프롬프트 생성 및 모델 호출
    print("\nBuilding prompts and calling FMAPIClient...")

    try:
        with FMAPIClient(profile="ai_devtools", timeout_seconds=30) as client:
            parsed_outputs = []

            for sample in samples:
                messages = task.build_prompt(sample)

                print(f"\n  Sample {sample.sample_id} ({sample.lang}):")
                print(f"    Text: {sample.inputs['text'][:50]}...")

                # FMAPI 호출
                response = client.chat(
                    endpoint="databricks-gpt-5-6-sol",
                    messages=messages,
                    max_tokens=64,
                    extra_params={"reasoning_effort": "none"},
                )

                print(f"    Raw response: {response.text}")

                # 파싱
                parsed = task.parse_output(response.text, sample)
                parsed_outputs.append(parsed)

                print(f"    Parsed: {parsed} (expected: {sample.reference})")

            # 채점
            print("\n\nScoring...")
            scores = task.score(parsed_outputs, samples)

            print(f"\n=== Results ===")
            print(f"Accuracy: {scores['accuracy']:.3f}")
            print(f"Macro F1: {scores['macro_f1']:.3f}")
            print(f"Evaluated: {scores['n_evaluated']}/{len(samples)}")
            print(f"Unparsed: {scores['n_unparsed']}")

            print(f"\nPer-language breakdown:")
            for lang, stats in scores["per_language"].items():
                print(f"  {lang}:")
                print(f"    Evaluated: {stats['n_evaluated']}")
                print(f"    Accuracy: {stats['accuracy']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
