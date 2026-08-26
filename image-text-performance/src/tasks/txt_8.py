"""TXT-8 비속어/혐오성 텍스트 검출 (한/영 병행).

이 태스크는 영어와 한국어 텍스트의 혐오성/비속어 이진 분류를 구현한다.
- 영어: Jigsaw 독성 댓글 분류 데이터셋 (toxic=1, clean=0)
- 한국어: APEACH 한국어 혐오성 텍스트 데이터셋 (class=1: 혐오성, class=0: 정상)

부분 언어별 샘플 분할을 통해 양 언어를 고르게 평가한다.
"""

from __future__ import annotations

from typing import Any

from src.adapters.fmapi import build_text_message
from src.datasets_loader import load_registry
from src.tasks.base import Task, Sample, register
from src.tasks.common import (
    binary_score_summary,
    detect_column,
    language_batches,
    parse_binary_label,
)

LABEL_COLUMNS = (
    "toxic",
    "class",
    "severe_toxic",
    "obscene",
    "threat",
    "insult",
    "identity_hate",
    "id",
)


@register
class Txt8Task(Task):
    """비속어/혐오성 텍스트 이진 분류 태스크 (TXT-8)."""

    task_id: str = "TXT-8"
    kind: str = "binary"
    is_vision: bool = False

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """영어/한국어 혐오성 데이터셋에서 seed 고정 subset을 로드.

        n개 샘플을 언어별로 균등 분할: 약 n//2씩 영어와 한국어.
        각 언어별 컬럼명(comment_text/text, toxic/class)을 자동 감지하고
        binary 0/1 레이블로 정규화한다.
        """
        samples = []
        sample_id = 0

        for batch in language_batches(self.config, n, seed):
            hf_ds = batch.rows
            lang = batch.lang

            # 컬럼명 자동 감지
            col_text = self._detect_text_column(hf_ds, lang)
            col_label = self._detect_label_column(hf_ds, lang)

            for idx, row in enumerate(hf_ds):
                text = row[col_text]
                label_raw = row[col_label]

                # None 또는 NaN 값 스킵
                if label_raw is None or text is None:
                    continue

                # binary 정규화: toxic=1, clean=0
                label_int = int(label_raw)

                sample = Sample(
                    sample_id=sample_id,
                    inputs={"text": text},
                    reference=label_int,
                    lang=lang,
                    meta={
                        "dataset": batch.dataset_key,
                        "source_idx": idx,
                    },
                )
                samples.append(sample)
                sample_id += 1

        return samples

    def _detect_text_column(self, hf_ds: list[dict[str, Any]], lang: str) -> str:
        """HF 데이터셋에서 텍스트 컬럼명을 자동 감지.

        영어: 'comment_text' (Jigsaw dataset)
        한국어: 'text' (APEACH dataset)
        """
        priority_cols = {
            "en": ["comment_text", "text"],
            "ko": ["text", "comment_text"],
        }
        return detect_column(
            hf_ds,
            priority_cols.get(lang, ["text", "comment_text"]),
            exclude=LABEL_COLUMNS,
            what="텍스트 컬럼",
        )

    def _detect_label_column(self, hf_ds: list[dict[str, Any]], lang: str) -> str:
        """HF 데이터셋에서 레이블 컬럼명을 자동 감지.

        영어: 'toxic' (Jigsaw dataset)
        한국어: 'class' (APEACH dataset)
        """
        priority_cols = {
            "en": ["toxic", "class"],
            "ko": ["class", "toxic"],
        }
        return detect_column(
            hf_ds,
            priority_cols.get(lang, ["toxic", "class"]),
            what="레이블 컬럼",
        )

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """명확한 혐오성/비속어 검출 프롬프트 구성.

        모델에게 "toxic" 또는 "clean" 중 정확히 하나의 단어로
        응답하도록 요청한다. 한국어 샘플도 영어 지시로 통일.
        이는 임상적·중립적인 콘텐츠 관리 벤치마크 프롬프트이다.
        """
        text = sample.inputs["text"]

        prompt = f"""You are a content moderation classifier. Analyze the following text and determine if it contains profanity, hate speech, or toxic language. Respond with exactly one word: "toxic" or "clean".

Text: {text}

Respond with exactly one word: toxic or clean"""

        return build_text_message(prompt)

    def parse_output(self, raw_text: str, sample: Sample) -> int | None:
        """모델 응답을 binary 라벨로 파싱.

        "toxic"→1, "clean"→0으로 매핑.
        대소문자 무시, 여러 단어 포함 시 첫 단어만 추출.
        한국어 응답("유해"/"정상") 처리.
        파싱 불가 시 None 반환.
        """
        return parse_binary_label(
            raw_text,
            positive=("toxic", "유해"),
            negative=("clean", "정상"),
        )

    def score(self, parsed: list[int | None], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 예측 결과를 집계해 메트릭 계산.

        - None 예측값 제외 (unparseable)
        - binary_metrics로 accuracy/f1 계산
        - 혼동 행렬(confusion matrix) 포함
        - 언어별 분석 및 평가 샘플 수 포함
        """
        return binary_score_summary(parsed, samples, per_language=True)


if __name__ == "__main__":
    """간단한 인라인 테스트: 6샘플(영어 3, 한국어 3)로 end-to-end 실행."""
    import sys

    from src.adapters.fmapi import FMAPIClient

    # 테스트용 작은 태스크 config
    test_config = {
        "datasets": {
            "en": "toxicity_en",
            "ko": "toxicity_ko",
        }
    }

    registry = load_registry()
    task = Txt8Task(test_config, registry)

    # 6샘플 로드 (영어 3, 한국어 3)
    print("Loading 6 samples (3 en, 3 ko)...")
    samples = task.load_samples(n=6, seed=42)
    print(f"Loaded {len(samples)} samples")

    for sample in samples:
        text_preview = sample.inputs["text"][:60].replace("\n", " ")
        print(f"  [{sample.sample_id}] {sample.lang}: {text_preview}... (ref={sample.reference})")

    # 프롬프트 생성 및 모델 호출
    print("\nBuilding prompts and calling FMAPIClient...")

    try:
        with FMAPIClient(profile="ai_devtools", timeout_seconds=30) as client:
            parsed_outputs = []

            for sample in samples:
                messages = task.build_prompt(sample)

                text_preview = sample.inputs["text"][:50].replace("\n", " ")
                print(f"\n  Sample {sample.sample_id} ({sample.lang}):")
                print(f"    Text: {text_preview}...")

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
            print(f"F1: {scores['f1']:.3f}")
            print(f"Evaluated: {scores['n_evaluated']}/{len(samples)}")
            print(f"Unparsed: {scores['n_unparsed']}")
            print(f"Confusion matrix: {scores['confusion_matrix']}")

            print(f"\nPer-language breakdown:")
            for lang, stats in scores["per_language"].items():
                print(f"  {lang}:")
                print(f"    Evaluated: {stats['n_evaluated']}")
                print(f"    Accuracy: {stats['accuracy']}")
                print(f"    F1: {stats['f1']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
