"""TXT-1 문서(PDF) 이해 QA (Document QA).

이 태스크는 DocumentVQA 데이터셋을 사용하여 문서 이미지 기반 질의응답을 평가한다.
TXT-1은 텍스트 태스크(is_vision=False)이므로, 이미지는 사용하지 않고,
대신 문서의 OCR 텍스트 컨텍스트(있으면)를 활용하여 텍스트 기반 QA 프롬프트를 구성한다.

실제 DocumentVQA 데이터셋에서:
- question: 문서에 대한 질문
- answers: 정답 문자열 리스트
- image: PIL Image (사용 안 함, is_vision=False)
- 텍스트 신호: 'words'/'ocr' 필드가 있으면 문서의 OCR 텍스트 이용, 없으면 질문만으로 평가

Token-level F1과 정확도 매칭을 계산하며, LLM 판사(judge)를 통한 정성적 평가도 지원한다.
"""

from __future__ import annotations

from typing import Any, ClassVar

from src.adapters.fmapi import FMAPIClient, build_text_message
from src.datasets_loader import load_registry
from src.scoring.metrics import token_f1, exact_match
from src.tasks.base import Task, Sample, register
from src.tasks.common import best_reference_score, mean, single_language_rows


@register
class Txt1Task(Task):
    """문서(PDF) 이해 QA 태스크 (TXT-1)."""

    task_id: str = "TXT-1"
    kind: str = "qa"
    is_vision: bool = False

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """DocumentVQA 데이터셋에서 seed 고정 subset을 로드.

        DocumentVQA는 다음 구조:
        - question: 문서에 대한 질문 (str)
        - answers: 정답 리스트 (list[str])
        - words: OCR 토큰 리스트 (dict 또는 list, 선택적)
        - image: PIL Image (is_vision=False이므로 사용 안 함)

        텍스트 신호: 'words' 필드가 있으면 OCR 텍스트로 활용, 없으면 질문만으로 평가.
        """
        dataset_key, hf_ds = single_language_rows(
            self.config,
            "en",
            n,
            seed,
            default_split="validation",
            dataset_hint="doc_vqa",
        )

        samples = []
        for sample_id, row in enumerate(hf_ds):
            question = row.get("question", "")
            answers_raw = row.get("answers", [])

            # answers 정규화: 문자열 리스트로
            if isinstance(answers_raw, list):
                answers_list = [str(a) for a in answers_raw]
            else:
                answers_list = [str(answers_raw)]

            # OCR 텍스트 추출 시도
            # DocumentVQA 구조: 'words'는 dict {coord: token} 또는 list of tokens
            words_raw = row.get("words", None)
            ocr_context = None

            if words_raw:
                if isinstance(words_raw, dict):
                    # {coord: token, ...} 형태 → 토큰 추출
                    ocr_context = " ".join(words_raw.values())
                elif isinstance(words_raw, list):
                    # [token, ...] 형태
                    ocr_context = " ".join(str(w) for w in words_raw if w)

            sample = Sample(
                sample_id=sample_id,
                inputs={
                    "question": question,
                    "context": ocr_context,  # OCR 텍스트 또는 None
                },
                reference=answers_list,  # 여러 정답 리스트
                lang="en",
                meta={
                    "dataset": dataset_key,
                    "has_ocr": ocr_context is not None,
                },
            )
            samples.append(sample)

        return samples

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """문서 QA 프롬프트 구성.

        OCR 텍스트가 있으면 문맥으로 함께 제시, 없으면 질문만 제시.
        모델이 문서 텍스트에서 정답을 추출하도록 지시.
        """
        question = sample.inputs["question"]
        context = sample.inputs["context"]

        if context:
            # OCR 텍스트가 있는 경우
            prompt = f"""Based on the following document text, answer the question accurately and concisely.

Document text:
{context}

Question: {question}

Answer:"""
        else:
            # OCR 텍스트가 없는 경우 (텍스트 신호 부재)
            prompt = f"""Answer the following question:

Question: {question}

Answer:"""

        return build_text_message(prompt)

    def parse_output(self, raw_text: str, sample: Sample) -> str:
        """모델 응답을 정답으로 파싱.

        Token-F1과 정확도 계산을 위해 원문 그대로 반환 (정규화 없음).
        """
        return raw_text.strip()

    def score(self, parsed: list[str], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 예측 결과를 집계해 Token-F1과 정확도 계산.

        각 샘플별로 여러 정답 중 최고 점수를 취한다.
        """
        if not parsed or not samples:
            return {
                "token_f1": 0.0,
                "exact_match": 0.0,
                "n_evaluated": 0,
            }

        token_f1_scores = []
        exact_match_scores = []

        for pred, sample in zip(parsed, samples):
            if not pred:
                # 빈 예측
                token_f1_scores.append(0.0)
                exact_match_scores.append(0.0)
                continue

            # 여러 정답 중 최고 점수
            reference_list = sample.reference  # list[str]
            token_f1_scores.append(
                best_reference_score(pred, reference_list, lambda p, g: token_f1(p, g, "en"))
            )
            exact_match_scores.append(best_reference_score(pred, reference_list, exact_match))

        return {
            "token_f1": mean(token_f1_scores),
            "exact_match": mean(exact_match_scores),
            "n_evaluated": len(parsed),
        }

    # judge_scores는 Task 기본 구현 사용 (질문/루브릭만 태스크별로 지정)
    judge_rubric_fallback: ClassVar[dict[str, Any]] = {
        "name": "Document QA",
        "description": "Extract accurate answers from document text",
        "anchors": {
            "1": "Answer is unrelated or completely incorrect",
            "2": "Answer reflects only partial document content or core is distorted",
            "3": "Answer accurately reflects most of document content but with minor errors or omissions",
            "4": "Answer is nearly identical to reference with only minor phrasing differences",
            "5": "Answer is identical or equivalent to reference with perfect accuracy",
        },
    }

    def judge_question(self, sample: Sample) -> str:
        """OCR 문맥이 있으면 문서 텍스트와 질문을 함께 제시."""
        question = sample.inputs["question"]
        context = sample.inputs.get("context", "")

        if context:
            return f"Document text: {context}\n\nQuestion: {question}"
        return f"Question: {question}"


if __name__ == "__main__":
    """End-to-end 테스트: 3개 DocumentVQA 샘플로 token_f1, exact_match, judge 점수 계산."""
    import sys

    # 테스트용 태스크 config
    test_config = {
        "datasets": {
            "en": "doc_vqa",
        }
    }

    registry = load_registry()
    task = Txt1Task(test_config, registry)

    # 3샘플 로드
    print("=" * 70)
    print("TXT-1 Document QA Task - End-to-End Test")
    print("=" * 70)
    print("Loading 3 DocumentVQA samples...")
    samples = task.load_samples(n=3, seed=42)
    print(f"✓ Loaded {len(samples)} samples\n")

    for sample in samples:
        q_preview = sample.inputs["question"][:60]
        ocr_preview = (
            (sample.inputs["context"][:60] if sample.inputs["context"] else "(no OCR)")
            .replace("\n", " ")
        )
        ref_preview = sample.reference[0][:40] if sample.reference else "(no ref)"
        print(f"[샘플 {sample.sample_id}]")
        print(f"  질문: {q_preview}...")
        print(f"  OCR 텍스트: {ocr_preview}...")
        print(f"  정답(첫번째): {ref_preview}...")
        print()

    # 프롬프트 생성 및 모델 호출
    print("=" * 70)
    print("Calling FMAPIClient (databricks-gpt-5-6-sol)...")
    print("=" * 70)

    try:
        with FMAPIClient(profile="ai_devtools", timeout_seconds=30) as client:
            parsed_outputs = []

            for sample in samples:
                messages = task.build_prompt(sample)

                print(f"\n[샘플 {sample.sample_id}]")
                print(f"  질문: {sample.inputs['question'][:50]}...")

                # FMAPI 호출 (databricks-gpt-5-6-sol + minimal reasoning)
                response = client.chat(
                    endpoint="databricks-gpt-5-6-sol",
                    messages=messages,
                    max_tokens=128,
                    extra_params={"reasoning_effort": "none"},
                )

                print(f"  모델 응답: {response.text[:80]}...")

                # 파싱
                parsed = task.parse_output(response.text, sample)
                parsed_outputs.append(parsed)

                if sample.reference:
                    print(f"  정답(첫번째): {sample.reference[0]}")
                print(f"  파싱된 예측: {parsed}")

            # 수치 채점 (Token-F1, Exact Match)
            print("\n" + "=" * 70)
            print("Computing Token-F1 and Exact Match...")
            print("=" * 70)
            scores = task.score(parsed_outputs, samples)

            print(f"\n=== 수치 점수 ===")
            print(f"Token-F1: {scores['token_f1']:.4f}")
            print(f"Exact Match: {scores['exact_match']:.4f}")
            print(f"평가 샘플 수: {scores['n_evaluated']}")

            # Judge 호출
            print("\n" + "=" * 70)
            print("Calling Judge (databricks-gemini-3-1-pro)...")
            print("=" * 70)

            try:
                with FMAPIClient(profile="ai_devtools", timeout_seconds=60) as judge_client:
                    judge_result = task.judge_scores(
                        parsed_outputs,
                        samples,
                        judge_client,
                        judge_endpoint="databricks-gemini-3-1-pro",
                    )

                    print(f"\n=== Judge 점수 (1-5) ===")
                    print(f"개별 점수: {judge_result['judge_scores']}")
                    print(f"평균 점수: {judge_result['judge_mean']:.2f}")
                    print(f"평가 샘플 수: {judge_result['n_judged']}")
            except Exception as e:
                print(f"Judge 호출 실패: {e}")

            print("\n" + "=" * 70)
            print("테스트 완료")
            print("=" * 70)

    except Exception as e:
        print(f"오류 발생: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
