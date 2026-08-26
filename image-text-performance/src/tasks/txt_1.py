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

import sys
from typing import Any

from src.adapters.fmapi import build_text_message, FMAPIClient
from src.datasets_loader import load_hf_split, load_registry, resolve_dataset_entry
from src.scoring.metrics import token_f1, exact_match
from src.scoring.judge import load_rubrics, build_judge_prompt, parse_judge_score
from src.tasks.base import Task, Sample, register


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
        registry = load_registry()
        config = self.config

        if "datasets" not in config or "en" not in config["datasets"]:
            raise ValueError("config에 datasets.en (doc_vqa)가 없음")

        dataset_key = config["datasets"]["en"]
        dataset_entry = resolve_dataset_entry(registry, dataset_key)

        hf_id = dataset_entry["hf_id"]
        split = dataset_entry.get("split", "validation")
        config_name = dataset_entry.get("config")

        # HF 데이터셋 로드 (seed 고정)
        hf_ds = load_hf_split(hf_id, split, n, seed, config_name)

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
            max_f1 = 0.0
            max_em = 0.0

            for gold in reference_list:
                # Token-F1 (영어 토크나이제이션)
                f1 = token_f1(pred, gold, "en")
                max_f1 = max(max_f1, f1)

                # Exact match (정규화: strip + lowercase)
                em = exact_match(pred, gold)
                max_em = max(max_em, em)

            token_f1_scores.append(max_f1)
            exact_match_scores.append(max_em)

        return {
            "token_f1": sum(token_f1_scores) / len(token_f1_scores) if token_f1_scores else 0.0,
            "exact_match": sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0.0,
            "n_evaluated": len(parsed),
        }

    def judge_scores(
        self,
        parsed: list[str],
        samples: list[Sample],
        judge_client: FMAPIClient,
        judge_endpoint: str = "databricks-gemini-3-1-pro",
    ) -> dict[str, Any]:
        """LLM 판사를 사용한 정성적 평가.

        각 샘플에 대해 판사 모델을 호출해 1-5 점수를 얻는다.
        """
        if not parsed or not samples:
            return {
                "judge_scores": [],
                "judge_mean": None,
                "n_judged": 0,
                "n_requested": 0,
                "n_unparsed": 0,
                "judge_errors": [],
            }

        # Rubric 로드
        try:
            rubrics = load_rubrics("config/judge_rubrics.yaml")
        except FileNotFoundError:
            rubrics = {}

        # TXT-1용 rubric (없으면 generic QA 사용)
        if "TXT-1" in rubrics:
            rubric = rubrics["TXT-1"]
        else:
            # Fallback: generic QA rubric
            rubric = {
                "name": "Document QA",
                "description": "Extract accurate answers from document text",
                "anchors": {
                    "1": "Answer is unrelated or completely incorrect",
                    "2": "Answer reflects only partial document content or core is distorted",
                    "3": "Answer accurately reflects most of document content but with minor errors or omissions",
                    "4": "Answer is nearly identical to reference with only minor phrasing differences",
                    "5": "Answer is identical or equivalent to reference with perfect accuracy",
                }
            }

        judge_scores: list[int | None] = []
        judge_errors: list[str] = []
        n_unparsed = 0
        for pred, sample in zip(parsed, samples):
            question = sample.inputs["question"]
            context = sample.inputs.get("context", "")
            # reference_list 중 첫 번째를 참고정답으로 사용
            reference = sample.reference[0] if sample.reference else ""

            # Judge prompt 구성
            if context:
                judge_prompt = build_judge_prompt(
                    task_id="TXT-1",
                    question=f"Document text: {context}\n\nQuestion: {question}",
                    reference=reference,
                    candidate=pred,
                    rubric=rubric,
                )
            else:
                judge_prompt = build_judge_prompt(
                    task_id="TXT-1",
                    question=f"Question: {question}",
                    reference=reference,
                    candidate=pred,
                    rubric=rubric,
                )

            try:
                # Judge 호출
                response = judge_client.chat(
                    endpoint=judge_endpoint,
                    messages=build_text_message(judge_prompt),
                    max_tokens=256,
                    extra_params={},
                )

                # 점수 파싱
                score = parse_judge_score(response.text)
                if score is None:
                    n_unparsed += 1
                    print(f"Judge 점수 파싱 실패 (샘플 {sample.sample_id})", file=sys.stderr)
                judge_scores.append(score)
            except Exception as e:
                judge_errors.append(f"sample {sample.sample_id}: {type(e).__name__}: {e}")
                print(
                    f"Judge 호출 실패 (샘플 {sample.sample_id}): {type(e).__name__}: {e}",
                    file=sys.stderr,
                )
                judge_scores.append(None)

        # 판사 호출·파싱 실패는 점수로 대체하지 않고 제외 (평균 왜곡 방지)
        valid = [s for s in judge_scores if s is not None]
        mean_score = sum(valid) / len(valid) if valid else None

        return {
            "judge_scores": judge_scores,
            "judge_mean": mean_score,
            "n_judged": len(valid),
            "n_requested": len(judge_scores),
            "n_unparsed": n_unparsed,
            "judge_errors": judge_errors,
        }


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
                    mean = judge_result["judge_mean"]
                    print(f"평균 점수: {mean:.2f}" if mean is not None else "평균 점수: N/A")
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
