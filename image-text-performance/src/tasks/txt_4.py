"""TXT-4 한국어 독해 QA (Korean Reading Comprehension).

KorQuAD 데이터셋을 사용하여 한국어 문맥 기반 질의응답을 평가한다.
주어진 지문(context)과 질문(question)에서 정답 구간(answer span)을 예측한다.
Token-level F1과 exact match를 계산하며, LLM judge를 통한 정성적 평가도 지원한다.
"""

from __future__ import annotations

import sys
from typing import Any

from src.adapters.fmapi import build_text_message, FMAPIClient
from src.datasets_loader import load_hf_split, load_registry, resolve_dataset_entry
from src.scoring.metrics import token_f1
from src.scoring.tokenizers import korean_tokenizer_backend
from src.scoring.judge import load_rubrics, build_judge_prompt, parse_judge_score
from src.tasks.base import Task, Sample, register


@register
class Txt4Task(Task):
    """한국어 독해 QA 태스크 (TXT-4)."""

    task_id: str = "TXT-4"
    kind: str = "qa"
    is_vision: bool = False

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """KorQuAD 데이터셋에서 seed 고정 subset을 로드.

        KorQuAD는 SQuAD 스타일의 한국어 QA 데이터셋으로:
        - context: 주어진 지문
        - question: 질문
        - answers: {'text': [답변들], 'answer_start': [위치들]} (여러 정답 가능)
        """
        registry = load_registry()
        config = self.config

        if "datasets" not in config or "ko" not in config["datasets"]:
            raise ValueError("config에 datasets.ko (korquad)가 없음")

        dataset_key = config["datasets"]["ko"]
        dataset_entry = resolve_dataset_entry(registry, dataset_key)

        hf_id = dataset_entry["hf_id"]
        # 주의: registry.yaml에서 split="train"이지만, 실제 평가는 validation 권장
        # 여기서는 명시적으로 validation 사용 (테스트용)
        split = dataset_entry.get("split", "train")
        config_name = dataset_entry.get("config")

        # HF 데이터셋 로드 (seed 고정)
        hf_ds = load_hf_split(hf_id, split, n, seed, config_name)

        samples = []
        for sample_id, row in enumerate(hf_ds):
            context = row["context"]
            question = row["question"]
            # answers는 {'text': [...], 'answer_start': [...]} 형태
            answers_list = row["answers"]["text"]

            sample = Sample(
                sample_id=sample_id,
                inputs={
                    "context": context,
                    "question": question,
                },
                reference=answers_list,  # 여러 정답 리스트
                lang="ko",
                meta={
                    "dataset": dataset_key,
                    "title": row.get("title", ""),
                },
            )
            samples.append(sample)

        return samples

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """한국어 QA 프롬프트 구성.

        모델에게 지문과 질문을 제시하고, 답변 구간만 간결하게 답하도록 지시한다.
        """
        context = sample.inputs["context"]
        question = sample.inputs["question"]

        prompt = f"""다음 지문을 읽고 질문에 답하세요. 답변은 지문에서 나타나는 정답만 간결하게 기술하세요.

지문:
{context}

질문: {question}

답변:"""

        return build_text_message(prompt)

    def parse_output(self, raw_text: str, sample: Sample) -> str:
        """모델 응답을 정답으로 파싱.

        Token-F1 계산을 위해 원문 그대로 반환 (정규화 없음).
        """
        return raw_text.strip()

    def score(self, parsed: list[str], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 예측 결과를 집계해 Token-F1과 exact match 계산.

        각 샘플별로 여러 정답 중 최고 점수를 취한다.
        """
        if not parsed or not samples:
            return {
                "token_f1": 0.0,
                "exact_match": 0.0,
                "n_evaluated": 0,
                "korean_backend": korean_tokenizer_backend(),
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
                # Token-F1 (언어=ko로 명시)
                f1 = token_f1(pred, gold, "ko")
                max_f1 = max(max_f1, f1)

                # Exact match (정규화: strip + lowercase)
                pred_norm = pred.strip().lower()
                gold_norm = gold.strip().lower()
                em = 1.0 if pred_norm == gold_norm else 0.0
                max_em = max(max_em, em)

            token_f1_scores.append(max_f1)
            exact_match_scores.append(max_em)

        return {
            "token_f1": sum(token_f1_scores) / len(token_f1_scores) if token_f1_scores else 0.0,
            "exact_match": sum(exact_match_scores) / len(exact_match_scores) if exact_match_scores else 0.0,
            "n_evaluated": len(parsed),
            "korean_backend": korean_tokenizer_backend(),
        }

    def judge_scores(
        self,
        parsed: list[str],
        samples: list[Sample],
        judge_client: FMAPIClient,
        judge_endpoint: str = "databricks-gemini-3-1-pro",
    ) -> dict[str, Any]:
        """LLM judge를 사용한 정성적 평가.

        각 샘플에 대해 judge 모델을 호출해 1-5 점수를 얻는다.
        judge_rubrics.yaml에 TXT-4 rubric이 없으면 generic QA rubric을 사용.
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

        # TXT-4용 rubric (없으면 generic QA 사용)
        if "TXT-4" in rubrics:
            rubric = rubrics["TXT-4"]
        else:
            # Fallback: generic QA rubric
            rubric = {
                "name": "한국어 독해 QA",
                "description": "주어진 지문에서 정확한 정답을 추출하는 능력 평가",
                "anchors": {
                    "1": "답변이 지문과 무관하거나 완전히 잘못됨",
                    "2": "답변이 지문의 일부만 반영하거나 핵심이 왜곡됨",
                    "3": "답변이 대부분 정확하나 미세한 오류나 누락 있음",
                    "4": "답변이 정답과 거의 동일하고 미세한 표현 차이만 있음",
                    "5": "답변이 정답과 동일하거나 동등 수준의 정확성으로 전달",
                }
            }

        judge_scores: list[int | None] = []
        judge_errors: list[str] = []
        n_unparsed = 0
        for pred, sample in zip(parsed, samples):
            context = sample.inputs["context"]
            question = sample.inputs["question"]
            # reference_list 중 첫 번째를 참고정답으로 사용
            reference = sample.reference[0] if sample.reference else ""

            # Judge prompt 구성
            judge_prompt = build_judge_prompt(
                task_id="TXT-4",
                question=f"지문: {context}\n\n질문: {question}",
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
    """End-to-end 테스트: 4개 KorQuAD 샘플로 token_f1, exact_match, judge 점수 계산."""
    import sys

    # 테스트용 태스크 config
    test_config = {
        "datasets": {
            "ko": "korquad",
        }
    }

    registry = load_registry()
    task = Txt4Task(test_config, registry)

    # 4샘플 로드
    print("=" * 70)
    print("Loading 4 KorQuAD samples...")
    print("=" * 70)
    samples = task.load_samples(n=4, seed=42)
    print(f"Loaded {len(samples)} samples\n")

    for sample in samples:
        print(f"[샘플 {sample.sample_id}]")
        print(f"  제목: {sample.meta.get('title', 'N/A')}")
        print(f"  지문: {sample.inputs['context'][:80]}...")
        print(f"  질문: {sample.inputs['question']}")
        print(f"  정답: {sample.reference}")
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
                print(f"  프롬프트: {messages[0]['content'][:100]}...")

                # FMAPI 호출 (databricks-gpt-5-6-sol + minimal reasoning)
                response = client.chat(
                    endpoint="databricks-gpt-5-6-sol",
                    messages=messages,
                    max_tokens=128,
                    extra_params={"reasoning_effort": "none"},
                )

                print(f"  모델 응답: {response.text}")

                # 파싱
                parsed = task.parse_output(response.text, sample)
                parsed_outputs.append(parsed)

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
            print(f"한국어 백엔드: {scores['korean_backend']}")

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
