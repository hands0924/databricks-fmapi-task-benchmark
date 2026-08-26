"""TXT-2 표(엑셀) 이해 QA (Table QA).

이 태스크는 WikiTableQuestions 데이터셋을 사용하여 테이블 기반 질의응답을 평가한다.
TXT-2는 텍스트 태스크(is_vision=False)로, 마크다운 형식의 테이블 텍스트와 질문을
입력으로 받아 테이블에서 정보를 추출하여 정답을 생성한다.

WikiTableQuestions 구조:
- question: 테이블에 대한 자연어 질문
- answers: 정답 문자열 리스트 (숫자 또는 텍스트)
- table_md: 마크다운 형식의 테이블
- table: 구조화된 테이블 dict (header, rows)

정확도(숫자/문자 값 매칭, whitespace/case 관대) + Token-F1을 계산.
LLM 판사(judge)를 통한 정성적 평가도 지원한다.
"""

from __future__ import annotations

import sys
from typing import Any

from src.adapters.fmapi import build_text_message, FMAPIClient
from src.datasets_loader import load_hf_split, load_registry, resolve_dataset_entry
from src.scoring.metrics import token_f1
from src.scoring.judge import load_rubrics, build_judge_prompt, parse_judge_score
from src.tasks.base import Task, Sample, register


def _normalize_answer(answer: str) -> str:
    """정답 정규화: 공백 제거, 소문자, 숫자 표준화."""
    answer = answer.strip().lower()
    # 숫자 후행 공백 제거 (e.g., "2004 " → "2004")
    answer = " ".join(answer.split())
    return answer


def _accuracy_match(pred: str, gold_list: list[str]) -> float:
    """정확도: 정규화된 정답과의 부분 일치 (최대값).

    gold_list 중 하나와 정확히 매칭되면 1.0, 아니면 0.0.
    """
    pred_norm = _normalize_answer(pred)

    for gold in gold_list:
        gold_norm = _normalize_answer(gold)
        if pred_norm == gold_norm:
            return 1.0

    return 0.0


@register
class Txt2Task(Task):
    """표(엑셀) 이해 QA 태스크 (TXT-2)."""

    task_id: str = "TXT-2"
    kind: str = "qa"
    is_vision: bool = False

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """WikiTableQuestions 데이터셋에서 seed 고정 subset을 로드.

        WikiTableQuestions 구조:
        - question: 자연어 질문
        - answers: 정답 리스트 (str)
        - table_md: 마크다운 형식 테이블
        - table: 구조화된 테이블 (header, rows)

        마크다운 테이블을 입력으로 사용.
        """
        registry = load_registry()
        config = self.config

        if "datasets" not in config or "en" not in config["datasets"]:
            raise ValueError("config에 datasets.en (table_qa)가 없음")

        dataset_key = config["datasets"]["en"]
        dataset_entry = resolve_dataset_entry(registry, dataset_key)

        hf_id = dataset_entry["hf_id"]
        split = dataset_entry.get("split", "train")
        config_name = dataset_entry.get("config")

        # split이 "default"이면 "train"으로 치환 (일부 HF 데이터셋 호환성)
        if split == "default":
            split = "train"

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

            # 테이블 텍스트 추출: table_md (마크다운) 우선, 없으면 table dict 변환
            table_text = row.get("table_md", "")

            if not table_text:
                # table dict를 마크다운으로 변환
                table_dict = row.get("table", {})
                if table_dict:
                    table_text = self._table_dict_to_markdown(table_dict)

            sample = Sample(
                sample_id=sample_id,
                inputs={
                    "table": table_text,
                    "question": question,
                },
                reference=answers_list,  # 여러 정답 리스트
                lang="en",
                meta={
                    "dataset": dataset_key,
                    "source_id": row.get("id", ""),
                },
            )
            samples.append(sample)

        return samples

    def _table_dict_to_markdown(self, table_dict: dict[str, Any]) -> str:
        """구조화된 테이블 dict를 마크다운 문자열로 변환.

        table_dict 구조:
        {
            "header": ["col1", "col2", ...],
            "rows": [["val1", "val2", ...], ...],
            "name": "table_name" (optional)
        }
        """
        if not table_dict:
            return ""

        header = table_dict.get("header", [])
        rows = table_dict.get("rows", [])

        if not header:
            return ""

        # 마크다운 테이블 구성
        md_lines = []

        # 헤더 행
        md_lines.append("| " + " | ".join(str(h) for h in header) + " |")

        # 구분자 행
        md_lines.append("| " + " | ".join(["---"] * len(header)) + " |")

        # 데이터 행
        for row in rows:
            # row가 리스트 또는 dict일 수 있음
            if isinstance(row, dict):
                values = [str(row.get(h, "")) for h in header]
            else:
                values = [str(v) for v in row]

            # 컬럼 개수 맞추기
            while len(values) < len(header):
                values.append("")

            md_lines.append("| " + " | ".join(values[: len(header)]) + " |")

        return "\n".join(md_lines)

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """테이블 QA 프롬프트 구성.

        테이블 마크다운과 질문을 제시하고, 정확한 셀값 추출을 요청한다.
        """
        table = sample.inputs["table"]
        question = sample.inputs["question"]

        prompt = f"""Based on the following table, answer the question accurately.

Table:
{table}

Question: {question}

Answer:"""

        return build_text_message(prompt)

    def parse_output(self, raw_text: str, sample: Sample) -> str:
        """모델 응답을 정답으로 파싱.

        Token-F1과 정확도 계산을 위해 원문 그대로 반환 (정규화 없음).
        """
        return raw_text.strip()

    def score(self, parsed: list[str], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 예측 결과를 집계해 정확도와 Token-F1 계산.

        각 샘플별로 여러 정답 중 최고 점수를 취한다.
        """
        if not parsed or not samples:
            return {
                "accuracy": 0.0,
                "token_f1": 0.0,
                "n_evaluated": 0,
            }

        accuracy_scores = []
        token_f1_scores = []

        for pred, sample in zip(parsed, samples):
            if not pred:
                # 빈 예측
                accuracy_scores.append(0.0)
                token_f1_scores.append(0.0)
                continue

            # 여러 정답 중 최고 점수
            reference_list = sample.reference  # list[str]

            # 정확도: 값 매칭 (관대한 정규화)
            max_accuracy = _accuracy_match(pred, reference_list)
            accuracy_scores.append(max_accuracy)

            # Token-F1: 모든 정답 중 최고값
            max_f1 = 0.0
            for gold in reference_list:
                f1 = token_f1(pred, gold, "en")
                max_f1 = max(max_f1, f1)

            token_f1_scores.append(max_f1)

        return {
            "accuracy": sum(accuracy_scores) / len(accuracy_scores) if accuracy_scores else 0.0,
            "token_f1": sum(token_f1_scores) / len(token_f1_scores) if token_f1_scores else 0.0,
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

        # TXT-2용 rubric (없으면 generic QA 사용)
        if "TXT-2" in rubrics:
            rubric = rubrics["TXT-2"]
        else:
            # Fallback: generic table QA rubric
            rubric = {
                "name": "Table QA",
                "description": "Extract accurate cell values from tables",
                "anchors": {
                    "1": "Extracted value does not match table or is completely wrong",
                    "2": "Extracted value partially matches with errors in key values",
                    "3": "Extracted value mostly correct but with minor cell errors or format issues",
                    "4": "Extracted value nearly accurate with only minor format differences",
                    "5": "Extracted value is accurate and perfectly matches reference cell values",
                }
            }

        judge_scores: list[int | None] = []
        judge_errors: list[str] = []
        n_unparsed = 0
        for pred, sample in zip(parsed, samples):
            question = sample.inputs["question"]
            table = sample.inputs["table"]
            # reference_list 중 첫 번째를 참고정답으로 사용
            reference = sample.reference[0] if sample.reference else ""

            # Judge prompt 구성
            judge_prompt = build_judge_prompt(
                task_id="TXT-2",
                question=f"Table:\n{table}\n\nQuestion: {question}",
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
    """End-to-end 테스트: 3개 WikiTableQuestions 샘플로 accuracy, token_f1, judge 점수 계산."""
    import sys

    # 테스트용 태스크 config
    test_config = {
        "datasets": {
            "en": "table_qa",
        }
    }

    registry = load_registry()
    task = Txt2Task(test_config, registry)

    # 3샘플 로드
    print("=" * 70)
    print("TXT-2 Table QA Task - End-to-End Test")
    print("=" * 70)
    print("Loading 3 WikiTableQuestions samples...")
    samples = task.load_samples(n=3, seed=42)
    print(f"✓ Loaded {len(samples)} samples\n")

    for sample in samples:
        q_preview = sample.inputs["question"][:60]
        table_preview = sample.inputs["table"][:80].replace("\n", " ")
        ref_preview = sample.reference[0][:40] if sample.reference else "(no ref)"
        print(f"[샘플 {sample.sample_id}]")
        print(f"  질문: {q_preview}...")
        print(f"  테이블: {table_preview}...")
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

            # 수치 채점 (Accuracy, Token-F1)
            print("\n" + "=" * 70)
            print("Computing Accuracy and Token-F1...")
            print("=" * 70)
            scores = task.score(parsed_outputs, samples)

            print(f"\n=== 수치 점수 ===")
            print(f"Accuracy: {scores['accuracy']:.4f}")
            print(f"Token-F1: {scores['token_f1']:.4f}")
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

                    print("\n=== Judge 점수 (1-5) ===")
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
