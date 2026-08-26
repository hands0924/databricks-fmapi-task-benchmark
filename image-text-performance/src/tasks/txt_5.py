"""TXT-5 텍스트 요약 (한/영 병행).

이 태스크는 영어와 한국어 문서의 요약을 생성하고 평가한다.
- 영어: CNN/DailyMail 뉴스 기사 요약
- 한국어: 네이버 뉴스 요약

ROUGE 메트릭은 언어별로 적절한 토크나이제이션을 적용한다:
- 영어: 공백 분리
- 한국어: Mecab 형태소(또는 음절 fallback)로 사전 토크나이제이션 후 ROUGE 계산
"""

from __future__ import annotations

from typing import Any

from src.adapters.fmapi import FMAPIClient, build_text_message
from src.datasets_loader import load_registry
from src.scoring.judge import DEFAULT_JUDGE_ENDPOINT, JudgeItem, judge_batch, resolve_rubric
from src.scoring.tokenizers import korean_tokenizer_backend, tokenize
from src.tasks.base import Task, Sample, register
from src.tasks.common import detect_column, language_batches, mean


def _compute_rouge(pred: str, gold: str, lang: str) -> dict[str, float]:
    """
    ROUGE 점수 계산 (Korean 형태소 사전 토크나이제이션 포함).

    한국어: tokenize + 공백 재결합 후 rouge_score 적용
    영어: 그대로 적용

    Args:
        pred: 예측(생성) 텍스트
        gold: 참조(정답) 텍스트
        lang: 언어 코드 ('en' | 'ko')

    Returns:
        dict with keys: rouge1, rouge2, rougeL
        각 값은 [0, 1] 범위의 float (f-measure)
    """
    from rouge_score import rouge_scorer

    # 한국어는 사전 토크나이제이션
    if lang == "ko":
        pred_tokens = tokenize(pred, "ko")
        gold_tokens = tokenize(gold, "ko")
        pred_text = " ".join(pred_tokens)
        gold_text = " ".join(gold_tokens)
    else:
        pred_text = pred
        gold_text = gold

    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=False
    )
    scores = scorer.score(gold_text, pred_text)

    return {
        "rouge1": scores["rouge1"].fmeasure,
        "rouge2": scores["rouge2"].fmeasure,
        "rougeL": scores["rougeL"].fmeasure,
    }


@register
class Txt5Task(Task):
    """텍스트 요약 생성 태스크 (TXT-5)."""

    task_id: str = "TXT-5"
    kind: str = "generation"
    is_vision: bool = False

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """영어/한국어 요약 데이터셋에서 seed 고정 subset을 로드.

        n개 샘플을 언어별로 균등 분할: 약 n//2씩 영어와 한국어.
        각 언어별 컬럼명(article/document, highlights/summary)을 자동 감지하고
        문서가 너무 길면 6000자 제한.
        """
        samples = []
        sample_id = 0

        for batch in language_batches(self.config, n, seed):
            hf_ds = batch.rows

            # 컬럼명 자동 감지
            col_article = detect_column(
                hf_ds,
                ["article", "document", "text", "content"],
                exclude=["summary", "highlights", "label"],
                what="문서 컬럼",
            )
            col_summary = detect_column(
                hf_ds,
                ["summary", "highlights", "summary_text", "target"],
                what="요약 컬럼",
            )

            for idx, row in enumerate(hf_ds):
                article_raw = row[col_article]
                summary_raw = row[col_summary]

                # 문서 길이 제한 (6000자 이상 시 잘라냄)
                MAX_ARTICLE_LEN = 6000
                if isinstance(article_raw, str) and len(article_raw) > MAX_ARTICLE_LEN:
                    article = article_raw[:MAX_ARTICLE_LEN]
                else:
                    article = article_raw

                # 참조(정답)를 문자열로 정규화
                if isinstance(summary_raw, list):
                    # CNN/DailyMail의 일부 분할에서는 summary가 리스트일 수 있음
                    reference = " ".join(summary_raw)
                else:
                    reference = str(summary_raw)

                sample = Sample(
                    sample_id=sample_id,
                    inputs={"document": article},
                    reference=reference,
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
        """요약 생성 프롬프트 구성.

        문서를 제시하고 간결한 요약을 요청한다.
        한국어 문서는 한국어로 요약하도록 지시.
        """
        document = sample.inputs["document"]
        lang = sample.lang

        if lang == "ko":
            prompt = f"""다음 문서에 대한 간결한 요약을 생성하세요. 요약은 원문의 핵심 내용을 2-3문장으로 정리해야 합니다.

문서:
{document}

요약:"""
        else:
            prompt = f"""Generate a concise summary of the following document. The summary should capture the key points in 2-3 sentences.

Document:
{document}

Summary:"""

        return build_text_message(prompt)

    def parse_output(self, raw_text: str, sample: Sample) -> str:
        """모델 응답을 요약 텍스트로 파싱.

        단순히 공백을 제거하고 반환. 요약은 자유형 텍스트이므로
        특별한 파싱이 필요 없음.
        """
        return raw_text.strip()

    def score(self, parsed: list[str], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 예측 결과를 집계해 ROUGE 메트릭 계산.

        - 모든 샘플에 대해 ROUGE 계산 (공백 예측도 계산)
        - 언어별 평균 ROUGE
        - BERTScore는 deferred (torch 미설치 환경 대비)
        - 한국어 토크나이저 백엔드 정보 포함
        """
        rouge_scores_per_lang = {"en": [], "ko": []}

        for pred, sample in zip(parsed, samples):
            lang = sample.lang
            reference = sample.reference

            rouge = _compute_rouge(pred, reference, lang)
            rouge_scores_per_lang[lang].append(rouge)

        # 언어별 평균 ROUGE
        per_language = {}
        for lang, scores_list in rouge_scores_per_lang.items():
            if scores_list:
                per_language[lang] = {
                    "rouge1": mean([s["rouge1"] for s in scores_list]),
                    "rouge2": mean([s["rouge2"] for s in scores_list]),
                    "rougeL": mean([s["rougeL"] for s in scores_list]),
                    "n_evaluated": len(scores_list),
                }
            else:
                per_language[lang] = {
                    "rouge1": None,
                    "rouge2": None,
                    "rougeL": None,
                    "n_evaluated": 0,
                }

        # 전체 평균 (모든 샘플)
        all_scores = rouge_scores_per_lang["en"] + rouge_scores_per_lang["ko"]

        return {
            "rouge1": mean([s["rouge1"] for s in all_scores]),
            "rouge2": mean([s["rouge2"] for s in all_scores]),
            "rougeL": mean([s["rougeL"] for s in all_scores]),
            "n_evaluated": len(parsed),
            "per_language": per_language,
            "bertscore": "deferred (torch 미설치)",
            "korean_backend": korean_tokenizer_backend(),
        }

    def judge_scores(
        self,
        parsed: list[str],
        samples: list[Sample],
        judge_client: FMAPIClient,
        judge_endpoint: str = DEFAULT_JUDGE_ENDPOINT,
    ) -> dict[str, Any]:
        """LLM 판사를 이용한 요약 품질 평가.

        Returns:
            dict with keys:
            - judge_score_mean: 평균 점수 (1-5)
            - judge_scores: 샘플별 점수 리스트
            - n_evaluated: 평가된 샘플 수
            - per_language: 언어별 평균 점수
        """
        items = [
            JudgeItem(
                question=f"Summarize the following document (language: {sample.lang}):",
                reference=str(sample.reference),
                candidate=pred,
                sample_id=sample.sample_id,
            )
            for pred, sample in zip(parsed, samples)
        ]

        scores = judge_batch(
            judge_client,
            items,
            task_id=self.task_id,
            rubric=resolve_rubric(self.task_id),
            judge_endpoint=judge_endpoint,
        )

        # 언어별 평균
        per_language = {}
        for lang in ["en", "ko"]:
            lang_scores = [
                score for score, sample in zip(scores, samples) if sample.lang == lang
            ]
            per_language[lang] = (
                {"mean": mean(lang_scores), "n": len(lang_scores)}
                if lang_scores
                else {"mean": None, "n": 0}
            )

        return {
            "judge_score_mean": mean(scores, default=3.0),
            "judge_scores": scores,
            "n_evaluated": len(scores),
            "per_language": per_language,
        }


if __name__ == "__main__":
    """
    End-to-end 테스트: 4샘플(영어 2, 한국어 2)로 실행.

    실행:
    cd /Users/stefano.jang/workspace/sme-llm-benchmark/image-text-performance
    python3 -m src.tasks.txt_5
    """
    import sys

    # 테스트용 작은 태스크 config
    test_config = {
        "datasets": {
            "en": "summarization_en",
            "ko": "summarization_ko",
        }
    }

    registry = load_registry()
    task = Txt5Task(test_config, registry)

    # 4샘플 로드 (영어 2, 한국어 2)
    print("=" * 70)
    print("TXT-5 Summarization Task - End-to-End Test")
    print("=" * 70)
    print("\nLoading 4 samples (2 en, 2 ko)...")
    samples = task.load_samples(n=4, seed=42)
    print(f"✓ Loaded {len(samples)} samples\n")

    for sample in samples:
        doc_preview = sample.inputs["document"][:60].replace("\n", " ")
        ref_preview = sample.reference[:60].replace("\n", " ")
        print(
            f"  [{sample.sample_id}] {sample.lang.upper()}: "
            f"doc={doc_preview}... "
            f"ref={ref_preview}..."
        )

    # 프롬프트 생성 및 모델 호출
    print("\n" + "=" * 70)
    print("Calling FMAPIClient (databricks-gpt-5-6-sol)...")
    print("=" * 70)

    try:
        with FMAPIClient(
            profile="ai_devtools", timeout_seconds=30, max_retries=3
        ) as client:
            parsed_outputs = []

            for sample in samples:
                messages = task.build_prompt(sample)

                print(f"\n[Sample {sample.sample_id}] ({sample.lang.upper()})")
                doc_preview = sample.inputs["document"][:40].replace("\n", " ")
                print(f"  Document: {doc_preview}...")

                # FMAPI 호출 (sol 모델, reasoning 비활성화)
                response = client.chat(
                    endpoint="databricks-gpt-5-6-sol",
                    messages=messages,
                    max_tokens=256,
                    extra_params={"reasoning_effort": "none"},
                )

                summary_preview = response.text[:80].replace("\n", " ")
                print(f"  Generated summary: {summary_preview}...")

                # 파싱
                parsed = task.parse_output(response.text, sample)
                parsed_outputs.append(parsed)

            # ROUGE 채점
            print("\n" + "=" * 70)
            print("Computing ROUGE Scores...")
            print("=" * 70)

            scores = task.score(parsed_outputs, samples)

            print(f"\nROUGE Metrics (overall):")
            print(f"  ROUGE-1: {scores['rouge1']:.4f}")
            print(f"  ROUGE-2: {scores['rouge2']:.4f}")
            print(f"  ROUGE-L: {scores['rougeL']:.4f}")
            print(f"  Samples evaluated: {scores['n_evaluated']}")
            print(f"  Korean backend: {scores['korean_backend']}")

            print(f"\nPer-language ROUGE:")
            for lang, lang_scores in scores["per_language"].items():
                if lang_scores["n_evaluated"] > 0:
                    print(
                        f"  {lang.upper()}:"
                        f" R1={lang_scores['rouge1']:.4f}"
                        f" R2={lang_scores['rouge2']:.4f}"
                        f" RL={lang_scores['rougeL']:.4f}"
                        f" (n={lang_scores['n_evaluated']})"
                    )

            # Judge 채점
            print("\n" + "=" * 70)
            print("Computing Judge Scores (databricks-gemini-3-1-pro)...")
            print("=" * 70)

            judge_results = task.judge_scores(
                parsed_outputs,
                samples,
                judge_client=client,
                judge_endpoint="databricks-gemini-3-1-pro",
            )

            print(f"\nJudge Metrics (1-5 scale):")
            print(f"  Mean judge score: {judge_results['judge_score_mean']:.2f}")
            print(f"  Samples evaluated: {judge_results['n_evaluated']}")

            print(f"\nPer-language judge scores:")
            for lang, lang_judge in judge_results["per_language"].items():
                if lang_judge["n"] > 0:
                    print(
                        f"  {lang.upper()}: "
                        f"mean={lang_judge['mean']:.2f} "
                        f"(n={lang_judge['n']})"
                    )

            print(f"\nIndividual judge scores:")
            for i, score in enumerate(judge_results["judge_scores"]):
                print(f"  Sample {i}: {score}/5")

            print("\n" + "=" * 70)
            print("✓ Test completed successfully")
            print("=" * 70)

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
