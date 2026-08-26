"""IMG-1 이미지 캡셔닝 태스크 (generation).

이 태스크는 이미지를 주고 모델이 한 문장의 캡션을 생성하도록 한다.
- 데이터셋: yerevann/coco-karpathy (COCO 이미지 + 5개 참고 캡션)
- 메트릭: token_f1 (lexical overlap, 임시), judge (LLM-as-judge)
- vision=True: GLM 등 vision 미지원 모델은 N/A로 자동 스킵
"""

from __future__ import annotations

import urllib.request
from io import BytesIO
from typing import Any, ClassVar

from PIL import Image

from src.adapters.fmapi import build_image_message, FMAPIClient
from src.adapters.images import pil_to_data_url
from src.datasets_loader import load_registry
from src.scoring.metrics import token_f1
from src.scoring.judge import DEFAULT_JUDGE_ENDPOINT, JudgeItem, judge_batch, resolve_rubric
from src.tasks.base import Task, Sample, register
from src.tasks.common import best_reference_score, load_dataset_rows, mean


@register
class Img1Task(Task):
    """이미지 캡셔닝 생성 태스크 (IMG-1)."""

    task_id: str = "IMG-1"
    kind: str = "generation"
    is_vision: bool = True

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """COCO-Karpathy 데이터셋에서 이미지+참고 캡션을 로드.

        yerevann/coco-karpathy는 URL을 제공하므로 해당 URL에서
        이미지를 다운로드해 PIL.Image로 변환한다.
        reference는 5개 캡션 리스트를 저장한다.
        """
        # HF 데이터셋 로드
        hf_ds = load_dataset_rows("img_caption", n, seed, default_split="validation")

        samples = []
        for idx, row in enumerate(hf_ds):
            url = row.get("url")
            sentences = row.get("sentences", [])

            if not url or not sentences:
                continue

            # URL에서 이미지 다운로드
            try:
                with urllib.request.urlopen(url, timeout=5) as response:
                    image_data = response.read()
                image = Image.open(BytesIO(image_data))
                image.load()  # 즉시 로드 (lazy loading 방지)
            except Exception as e:
                # URL 접근 실패 시 스킵
                print(f"  이미지 로드 실패 (URL: {url[:50]}...): {e}")
                continue

            sample = Sample(
                sample_id=idx,
                inputs={"image": image},
                reference=sentences,  # list of 5 captions
                lang="en",
                meta={
                    "dataset": "img_caption",
                    "filename": row.get("filename", ""),
                    "cocoid": row.get("cocoid", ""),
                    "url": url,
                },
            )
            samples.append(sample)

            if len(samples) >= n:
                break

        return samples

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """이미지를 보고 한 문장의 캡션을 생성하는 프롬프트.

        PIL 이미지를 data URL로 변환 후 멀티모달 메시지 구성.
        """
        image = sample.inputs["image"]
        image_url = pil_to_data_url(image, max_side=768)

        prompt = "Describe this image in one sentence."
        return build_image_message(prompt, image_url)

    def parse_output(self, raw_text: str, sample: Sample) -> str:
        """모델 응답을 정규화된 캡션 텍스트로 파싱.

        단순히 앞뒤 공백을 제거하고 반환.
        """
        return raw_text.strip()

    def score(self, parsed: list[str], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 캡션을 참고 캡션과 비교해 token_f1 계산.

        각 샘플마다 생성된 캡션과 5개 참고 캡션 중 최고 F1을 선택.
        평균 token_f1을 메인 메트릭으로, bertscore는 torch 미설치로
        아직 계산 불가 (장기 계획).
        """
        if not parsed or not samples:
            return {
                "caption_token_f1": 0.0,
                "n_evaluated": 0,
                "notes": "bertscore deferred (torch 미설치)",
            }

        best_f1_scores = [
            best_reference_score(
                pred,
                sample.reference,
                lambda p, g, lang=sample.lang: token_f1(p, g, lang=lang),
            )
            for pred, sample in zip(parsed, samples)
            if pred
        ]

        return {
            "caption_token_f1": float(mean(best_f1_scores)),
            "n_evaluated": len(best_f1_scores),
            "notes": "bertscore deferred (torch 미설치)",
        }

    judge_rubric_fallback: ClassVar[dict[str, Any]] = {
        "name": "Image Captioning",
        "description": "Evaluate image captions for accuracy, completeness, and clarity.",
        "anchors": {
            "1": "Inaccurate or missing caption",
            "2": "Partially correct caption",
            "3": "Acceptable caption with minor issues",
            "4": "Good caption with minor omissions",
            "5": "Excellent, complete, and accurate caption",
        },
    }

    def judge_scores(
        self,
        parsed: list[str],
        samples: list[Sample],
        judge_client: FMAPIClient,
        judge_endpoint: str = DEFAULT_JUDGE_ENDPOINT,
    ) -> dict[str, Any]:
        """LLM-as-judge로 캡션 품질 평가.

        생성된 캡션과 참고 캡션을 텍스트로만 비교 (이미지는 불필요).
        빈 캡션과 judge 호출/파싱 실패는 None으로 기록하고 평균에서 제외한다.
        """
        if not parsed or not samples:
            return {
                "judge_score_mean": 0.0,
                "judge_scores": [],
                "n_evaluated": 0,
            }

        items = [
            JudgeItem(
                question="Describe this image in one sentence.",
                reference=self.judge_reference(sample),
                candidate=pred,
                sample_id=sample.sample_id,
            )
            for pred, sample in zip(parsed, samples)
        ]

        judge_scores = judge_batch(
            judge_client,
            items,
            task_id=self.task_id,
            rubric=resolve_rubric(self.task_id, self.judge_rubric_fallback),
            judge_endpoint=judge_endpoint,
            fallback_score=None,
            skip_empty=True,
        )

        valid_scores = [s for s in judge_scores if s is not None]

        return {
            "judge_score_mean": float(mean(valid_scores)),
            "judge_scores": judge_scores,
            "n_evaluated": len(valid_scores),
        }


if __name__ == "__main__":
    """간단한 end-to-end 테스트: 2샘플로 캡셔닝 실행."""
    import sys

    print("=== IMG-1 캡셔닝 테스트 ===\n")

    registry = load_registry()
    task_config = {"datasets": {"en": "img_caption"}}
    task = Img1Task(task_config, registry)

    # 2샘플 로드
    print("Loading 2 samples...")
    try:
        samples = task.load_samples(n=2, seed=42)
        print(f"Loaded {len(samples)} samples\n")
    except Exception as e:
        print(f"Error loading samples: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)

    if not samples:
        print("No samples loaded!", file=sys.stderr)
        sys.exit(1)

    for sample in samples[:1]:
        print(f"[Sample {sample.sample_id}]")
        print(f"  Image: PIL shape {sample.inputs['image'].size}")
        print(f"  Reference captions: {sample.reference[:2]}...")
        print()

    # 프롬프트 생성 및 모델 호출
    print("Calling FMAPI (claude-opus-5)...\n")

    try:
        with FMAPIClient(profile="ai_devtools", timeout_seconds=30) as client:
            parsed_outputs = []

            for i, sample in enumerate(samples):
                print(f"[Sample {sample.sample_id}]")

                # 프롬프트 구성
                messages = task.build_prompt(sample)

                # FMAPI 호출
                response = client.chat(
                    endpoint="databricks-claude-opus-5",
                    messages=messages,
                    max_tokens=256,
                    extra_params={"thinking": {"type": "disabled"}},
                )

                # 파싱
                parsed = task.parse_output(response.text, sample)
                parsed_outputs.append(parsed)

                print(f"  Raw response: {response.text[:100]}...")
                print(f"  Parsed: {parsed[:80]}...")
                print(f"  Reference[0]: {sample.reference[0][:80]}...")
                print()

            # 채점
            print("\n=== Scoring ===\n")
            scores = task.score(parsed_outputs, samples)

            print(f"Caption token_f1: {scores['caption_token_f1']:.3f}")
            print(f"Evaluated: {scores['n_evaluated']}/{len(samples)}")
            print(f"Notes: {scores['notes']}")

            # Judge 점수 (선택사항)
            print("\n=== Judge Scoring (with Gemini) ===\n")
            judge_scores = task.judge_scores(parsed_outputs, samples, client)
            print(f"Judge mean score: {judge_scores['judge_score_mean']:.2f}")
            print(f"Judge scores: {judge_scores['judge_scores']}")
            print(f"Evaluated by judge: {judge_scores['n_evaluated']}/{len(samples)}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)
