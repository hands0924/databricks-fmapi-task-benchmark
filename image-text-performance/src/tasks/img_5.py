"""IMG-5 사람 포함 여부 이진 분류 (binary).

이 태스크는 이미지에서 사람(person) 객체의 존재 여부를 판별한다.
- 데이터셋: detection-datasets/coco (validation split)
- 레이블: binary (1=사람 있음, 0=사람 없음)
- 메트릭: accuracy, f1 (binary_metrics)
- vision=True: GLM 등 vision 미지원 모델은 N/A로 자동 스킵

COCO 데이터셋의 objects.category config에서:
- 클래스 ID 0 = person (COCO 공식 클래스 순서)
- 클래스 ID 1-79 = 기타 객체들

사람 판별: 객체 카테고리에 ID 0(person)이 있으면 1, 없으면 0.
"""

from __future__ import annotations

from typing import Any

from src.adapters.fmapi import build_image_message, FMAPIClient
from src.adapters.images import pil_to_data_url
from src.datasets_loader import load_registry
from src.tasks.base import Task, Sample, register
from src.tasks.common import binary_score_summary, load_dataset_rows, parse_binary_label


@register
class Img5Task(Task):
    """사람 포함 여부 이진 분류 태스크 (IMG-5)."""

    task_id: str = "IMG-5"
    kind: str = "binary"
    is_vision: bool = True

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """COCO 데이터셋에서 seed 고정 subset을 로드 (이진 분류: 사람 포함 여부).

        각 샘플은 image + objects(카테고리 ID list)를 갖는다.
        objects의 category에 0(person)이 있으면 label=1(사람 있음),
        없으면 label=0(사람 없음).

        COCO 데이터셋은 대부분 사람을 포함하므로 밸런스 확보를 위해
        더 큰 n을 요청한 후 Python에서 n개까지만 사용.
        load_hf_split은 streaming이므로 n을 크게 설정해도 필요한 만큼만 로드.

        load_hf_split은 list[dict]를 반환하므로, 각 행을 dict로 처리.
        """
        # 보다 더 많은 샘플을 로드해서 클래스 밸런스 확보
        # streaming이므로 load_n을 크게 설정해도 필요한 만큼만 스트리밍 로드
        load_n = max(n * 3, 30)
        hf_ds = load_dataset_rows("img_person", load_n, seed, default_split="val")

        samples = []
        sample_id = 0
        class_counts = {0: 0, 1: 0}

        for idx, row in enumerate(hf_ds):
            image = row.get("image")
            objects = row.get("objects")

            if image is None:
                continue

            # objects는 dict with "category" key containing list of category IDs
            category_ids = objects.get("category", []) if isinstance(objects, dict) else []
            if not isinstance(category_ids, list):
                category_ids = [category_ids]

            # 사람 판별: COCO person class ID = 0
            has_person = any(
                isinstance(cat_id, int) and cat_id == 0 for cat_id in category_ids
            )

            label_int = 1 if has_person else 0
            class_counts[label_int] += 1

            sample = Sample(
                sample_id=sample_id,
                inputs={"image": image},
                reference=label_int,
                lang="en",
                meta={
                    "dataset": "img_person",
                    "source_idx": idx,
                    "num_objects": len(category_ids) if isinstance(category_ids, list) else 0,
                    "has_person": has_person,
                },
            )
            samples.append(sample)
            sample_id += 1

            if len(samples) >= n:
                break

        return samples

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """사람 포함 여부 판별 프롬프트.

        PIL 이미지를 data URL로 변환 후 멀티모달 메시지 구성.
        명확한 yes/no 질문으로 구성.
        """
        image = sample.inputs["image"]
        image_url = pil_to_data_url(image, max_side=768)

        prompt = "Is there a person visible in this image? Answer exactly yes or no."
        return build_image_message(prompt, image_url)

    def parse_output(self, raw_text: str, sample: Sample) -> int | None:
        """모델 응답을 binary 라벨로 파싱.

        "yes"→1, "no"→0으로 매핑.
        대소문자 무시, 여러 단어 포함 시에도 동작.
        부정 표현("no", "not") 우선 검사로 "no person" 오분류 방지.
        파싱 불가 시 None 반환.
        """
        return parse_binary_label(
            raw_text,
            positive=("yes", "person", "people"),
            negative=("no", "not", "none"),
            negative_first=True,
        )

    def score(self, parsed: list[int | None], samples: list[Sample]) -> dict[str, Any]:
        """파싱된 예측 결과를 집계해 메트릭 계산.

        - None 예측값 제외 (unparseable)
        - binary_metrics로 accuracy/f1 계산
        - 혼동 행렬(confusion matrix) 포함
        - 클래스 밸런스 포함
        """
        return binary_score_summary(
            parsed,
            samples,
            class_balance_keys=("class_0", "class_1"),
        )


if __name__ == "__main__":
    """간단한 end-to-end 테스트: 4샘플로 사람 검출 실행."""
    import sys
    from PIL import Image
    import io
    import urllib.request
    from io import BytesIO

    print("=== IMG-5 사람 포함 여부 테스트 ===\n")

    registry = load_registry()
    task_config = {"datasets": {"en": "img_person"}}
    task = Img5Task(task_config, registry)

    # 테스트용 실제 이미지 URL 사용 (COCO 데이터셋의 검증 이미지)
    # person=1인 이미지, person=0인 이미지 혼합
    print("Downloading test images from COCO...\n")

    test_urls = [
        ("http://images.cocodataset.org/val2014/COCO_val2014_000000184613.jpg", 1),  # people with umbrella
        ("http://images.cocodataset.org/val2014/COCO_val2014_000000581921.jpg", 1),  # people
        ("http://images.cocodataset.org/val2014/COCO_val2014_000000222564.jpg", 0),  # dogs (no person)
        ("http://images.cocodataset.org/val2014/COCO_val2014_000000171230.jpg", 0),  # outdoor scene
    ]

    mock_samples = []
    for idx, (url, label) in enumerate(test_urls):
        try:
            with urllib.request.urlopen(url, timeout=5) as response:
                image_data = response.read()
            image = Image.open(BytesIO(image_data))
            image.load()

            sample = Sample(
                sample_id=idx,
                inputs={"image": image},
                reference=label,
                lang="en",
                meta={"dataset": "img_person", "source_url": url},
            )
            mock_samples.append(sample)
            print(f"  ✓ Sample {idx}: {url.split('/')[-1]} (person={label})")
        except Exception as e:
            print(f"  ✗ Failed to load {url}: {e}")

    if not mock_samples:
        print("\n✗ Failed to load test images!", file=sys.stderr)
        sys.exit(1)

    samples = mock_samples
    print(f"\nLoaded {len(samples)} test samples")

    # 클래스 분포
    class_0_count = sum(1 for s in samples if s.reference == 0)
    class_1_count = sum(1 for s in samples if s.reference == 1)
    print(f"Class balance: class_0={class_0_count}, class_1={class_1_count}\n")

    for sample in samples:
        print(
            f"[Sample {sample.sample_id}] Image: {sample.inputs['image'].size}, "
            f"Label: {sample.reference} (person={'yes' if sample.reference else 'no'})"
        )
    print()

    # 프롬프트 생성 및 모델 호출
    print("Calling FMAPI (databricks-claude-opus-5)...\n")

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
                    max_tokens=64,
                    extra_params={"thinking": {"type": "disabled"}},
                )

                # 파싱
                parsed = task.parse_output(response.text, sample)
                parsed_outputs.append(parsed)

                print(f"  Raw response: {response.text[:80]}")
                print(f"  Parsed: {parsed}")
                print(f"  Expected: {sample.reference}")
                print()

            # 채점
            print("\n=== Scoring (binary_metrics) ===\n")
            scores = task.score(parsed_outputs, samples)

            print(f"Accuracy: {scores['accuracy']:.3f}")
            print(f"F1: {scores['f1']:.3f}")
            print(f"Confusion matrix: {scores['confusion_matrix']}")
            print(f"Class balance: {scores['class_balance']}")
            print(f"Evaluated: {scores['n_evaluated']}/{len(samples)}")
            print(f"Unparsed: {scores['n_unparsed']}")

    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)
