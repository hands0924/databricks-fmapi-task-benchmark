"""IMG-3 무기/위협 존재 여부 이진 분류 (binary).

이 태스크는 이미지에서 무기 또는 위협 객체의 존재 여부를 판별한다.
- 데이터셋: Subh775/WeaponDetection (validation split)
- 레이블: binary (1=무기/위협 있음, 0=없음)
- 메트릭: accuracy, f1 (binary_metrics)
- vision=True: GLM 등 vision 미지원 모델은 N/A로 자동 스킵

무기 클래스 ID 매핑 (WeaponDetection 데이터셋의 29개 클래스):
0=Background, 1=Gun, 2=Knife, 3=Pistol, 4=Rifle, 5=Blood,
6=Machine Gun, 7=Shotgun, 8=Sword, 9=Revolver, 10=Hammer,
11=Grenade, 12=Axe, 13=Dynamite, 14=Pistol Silencer, 15=Machete,
16=Rifle Silencer, 17=Shotgun Silencer, 18=Bow, 19=Arrow,
20=Spike, 21=Scythe, 22=Crossbow, 23=Flashbang, 24=Hand Grenade,
25=Mine, 26=Molotov Cocktail, 27=Spear, 28=Staff/Stick

무기/위협 판별: 클래스 ID ≥ 1 (Background 제외)인 객체가 있으면 1, 없으면 0.
"""

from __future__ import annotations

from typing import Any

from src.adapters.fmapi import build_image_message, FMAPIClient
from src.adapters.images import pil_to_data_url
from src.datasets_loader import load_registry
from src.tasks.base import Task, Sample, register
from src.tasks.common import binary_score_summary, load_dataset_rows, parse_binary_label


@register
class Img3Task(Task):
    """무기/위협 존재 여부 이진 분류 태스크 (IMG-3)."""

    task_id: str = "IMG-3"
    kind: str = "binary"
    is_vision: bool = True

    def load_samples(self, n: int, seed: int) -> list[Sample]:
        """WeaponDetection 데이터셋에서 seed 고정 subset을 로드.

        각 샘플은 image + objects(카테고리 ID list)를 갖는다.
        objects.category에서 무기/위협 관련 클래스를 판별:
        - 무기 클래스: Guns(3), Guns perspective(4), Heavy Gun(6), Knife(7),
          Knife_Deploy(8), Knife_Weapon(9), Long guns(10), Pistol(12),
          Rifle(13), Shotgun(14), handgun(19), pistol(23), pistols(24),
          rifle(25), shotgun(26), weapon(28)
        - 배경 클래스: weapons(0), Aggressor(1), Blood(2), Person(11),
          Stabbing(15), Victim(16), al(17), larga(21), person(22),
          violence(27)

        객체 카테고리에 무기 클래스가 하나라도 있으면 label=1(무기 있음),
        없으면 label=0(무기 없음).
        """
        # 무기/위협 클래스 ID (WeaponDetection 데이터셋)
        WEAPON_CLASS_IDS = {3, 4, 6, 7, 8, 9, 10, 12, 13, 14, 19, 23, 24, 25, 26, 28}

        # HF 데이터셋 로드 (큰 데이터셋이므로 더 큰 슬라이스를 로드한 후 필터링)
        # 밸런스 확보를 위해 더 많이 로드
        load_n = max(n * 3, 150)
        hf_ds = load_dataset_rows("weapon", load_n, seed, default_split="validation")

        samples = []
        sample_id = 0

        for idx, row in enumerate(hf_ds):
            image = row.get("image")
            objects = row.get("objects")

            if image is None:
                continue

            # objects에서 category id 추출
            if objects is None:
                category_ids = []
            else:
                category_ids = objects.get("category", []) if isinstance(objects, dict) else []

            if not isinstance(category_ids, list):
                category_ids = [category_ids]

            # 무기/위협 판별: 무기 클래스가 하나라도 있으면 1
            has_weapon = any(
                isinstance(cat_id, int) and cat_id in WEAPON_CLASS_IDS
                for cat_id in category_ids
            )

            label_int = 1 if has_weapon else 0

            sample = Sample(
                sample_id=sample_id,
                inputs={"image": image},
                reference=label_int,
                lang="en",
                meta={
                    "dataset": "weapon",
                    "source_idx": idx,
                    "num_objects": len(category_ids),
                    "categories": category_ids,
                },
            )
            samples.append(sample)
            sample_id += 1

            if len(samples) >= n:
                break

        return samples

    def build_prompt(self, sample: Sample) -> list[dict[str, Any]]:
        """무기/위협 존재 여부 판별 프롬프트.

        PIL 이미지를 data URL로 변환 후 멀티모달 메시지 구성.
        명확한 yes/no 질문으로 구성.
        """
        image = sample.inputs["image"]
        image_url = pil_to_data_url(image, max_side=768)

        prompt = "Does this image contain a weapon or threat (gun, knife, pistol, rifle, blood, etc.)? Answer exactly yes or no."
        return build_image_message(prompt, image_url)

    def parse_output(self, raw_text: str, sample: Sample) -> int | None:
        """모델 응답을 binary 라벨로 파싱.

        "yes"→1, "no"→0으로 매핑.
        대소문자 무시, 여러 단어 포함 시에도 동작.
        부정 표현("no", "not") 우선 검사로 "no weapon" 오분류 방지.
        파싱 불가 시 None 반환.
        """
        return parse_binary_label(
            raw_text,
            positive=("yes", "threat", "weapon"),
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
    """간단한 end-to-end 테스트: 4샘플로 무기 검출 실행."""
    import sys

    print("=== IMG-3 무기/위협 검출 테스트 ===\n")

    registry = load_registry()
    task_config = {"datasets": {"en": "weapon"}}
    task = Img3Task(task_config, registry)

    # 4샘플 로드
    print("Loading 4 samples...")
    try:
        samples = task.load_samples(n=4, seed=42)
        print(f"Loaded {len(samples)} samples")

        # 클래스 분포 출력
        class_0_count = sum(1 for s in samples if s.reference == 0)
        class_1_count = sum(1 for s in samples if s.reference == 1)
        print(f"Class balance: class_0={class_0_count}, class_1={class_1_count}\n")

    except Exception as e:
        print(f"Error loading samples: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)

    if not samples:
        print("No samples loaded!", file=sys.stderr)
        sys.exit(1)

    for sample in samples:
        print(
            f"[Sample {sample.sample_id}] Image: {sample.inputs['image'].size}, "
            f"Label: {sample.reference} (weapon={'yes' if sample.reference else 'no'})"
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
