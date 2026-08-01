"""COCO instance-segmentation dataset for SAM3 concept (text-prompt) fine-tuning.

SAM3 는 "짧은 명사구(concept)" 하나를 받아서 그에 해당하는 **모든** 인스턴스를 찾는
Promptable Concept Segmentation(PCS) 모델이다. 따라서 학습 샘플 하나는

    (이미지, 텍스트 프롬프트 1개, 그 프롬프트에 해당하는 인스턴스 전부)

로 구성된다. 이 파일은 COCO 어노테이션을 그 형태로 잘라 준다.
한 이미지에 여러 카테고리가 있으면 (이미지, 카테고리) 조합마다 별개 샘플이 된다.

프롬프트에 해당하는 객체가 하나도 없는 **hard negative** 샘플도 섞는다.
논문에서 presence head 학습에 중요하다고 언급된 부분이고, 없으면 모델이
"프롬프트가 주어졌으면 뭐라도 있다"고 학습해 false positive 가 늘어난다.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


@dataclass
class Sample:
    image_path: Path
    text: str
    # xyxy, 원본 이미지 픽셀 좌표
    boxes: list[list[float]]
    # COCO polygon segmentation (annotation["segmentation"]) 리스트
    polygons: list[list]
    width: int
    height: int


def _polys_to_mask(polygons: list[list], height: int, width: int) -> np.ndarray:
    """COCO polygon / RLE 를 binary mask 로 렌더링한다."""
    from pycocotools import mask as mask_utils

    if not polygons:
        return np.zeros((height, width), dtype=np.uint8)

    if isinstance(polygons, dict):  # RLE
        rles = [polygons]
    else:
        # polygon 은 [[x1,y1,x2,y2,...], ...] 형태. 점이 3개 미만이면 면적이 없어 버린다.
        valid = [p for p in polygons if isinstance(p, list) and len(p) >= 6]
        if not valid:
            return np.zeros((height, width), dtype=np.uint8)
        rles = mask_utils.frPyObjects(valid, height, width)

    merged = mask_utils.merge(rles) if len(rles) > 1 else rles[0]
    return mask_utils.decode(merged).astype(np.uint8)


class CocoConceptSegmentation(Dataset):
    """COCO instances_*.json → SAM3 PCS 학습 샘플.

    Args:
        images_dir: 이미지가 들어 있는 디렉터리.
        annotations_path: COCO instances json 경로.
        processor: `Sam3Processor`. 이미지/텍스트 전처리와 리사이즈를 담당한다.
        categories: 사용할 카테고리 이름 화이트리스트. None 이면 전부 사용.
        min_area: 이 면적(원본 픽셀) 이하 인스턴스는 버린다. 아주 작은 객체는
            mask_size 로 다운샘플하면 사라져서 학습 신호가 노이즈가 된다.
        negative_ratio: 양성 샘플 개수 대비 hard negative 샘플 비율.
        max_instances: 샘플당 인스턴스 상한. num_queries 보다 훨씬 작게 유지한다.
        seed: negative 샘플링 재현용.
    """

    def __init__(
        self,
        images_dir: str | Path,
        annotations_path: str | Path,
        processor,
        categories: list[str] | None = None,
        min_area: float = 400.0,
        negative_ratio: float = 0.25,
        max_instances: int = 20,
        skip_crowd: bool = True,
        seed: int = 0,
    ):
        self.images_dir = Path(images_dir)
        self.processor = processor
        self.max_instances = max_instances

        # 모델이 마스크를 예측하는 해상도. GT 마스크도 여기에 맞춰 렌더링한다.
        mask_size = processor.image_processor.mask_size
        self.mask_height = mask_size["height"]
        self.mask_width = mask_size["width"]

        raw = json.loads(Path(annotations_path).read_text())
        id_to_name = {c["id"]: c["name"] for c in raw["categories"]}
        images = {im["id"]: im for im in raw["images"]}

        allowed = set(categories) if categories else None

        # (image_id, category_id) → annotation 리스트
        grouped: dict[tuple[int, int], list[dict]] = defaultdict(list)
        cats_per_image: dict[int, set[int]] = defaultdict(set)
        for ann in raw["annotations"]:
            if skip_crowd and ann.get("iscrowd", 0):
                continue
            if ann.get("area", 0) < min_area:
                continue
            name = id_to_name.get(ann["category_id"])
            if name is None or (allowed is not None and name not in allowed):
                continue
            if ann["image_id"] not in images:
                continue
            grouped[(ann["image_id"], ann["category_id"])].append(ann)
            cats_per_image[ann["image_id"]].add(ann["category_id"])

        self.samples: list[Sample] = []
        for (image_id, category_id), anns in grouped.items():
            im = images[image_id]
            anns = anns[:max_instances]
            self.samples.append(
                Sample(
                    image_path=self.images_dir / im["file_name"],
                    text=id_to_name[category_id],
                    boxes=[_xywh_to_xyxy(a["bbox"]) for a in anns],
                    polygons=[a["segmentation"] for a in anns],
                    width=im["width"],
                    height=im["height"],
                )
            )

        # hard negative: 이미지에 실제로 없는 카테고리를 프롬프트로 준다.
        # 완전 무작위 카테고리보다, 같은 데이터셋에 흔한 카테고리를 고르는 게 더 어렵다.
        if negative_ratio > 0 and self.samples:
            rng = random.Random(seed)
            pool = sorted({s.text for s in self.samples})
            n_neg = int(len(self.samples) * negative_ratio)
            positives = list(self.samples)
            for _ in range(n_neg):
                base = rng.choice(positives)
                image_id = next(
                    (i for i, im in images.items() if self.images_dir / im["file_name"] == base.image_path),
                    None,
                )
                present = {id_to_name[c] for c in cats_per_image.get(image_id, set())}
                choices = [c for c in pool if c not in present]
                if not choices:
                    continue
                self.samples.append(
                    Sample(
                        image_path=base.image_path,
                        text=rng.choice(choices),
                        boxes=[],
                        polygons=[],
                        width=base.width,
                        height=base.height,
                    )
                )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict:
        sample = self.samples[index]
        image = Image.open(sample.image_path).convert("RGB")

        encoding = self.processor(images=image, text=sample.text, return_tensors="pt")

        # 박스는 xyxy → [0,1] 정규화 후 cxcywh. Hungarian matching 과 L1/GIoU 손실을
        # 정규화 공간에서 계산하므로 원본 해상도에 무관해진다.
        boxes = torch.tensor(sample.boxes, dtype=torch.float32).reshape(-1, 4)
        if len(boxes):
            scale = torch.tensor(
                [sample.width, sample.height, sample.width, sample.height], dtype=torch.float32
            )
            boxes = (boxes / scale).clamp(0.0, 1.0)

        # 마스크는 모델 출력 해상도(mask_size)에 맞춰 렌더링한다. 원본 해상도로
        # 만들어서 collate 하면 이미지마다 크기가 달라 배치가 안 된다.
        masks = torch.zeros((len(sample.polygons), self.mask_height, self.mask_width), dtype=torch.float32)
        for i, polys in enumerate(sample.polygons):
            full = _polys_to_mask(polys, sample.height, sample.width)
            resized = np.array(
                Image.fromarray(full * 255).resize(
                    (self.mask_width, self.mask_height), Image.Resampling.BILINEAR
                )
            )
            masks[i] = torch.from_numpy((resized > 127).astype(np.float32))

        return {
            "pixel_values": encoding["pixel_values"][0],
            "input_ids": encoding["input_ids"][0],
            "attention_mask": encoding["attention_mask"][0],
            "boxes": boxes,          # (n, 4) cxcywh, normalized
            "masks": masks,          # (n, mask_h, mask_w) in {0, 1}
            "text": sample.text,
            "original_size": (sample.height, sample.width),
            "image_path": str(sample.image_path),
        }


def _xywh_to_xyxy(bbox: list[float]) -> list[float]:
    x, y, w, h = bbox
    return [x, y, x + w, y + h]


def collate_fn(batch: list[dict]) -> dict:
    """인스턴스 개수가 샘플마다 다르므로 target 은 리스트로 남겨 둔다.

    DETR 계열 손실은 배치 원소별로 Hungarian matching 을 돌리기 때문에
    target 을 패딩할 필요가 없다.
    """
    return {
        "pixel_values": torch.stack([b["pixel_values"] for b in batch]),
        "input_ids": torch.stack([b["input_ids"] for b in batch]),
        "attention_mask": torch.stack([b["attention_mask"] for b in batch]),
        "targets": [{"boxes": b["boxes"], "masks": b["masks"]} for b in batch],
        "texts": [b["text"] for b in batch],
        "original_sizes": [b["original_size"] for b in batch],
        "image_paths": [b["image_path"] for b in batch],
    }
