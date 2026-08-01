#!/usr/bin/env python
"""파인튜닝 전/후를 비교 평가한다.

PCS 태스크에 맞춰 두 가지를 본다:
  1. mask AP@[.5:.95] — 인스턴스 분할 품질 (COCO 방식의 간략 구현)
  2. presence accuracy — hard negative 에서 "없다"고 말할 수 있는지

    # 베이스 모델
    python evaluate.py --images-dir ... --annotations ... --categories person car

    # 어댑터 적용 후
    python evaluate.py --images-dir ... --annotations ... --categories person car \
        --adapter runs/sam3-lora-person-car/final
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from sam3_lora import CocoConceptSegmentation, collate_fn


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--categories", nargs="*", default=None)
    p.add_argument("--adapter", type=Path, default=None)
    p.add_argument("--model-id", default="facebook/sam3")
    p.add_argument("--image-size", type=int, default=1008)
    p.add_argument("--max-samples", type=int, default=200)
    p.add_argument("--threshold", type=float, default=0.5)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--device", default=None)
    p.add_argument("--tiny-debug", action="store_true")
    return p.parse_args()


def mask_iou_matrix(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """(P, H, W) x (G, H, W) → (P, G) IoU."""
    if len(pred) == 0 or len(gt) == 0:
        return torch.zeros(len(pred), len(gt))
    p = pred.flatten(1).float()
    g = gt.flatten(1).float()
    inter = p @ g.T
    union = p.sum(1, keepdim=True) + g.sum(1).unsqueeze(0) - inter
    return inter / union.clamp(min=1e-6)


def average_precision(records: list[tuple[float, bool]], n_gt: int) -> float:
    """단일 IoU 임계값에서의 AP (COCO 의 101-point interpolation)."""
    if n_gt == 0:
        return float("nan")
    if not records:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = np.cumsum([1 if m else 0 for _, m in records])
    fp = np.cumsum([0 if m else 1 for _, m in records])
    recall = tp / n_gt
    precision = tp / np.maximum(tp + fp, 1e-9)

    # precision 을 우측부터 단조 감소하도록 보정
    for i in range(len(precision) - 2, -1, -1):
        precision[i] = max(precision[i], precision[i + 1])

    thresholds = np.linspace(0, 1, 101)
    idx = np.searchsorted(recall, thresholds, side="left")
    interp = np.where(idx < len(precision), precision[np.clip(idx, 0, len(precision) - 1)], 0.0)
    return float(interp.mean())


def main() -> None:
    args = parse_args()
    from transformers import Sam3Config, Sam3Model, Sam3Processor

    device = (
        torch.device(args.device)
        if args.device
        else torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("mps")
        if torch.backends.mps.is_available()
        else torch.device("cpu")
    )
    dtype = getattr(torch, args.dtype)

    if args.tiny_debug:
        from smoke_test import build_processor, tiny_config

        model = Sam3Model(tiny_config()).to(dtype)
        processor = build_processor()
    else:
        config = Sam3Config.from_pretrained(args.model_id)
        config.image_size = args.image_size
        mask_size = args.image_size * 4 // 14
        model = Sam3Model.from_pretrained(args.model_id, config=config, dtype=dtype)
        processor = Sam3Processor.from_pretrained(
            args.model_id,
            size={"height": args.image_size, "width": args.image_size},
            mask_size={"height": mask_size, "width": mask_size},
        )

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        print(f"어댑터: {args.adapter}")
    else:
        print("어댑터 없음 (베이스 모델)")

    model.to(device).eval()

    dataset = CocoConceptSegmentation(
        images_dir=args.images_dir.expanduser(),
        annotations_path=args.annotations.expanduser(),
        processor=processor,
        categories=args.categories,
        negative_ratio=0.25,
    )
    if args.max_samples and len(dataset) > args.max_samples:
        dataset = Subset(dataset, list(range(args.max_samples)))
    loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)
    print(f"평가 샘플 {len(loader)}개")

    iou_thresholds = np.arange(0.5, 1.0, 0.05)
    records: dict[float, list[tuple[float, bool]]] = {t: [] for t in iou_thresholds}
    n_gt_total = 0

    presence_correct = 0
    n_positive = n_negative = 0
    fp_on_negative = 0

    for batch in loader:
        with torch.no_grad():
            outputs = model(
                pixel_values=batch["pixel_values"].to(device, dtype=dtype),
                input_ids=batch["input_ids"].to(device),
                attention_mask=batch["attention_mask"].to(device),
            )

        scores = (outputs.pred_logits.float().sigmoid() * outputs.presence_logits.float().sigmoid())[0]
        pred_masks = outputs.pred_masks.float()[0].sigmoid() > 0.5

        gt_masks = batch["targets"][0]["masks"]
        has_gt = len(gt_masks) > 0
        n_gt_total += len(gt_masks)

        # presence: 프롬프트 객체 존재 여부 예측이 맞았는지
        presence_pred = outputs.presence_logits.float().sigmoid().item() > 0.5
        if has_gt:
            n_positive += 1
        else:
            n_negative += 1
        presence_correct += int(presence_pred == has_gt)

        keep = scores > args.threshold
        kept_scores = scores[keep]
        kept_masks = pred_masks[keep]

        if not has_gt:
            fp_on_negative += int(keep.sum().item())
            continue

        if kept_masks.shape[-2:] != gt_masks.shape[-2:]:
            gt_masks = (
                torch.nn.functional.interpolate(
                    gt_masks.unsqueeze(1), size=kept_masks.shape[-2:], mode="nearest"
                )
                .squeeze(1)
                .bool()
            )

        ious = mask_iou_matrix(kept_masks.cpu(), gt_masks.cpu().bool())
        order = torch.argsort(kept_scores.cpu(), descending=True)

        for thr in iou_thresholds:
            matched: set[int] = set()
            for pi in order.tolist():
                if ious.shape[1] == 0:
                    records[thr].append((kept_scores[pi].item(), False))
                    continue
                candidates = [(ious[pi, gi].item(), gi) for gi in range(ious.shape[1]) if gi not in matched]
                best_iou, best_gi = max(candidates, default=(0.0, -1))
                hit = best_iou >= thr
                if hit:
                    matched.add(best_gi)
                records[thr].append((kept_scores[pi].item(), hit))

    aps = {thr: average_precision(records[thr], n_gt_total) for thr in iou_thresholds}
    valid = [v for v in aps.values() if not np.isnan(v)]

    print("\n=== 결과 ===")
    print(f"GT 인스턴스: {n_gt_total}")
    print(f"mask AP@0.50      : {aps[iou_thresholds[0]]:.4f}")
    print(f"mask AP@0.75      : {aps[min(iou_thresholds, key=lambda t: abs(t - 0.75))]:.4f}")
    print(f"mask AP@[.50:.95] : {np.mean(valid) if valid else float('nan'):.4f}")
    print(f"presence accuracy : {presence_correct}/{n_positive + n_negative} = "
          f"{presence_correct / max(1, n_positive + n_negative):.4f}  "
          f"(positive {n_positive}, negative {n_negative})")
    if n_negative:
        print(f"negative 이미지 평균 false positive: {fp_on_negative / n_negative:.2f}개")


if __name__ == "__main__":
    main()
