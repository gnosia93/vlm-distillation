#!/usr/bin/env python
"""학습한 LoRA 어댑터로 SAM3 추론 + 결과 시각화.

    python infer_lora.py \
        --adapter runs/sam3-lora-person-car/final \
        --image ~/fiftyone/coco-2017/validation/data/000000000139.jpg \
        --text person \
        --out out.png

--adapter 를 생략하면 베이스 모델 그대로 돌아간다. 파인튜닝 전/후 비교에 쓴다.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from PIL import Image


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--image", required=True, type=Path)
    p.add_argument("--text", required=True, help="concept 프롬프트 (짧은 명사구)")
    p.add_argument("--adapter", type=Path, default=None, help="save_pretrained 한 어댑터 디렉터리")
    p.add_argument("--model-id", default="facebook/sam3")
    p.add_argument("--threshold", type=float, default=0.5, help="인스턴스 점수 임계값")
    p.add_argument("--mask-threshold", type=float, default=0.5)
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])
    p.add_argument("--device", default=None)
    p.add_argument("--merge", action="store_true", help="LoRA 를 베이스에 merge (추론 속도 향상)")
    p.add_argument("--out", type=Path, default=None, help="시각화 PNG 저장 경로")
    return p.parse_args()


def pick_device(requested: str | None) -> torch.device:
    if requested:
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    from transformers import Sam3Config, Sam3Model, Sam3Processor

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)

    # 어댑터를 저장할 때 processor 도 같이 저장했으므로(processor_config.json 안에
    # image_processor 설정이 들어 있다) 학습과 동일한 해상도를 그대로 재현한다.
    # 해상도가 어긋나면 마스크 품질이 조용히 나빠진다.
    processor_src = (
        args.adapter
        if args.adapter and (args.adapter / "processor_config.json").exists()
        else args.model_id
    )
    processor = Sam3Processor.from_pretrained(processor_src)
    image_size = processor.image_processor.size["height"]
    print(f"processor: {processor_src} ({image_size}px)")

    # 학습 때 해상도를 바꿨으면 모델 config 도 같이 맞춰야 position embedding 이 맞는다.
    config = Sam3Config.from_pretrained(args.model_id)
    if config.image_size != image_size:
        config.image_size = image_size
    model = Sam3Model.from_pretrained(args.model_id, config=config, dtype=dtype)

    if args.adapter:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter)
        if args.merge:
            # merge_and_unload 는 LoRA 를 베이스 가중치에 흡수해 오버헤드를 없앤다.
            model = model.merge_and_unload()
        print(f"어댑터 로드: {args.adapter}")

    model.to(device).eval()

    image = Image.open(args.image).convert("RGB")
    inputs = processor(images=image, text=args.text, return_tensors="pt").to(device)
    inputs["pixel_values"] = inputs["pixel_values"].to(dtype)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=args.threshold,
        mask_threshold=args.mask_threshold,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

    scores = results["scores"]
    print(f"'{args.text}' → {len(scores)}개 검출 (presence={outputs.presence_logits.sigmoid().item():.3f})")
    for i, (score, box) in enumerate(zip(scores.tolist(), results["boxes"].tolist())):
        print(f"  #{i} score={score:.3f} box=[{', '.join(f'{v:.0f}' for v in box)}]")

    if args.out:
        visualize(image, results, args.text, args.out)
        print(f"시각화 저장: {args.out}")


def visualize(image: Image.Image, results: dict, text: str, out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    fig, ax = plt.subplots(figsize=(10, 10 * image.height / image.width))
    ax.imshow(image)
    ax.set_title(f'"{text}" — {len(results["scores"])} instances')
    ax.axis("off")

    cmap = plt.get_cmap("tab10")
    masks = results["masks"].cpu().numpy()
    for i, (mask, box, score) in enumerate(
        zip(masks, results["boxes"].cpu().numpy(), results["scores"].cpu().numpy())
    ):
        color = cmap(i % 10)
        overlay = np.zeros((*mask.shape, 4))
        overlay[mask > 0] = (*color[:3], 0.5)
        ax.imshow(overlay)
        x0, y0, x1, y1 = box
        ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor=color, linewidth=2))
        ax.text(x0, max(y0 - 4, 0), f"{score:.2f}", color="white", fontsize=9,
                bbox=dict(facecolor=color, alpha=0.8, pad=1, edgecolor="none"))

    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
