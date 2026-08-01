#!/usr/bin/env python
"""SAM3 (facebook/sam3) LoRA 파인튜닝.

예시:
    # 로컬 COCO validation 500장으로 person/car 두 개념만 학습 (스모크용)
    python train_lora.py \
        --images-dir ~/fiftyone/coco-2017/validation/data \
        --annotations ~/fiftyone/coco-2017/validation/labels.json \
        --categories person car \
        --epochs 2 --batch-size 1 --image-size 560 \
        --output-dir runs/sam3-lora-person-car

    # 실사용: 전체 카테고리, ViT 까지 LoRA, bf16
    python train_lora.py \
        --images-dir /data/coco/train2017 \
        --annotations /data/coco/annotations/instances_train2017.json \
        --lora-stages vision detr_encoder detr_decoder mask_decoder \
        --epochs 10 --batch-size 4 --dtype bfloat16 \
        --output-dir runs/sam3-lora-full

facebook/sam3 는 gated repo 라서 먼저 HF 에서 라이선스에 동의하고
`hf auth login` (또는 HF_TOKEN 환경변수) 을 해 두어야 한다.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader, random_split

from sam3_lora import (
    DEFAULT_MODULES_TO_SAVE,
    CocoConceptSegmentation,
    Sam3SetCriterion,
    build_lora_config,
    collate_fn,
    summarize_trainable,
)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)

    # 데이터
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--categories", nargs="*", default=None, help="사용할 COCO 카테고리 이름. 미지정=전체")
    p.add_argument("--min-area", type=float, default=400.0)
    p.add_argument("--negative-ratio", type=float, default=0.25, help="hard negative 샘플 비율")
    p.add_argument("--max-instances", type=int, default=20)
    p.add_argument("--val-fraction", type=float, default=0.1)

    # 모델
    p.add_argument("--model-id", default="facebook/sam3")
    p.add_argument(
        "--image-size",
        type=int,
        default=1008,
        help="입력 해상도. 1008 이 사전학습 해상도이고 낮추면 정확도가 떨어진다. "
        "메모리가 부족할 때만 560/728 등으로 내린다 (14의 배수 권장).",
    )
    p.add_argument("--dtype", default="float32", choices=["float32", "bfloat16", "float16"])

    # LoRA
    p.add_argument(
        "--lora-stages",
        nargs="+",
        default=["detr_encoder", "detr_decoder", "mask_decoder"],
        choices=["vision", "text", "geometry", "detr_encoder", "detr_decoder", "mask_decoder"],
    )
    p.add_argument("--lora-projections", nargs="+", default=["q_proj", "v_proj"])
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--no-modules-to-save", action="store_true", help="box/presence head 를 얼린다")

    # 학습
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--grad-accum", type=int, default=1)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--head-lr", type=float, default=1e-5, help="modules_to_save head 용 별도 lr")
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--warmup-ratio", type=float, default=0.05)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=10)
    p.add_argument("--device", default=None, help="cuda / mps / cpu. 미지정시 자동")
    p.add_argument("--max-steps", type=int, default=None, help="스모크 테스트용 스텝 제한")

    p.add_argument("--output-dir", type=Path, default=Path("runs/sam3-lora"))
    p.add_argument(
        "--tiny-debug",
        action="store_true",
        help="facebook/sam3 대신 축소 랜덤 초기화 모델로 학습 루프만 검증한다 (gated 접근 불필요)",
    )
    return p.parse_args()


def mask_size_for(image_size: int) -> int:
    """모델이 마스크를 출력하는 해상도.

    ViT patch 14 로 image_size/14 그리드가 나오고, FPN 의 최대 scale_factor 4.0
    레벨이 pixel decoder 의 최종 해상도가 된다. 즉 image_size * 4 / 14.
    (1008 → 288 로 Sam3ImageProcessor 기본값과 일치한다.)
    GT 마스크를 이 해상도로 렌더링하면 손실 계산 시 리샘플링이 없다.
    """
    return image_size * 4 // 14


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
    torch.manual_seed(args.seed)

    from transformers import Sam3Config, Sam3Model, Sam3Processor
    from peft import get_peft_model

    device = pick_device(args.device)
    dtype = getattr(torch, args.dtype)
    print(f"device={device} dtype={dtype}")

    # ---------------- 모델 & 프로세서 ----------------
    # GT 마스크를 모델 출력 해상도와 똑같이 렌더링해 손실에서 리샘플링을 없앤다.
    mask_size = mask_size_for(args.image_size)

    if args.tiny_debug:
        from smoke_test import build_processor, tiny_config

        print("[tiny-debug] 축소 랜덤 초기화 모델 사용 — 손실 값에 의미 없음")
        model = Sam3Model(tiny_config()).to(dtype)
        processor = build_processor()
        args.image_size = processor.image_processor.size["height"]
        mask_size = processor.image_processor.mask_size["height"]
    else:
        config = Sam3Config.from_pretrained(args.model_id)
        if args.image_size != config.image_size:
            # image_size 는 property 라서 vision/backbone config 까지 전파된다.
            config.image_size = args.image_size
            print(f"입력 해상도를 {args.image_size} 로 변경 (사전학습 1008 대비 정확도 손실 가능)")

        model = Sam3Model.from_pretrained(args.model_id, config=config, dtype=dtype)
        processor = Sam3Processor.from_pretrained(
            args.model_id,
            size={"height": args.image_size, "width": args.image_size},
            mask_size={"height": mask_size, "width": mask_size},
        )
    print(f"입력 {args.image_size}px → 마스크 출력 {mask_size}px")

    # LoRA 이전에 전체를 얼린다. PEFT 가 알아서 하지만 명시하는 편이 안전하다.
    model.requires_grad_(False)

    lora_config = build_lora_config(
        model,
        stages=tuple(args.lora_stages),
        projections=tuple(args.lora_projections),
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        modules_to_save=None if args.no_modules_to_save else DEFAULT_MODULES_TO_SAVE,
    )
    print(f"LoRA 대상 모듈 {len(lora_config.target_modules)}개 (stages={args.lora_stages})")

    model = get_peft_model(model, lora_config)
    model.to(device)
    print(summarize_trainable(model))

    # LoRA 는 fp32 로 두는 게 안정적이다. 베이스 가중치만 저정밀도로 유지한다.
    if dtype != torch.float32:
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.data = param.data.float()

    # ---------------- 데이터 ----------------
    dataset = CocoConceptSegmentation(
        images_dir=args.images_dir.expanduser(),
        annotations_path=args.annotations.expanduser(),
        processor=processor,
        categories=args.categories,
        min_area=args.min_area,
        negative_ratio=args.negative_ratio,
        max_instances=args.max_instances,
        seed=args.seed,
    )
    if len(dataset) == 0:
        raise SystemExit("샘플이 0개입니다. --categories / --min-area 를 확인하세요.")

    n_val = max(1, int(len(dataset) * args.val_fraction)) if args.val_fraction > 0 else 0
    n_train = len(dataset) - n_val
    if n_val:
        train_set, val_set = random_split(
            dataset, [n_train, n_val], generator=torch.Generator().manual_seed(args.seed)
        )
    else:
        train_set, val_set = dataset, None
    print(f"샘플: train {len(train_set)} / val {len(val_set) if val_set else 0} (전체 {len(dataset)})")

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=collate_fn,
        drop_last=False,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )
        if val_set
        else None
    )

    # ---------------- 손실 / 옵티마이저 ----------------
    criterion = Sam3SetCriterion()

    lora_params, head_params = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        (lora_params if "lora_" in name else head_params).append(param)

    optimizer = torch.optim.AdamW(
        [
            {"params": lora_params, "lr": args.lr},
            # 사전학습된 head 를 통째로 학습하므로 lr 을 낮춰 파괴를 막는다.
            {"params": head_params, "lr": args.head_lr},
        ],
        weight_decay=args.weight_decay,
    )

    steps_per_epoch = math.ceil(len(train_loader) / args.grad_accum)
    total_steps = steps_per_epoch * args.epochs
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    warmup_steps = max(1, int(total_steps * args.warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1 + math.cos(math.pi * min(1.0, progress)))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "args.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2, ensure_ascii=False)
    )

    def run_batch(batch: dict) -> dict[str, torch.Tensor]:
        outputs = model(
            pixel_values=batch["pixel_values"].to(device, dtype=dtype),
            input_ids=batch["input_ids"].to(device),
            attention_mask=batch["attention_mask"].to(device),
        )
        # 손실은 fp32 로 계산한다. bf16 에서 focal/dice 의 log 항이 불안정하다.
        outputs.pred_logits = outputs.pred_logits.float()
        outputs.pred_boxes = outputs.pred_boxes.float()
        outputs.pred_masks = outputs.pred_masks.float()
        outputs.presence_logits = outputs.presence_logits.float()
        return criterion(outputs, batch["targets"])

    # ---------------- 학습 루프 ----------------
    global_step = 0
    history = []
    for epoch in range(args.epochs):
        model.train()
        running: dict[str, float] = {}
        seen = 0
        t0 = time.time()

        for it, batch in enumerate(train_loader):
            losses = run_batch(batch)
            (losses["loss"] / args.grad_accum).backward()

            seen += 1
            for k, v in losses.items():
                running[k] = running.get(k, 0.0) + v.item()

            if (it + 1) % args.grad_accum == 0 or (it + 1) == len(train_loader):
                torch.nn.utils.clip_grad_norm_(
                    [p for p in model.parameters() if p.requires_grad], args.max_grad_norm
                )
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

                if global_step % args.log_every == 0:
                    parts = " ".join(f"{k.replace('loss_', '')}={running[k] / seen:.3f}" for k in sorted(running))
                    print(
                        f"epoch {epoch} step {global_step}/{total_steps} "
                        f"lr={scheduler.get_last_lr()[0]:.2e} {parts}"
                    )

                if args.max_steps and global_step >= args.max_steps:
                    break

        # max_steps 로 epoch 을 잘랐을 수 있으므로 실제 처리한 배치 수로 나눈다.
        line = {
            "epoch": epoch,
            "train_loss": round(running.get("loss", 0.0) / max(1, seen), 4),
            "sec": round(time.time() - t0, 1),
        }

        # ---------------- 검증 ----------------
        if val_loader:
            model.eval()
            val_running: dict[str, float] = {}
            with torch.no_grad():
                for batch in val_loader:
                    for k, v in run_batch(batch).items():
                        val_running[k] = val_running.get(k, 0.0) + v.item()
            line["val_loss"] = round(val_running.get("loss", 0.0) / len(val_loader), 4)
            line.update(
                {f"val_{k}": round(v / len(val_loader), 4) for k, v in val_running.items() if k != "loss"}
            )

        print(f"[epoch {epoch}] " + " ".join(f"{k}={v}" for k, v in line.items()))
        history.append(line)

        ckpt = args.output_dir / f"epoch-{epoch}"
        model.save_pretrained(ckpt)          # adapter_model.safetensors + adapter_config.json
        processor.save_pretrained(ckpt)      # 추론 시 동일 해상도/토크나이저 재현용
        print(f"저장: {ckpt}")

        if args.max_steps and global_step >= args.max_steps:
            break

    (args.output_dir / "history.json").write_text(json.dumps(history, indent=2))
    model.save_pretrained(args.output_dir / "final")
    processor.save_pretrained(args.output_dir / "final")
    print(f"완료. 최종 어댑터: {args.output_dir / 'final'}")


if __name__ == "__main__":
    main()
