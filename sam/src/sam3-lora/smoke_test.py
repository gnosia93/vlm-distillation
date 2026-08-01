#!/usr/bin/env python
"""facebook/sam3 (0.9B, gated) 다운로드 없이 파이프라인 전체를 검증한다.

축소된 랜덤 초기화 Sam3Config 로 모델을 만들어서
데이터셋 → collate → LoRA → forward → 손실 → backward → 저장/로드까지 확인한다.
gated repo 접근 권한이 없거나 GPU 가 없어도 돌아간다.

    python smoke_test.py --images-dir ~/fiftyone/coco-2017/validation/data \
                         --annotations ~/fiftyone/coco-2017/validation/labels.json
"""

from __future__ import annotations

import argparse
import tempfile
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from sam3_lora import (
    DEFAULT_MODULES_TO_SAVE,
    CocoConceptSegmentation,
    Sam3SetCriterion,
    build_lora_config,
    collate_fn,
    find_lora_targets,
    summarize_trainable,
)

IMAGE_SIZE = 112
MASK_SIZE = 32


def tiny_config():
    """실제 SAM3 구조를 그대로 유지하면서 폭/깊이만 줄인 config."""
    from transformers import CLIPTextConfig
    from transformers.models.sam3.configuration_sam3 import (
        Sam3Config,
        Sam3DETRDecoderConfig,
        Sam3DETREncoderConfig,
        Sam3GeometryEncoderConfig,
        Sam3MaskDecoderConfig,
        Sam3VisionConfig,
        Sam3ViTConfig,
    )

    return Sam3Config(
        vision_config=Sam3VisionConfig(
            backbone_config=Sam3ViTConfig(
                hidden_size=64,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=2,
                image_size=IMAGE_SIZE,
                patch_size=14,
                window_size=4,
                global_attn_indexes=[1],
                pretrain_image_size=IMAGE_SIZE,
            ),
            fpn_hidden_size=32,
        ),
        text_config=CLIPTextConfig(
            hidden_size=32,
            intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=2,
            projection_dim=32,
            max_position_embeddings=32,
        ),
        geometry_encoder_config=Sam3GeometryEncoderConfig(
            hidden_size=32, num_attention_heads=2, intermediate_size=64
        ),
        detr_encoder_config=Sam3DETREncoderConfig(
            hidden_size=32, num_layers=1, num_attention_heads=2, intermediate_size=64
        ),
        detr_decoder_config=Sam3DETRDecoderConfig(
            hidden_size=32, num_layers=2, num_queries=20, num_attention_heads=2, intermediate_size=64
        ),
        mask_decoder_config=Sam3MaskDecoderConfig(hidden_size=32, num_attention_heads=2),
    )


def build_processor():
    """축소 해상도용 Sam3Processor. 토크나이저는 CLIP 것을 그대로 쓴다."""
    from transformers import AutoTokenizer, Sam3ImageProcessor, Sam3Processor

    image_processor = Sam3ImageProcessor(
        size={"height": IMAGE_SIZE, "width": IMAGE_SIZE},
        mask_size={"height": MASK_SIZE, "width": MASK_SIZE},
    )
    tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
    return Sam3Processor(image_processor=image_processor, tokenizer=tokenizer)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--images-dir", required=True, type=Path)
    p.add_argument("--annotations", required=True, type=Path)
    p.add_argument("--steps", type=int, default=3)
    args = p.parse_args()

    from transformers import Sam3Model
    from peft import PeftModel, get_peft_model

    torch.manual_seed(0)

    print("== 1. 모델 생성 (축소 랜덤 초기화) ==")
    model = Sam3Model(tiny_config())
    print(f"   params {sum(x.numel() for x in model.parameters()) / 1e6:.2f}M")
    # 8번 라운드트립 검증용. 실제로는 from_pretrained 로 받는 사전학습 가중치에 해당한다.
    base_state = {k: v.detach().clone() for k, v in model.state_dict().items()}

    print("== 2. LoRA 대상 확인 ==")
    for stage in ["vision", "text", "geometry", "detr_encoder", "detr_decoder", "mask_decoder"]:
        n = len(find_lora_targets(model, (stage,), ("q_proj", "k_proj", "v_proj", "o_proj")))
        print(f"   {stage:14s} {n:3d} modules")

    print("== 3. 데이터셋 ==")
    processor = build_processor()
    dataset = CocoConceptSegmentation(
        images_dir=args.images_dir.expanduser(),
        annotations_path=args.annotations.expanduser(),
        processor=processor,
        categories=["person", "car", "chair"],
        negative_ratio=0.3,
        max_instances=8,
    )
    n_neg = sum(1 for s in dataset.samples if not s.boxes)
    print(f"   {len(dataset)} samples ({n_neg} hard negatives)")
    assert n_neg > 0, "hard negative 가 생성되지 않았다"

    sample = dataset[0]
    print(
        f"   sample: pixel_values={tuple(sample['pixel_values'].shape)} "
        f"boxes={tuple(sample['boxes'].shape)} masks={tuple(sample['masks'].shape)} text={sample['text']!r}"
    )
    assert sample["masks"].shape[-2:] == (MASK_SIZE, MASK_SIZE)
    assert sample["boxes"].min() >= 0 and sample["boxes"].max() <= 1, "박스가 [0,1] 정규화되지 않았다"
    assert sample["masks"].sum() > 0, "GT 마스크가 비어 있다 (polygon 렌더링 실패)"

    loader = DataLoader(dataset, batch_size=2, shuffle=True, collate_fn=collate_fn)

    print("== 4. LoRA 부착 ==")
    lora_config = build_lora_config(model, r=4, lora_alpha=8, modules_to_save=DEFAULT_MODULES_TO_SAVE)
    peft_model = get_peft_model(model, lora_config)
    print(f"   {summarize_trainable(peft_model)}")
    saved = {n.split(".modules_to_save")[0] for n, _ in peft_model.named_modules() if ".modules_to_save" in n}
    print(f"   modules_to_save 적용: {len(saved)}개")
    assert len(saved) == len(DEFAULT_MODULES_TO_SAVE), f"기대 {len(DEFAULT_MODULES_TO_SAVE)}, 실제 {len(saved)}"

    print("== 5. forward + 손실 + backward ==")
    criterion = Sam3SetCriterion(mask_points=256)
    optimizer = torch.optim.AdamW([p for p in peft_model.parameters() if p.requires_grad], lr=1e-3)

    before = {
        n: p.detach().clone() for n, p in peft_model.named_parameters() if p.requires_grad
    }

    saw_negative_batch = False
    for step, batch in enumerate(loader):
        if step >= args.steps:
            break
        outputs = peft_model(
            pixel_values=batch["pixel_values"],
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
        )
        assert outputs.pred_masks.shape[-2:] == (MASK_SIZE, MASK_SIZE), outputs.pred_masks.shape
        losses = criterion(outputs, batch["targets"])
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_([p for p in peft_model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        optimizer.zero_grad()

        if any(len(t["boxes"]) == 0 for t in batch["targets"]):
            saw_negative_batch = True

        print(
            "   step %d  " % step
            + " ".join(f"{k.replace('loss_', '')}={v.item():.3f}" for k, v in sorted(losses.items()))
        )
        assert torch.isfinite(losses["loss"]), "loss 가 NaN/Inf"

    print("== 6. 파라미터가 실제로 갱신됐는지 ==")
    changed = sum(
        1 for n, p in peft_model.named_parameters() if p.requires_grad and not torch.equal(p.detach(), before[n])
    )
    print(f"   {changed}/{len(before)} trainable 텐서 변경")
    assert changed > 0, "학습이 파라미터를 바꾸지 않았다"

    # lora_B 는 0 초기화라 첫 스텝 전 grad 가 0 인 경로가 있을 수 있다. lora_A 는 반드시 변해야 한다.
    lora_a_changed = any(
        "lora_A" in n and not torch.equal(p.detach(), before[n])
        for n, p in peft_model.named_parameters()
        if p.requires_grad
    )
    assert lora_a_changed, "LoRA A 행렬에 gradient 가 흐르지 않았다"

    print("== 7. hard negative 손실 동작 ==")
    # 인스턴스가 0개인 배치에서도 loss 가 유한하고 presence 가 학습되는지 별도로 확인
    empty_batch = collate_fn([{**dataset[i], "boxes": torch.zeros(0, 4), "masks": torch.zeros(0, MASK_SIZE, MASK_SIZE)} for i in range(2)])
    outputs = peft_model(
        pixel_values=empty_batch["pixel_values"],
        input_ids=empty_batch["input_ids"],
        attention_mask=empty_batch["attention_mask"],
    )
    empty_losses = criterion(outputs, empty_batch["targets"])
    print("   " + " ".join(f"{k.replace('loss_', '')}={v.item():.3f}" for k, v in sorted(empty_losses.items())))
    assert torch.isfinite(empty_losses["loss"])
    assert empty_losses["loss_bbox"].item() == 0.0, "빈 target 에서 box loss 가 0 이 아니다"
    assert empty_losses["loss_presence"].item() > 0, "presence loss 가 계산되지 않았다"
    empty_losses["loss"].backward()  # 빈 target 에서도 그래프가 살아 있어야 한다
    print(f"   (배치 중 negative 포함 여부: {saw_negative_batch})")

    print("== 8. 어댑터 저장/로드 라운드트립 ==")
    with tempfile.TemporaryDirectory() as tmp:
        peft_model.save_pretrained(tmp)
        files = sorted(p.name for p in Path(tmp).iterdir())
        print(f"   저장된 파일: {files}")

        peft_model.eval()
        forward_kwargs = {
            "pixel_values": batch["pixel_values"],
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
        }
        with torch.no_grad():
            ref = peft_model(**forward_kwargs).pred_masks

        # 실제 시나리오와 동일하게: 사전학습 베이스 가중치 + 어댑터.
        # 여기서는 "사전학습" 대신 학습 시작 시점의 베이스 가중치를 저장해 두고 쓴다.
        base = Sam3Model(tiny_config())
        base.load_state_dict(base_state, strict=True)
        reloaded = PeftModel.from_pretrained(base, tmp).eval()
        with torch.no_grad():
            got = reloaded(**forward_kwargs).pred_masks
        max_diff = (ref - got).abs().max().item()
        print(f"   재로드 후 출력 최대 차이: {max_diff:.2e}")
        assert max_diff < 1e-5, f"저장/로드 후 출력이 달라졌다 ({max_diff})"

        print("== 9. merge_and_unload ==")
        base2 = Sam3Model(tiny_config())
        base2.load_state_dict(base_state, strict=True)
        merged = PeftModel.from_pretrained(base2, tmp).merge_and_unload().eval()
        with torch.no_grad():
            merged_masks = merged(**forward_kwargs).pred_masks
        merge_diff = (ref - merged_masks).abs().max().item()
        print(f"   merge 후 출력 최대 차이: {merge_diff:.2e}")
        assert merge_diff < 1e-3, f"merge 결과가 어댑터 적용 결과와 다르다 ({merge_diff})"

    print("\n모든 검증 통과.")


if __name__ == "__main__":
    main()
