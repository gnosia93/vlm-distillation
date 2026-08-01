"""SAM3 에 LoRA 를 어디에 붙일지 결정하는 헬퍼.

SAM3 는 서브모듈이 많고 (ViT backbone → FPN neck → geometry encoder →
DETR encoder → DETR decoder → mask decoder) 각각 attention 이름이 다르다.
`target_modules=["q_proj","v_proj"]` 처럼 suffix 만 주면 텍스트 인코더까지
전부 잡히므로, 어느 스테이지에 붙일지 명시적으로 고르게 한다.

실측한 모듈 경로 (transformers 5.14.1):
  vision_encoder.backbone.layers.{i}.attention.{q,k,v,o}_proj   ViT RoPE self-attn
  text_encoder.text_model.encoder.layers.{i}.self_attn.{q,k,v,out}_proj  CLIP
  geometry_encoder.layers.{i}.{self_attn,cross_attn}.{q,k,v,o}_proj
  detr_encoder.layers.{i}.{self_attn,cross_attn}.{q,k,v,o}_proj
  detr_decoder.layers.{i}.{self_attn,text_cross_attn,vision_cross_attn}.{q,k,v,o}_proj
  mask_decoder.prompt_cross_attn.{q,k,v,o}_proj
"""

from __future__ import annotations

import re

import torch.nn as nn

# 스테이지 이름 → 모듈 경로 정규식
STAGE_PATTERNS: dict[str, str] = {
    "vision": r"^vision_encoder\.backbone\.layers\.\d+\.attention\.(q|k|v|o)_proj$",
    "text": r"^text_encoder\.text_model\.encoder\.layers\.\d+\.self_attn\.(q|k|v|out)_proj$",
    "geometry": r"^geometry_encoder\.layers\.\d+\.(self_attn|cross_attn)\.(q|k|v|o)_proj$",
    "detr_encoder": r"^detr_encoder\.layers\.\d+\.(self_attn|cross_attn)\.(q|k|v|o)_proj$",
    "detr_decoder": (
        r"^detr_decoder\.layers\.\d+\.(self_attn|text_cross_attn|vision_cross_attn)\.(q|k|v|o)_proj$"
    ),
    "mask_decoder": r"^mask_decoder\.prompt_cross_attn\.(q|k|v|o)_proj$",
}

# 기본 프리셋: 개념 이해와 위치추정을 담당하는 DETR 스택 + 마스크 decoder 를 적응시킨다.
# ViT backbone 은 이미 강력하고 파라미터가 많아(0.9B 중 대부분) 기본에서는 제외한다.
# 도메인이 크게 다르면(의료/위성/적외선) "vision" 을 추가하는 게 효과가 크다.
DEFAULT_STAGES = ("detr_encoder", "detr_decoder", "mask_decoder")

# LoRA 로는 표현이 안 되는 head 들. full fine-tune 대상으로 함께 저장한다.
# box_head / presence_head 는 출력 차원이 4, 1 이라 저랭크 근사가 의미 없고,
# 새 도메인에서 스케일이 크게 바뀌므로 통째로 학습하는 편이 안정적이다.
DEFAULT_MODULES_TO_SAVE = (
    "detr_decoder.box_head",
    "detr_decoder.presence_head",
    "dot_product_scoring.query_proj",
    "dot_product_scoring.text_proj",
)


def find_lora_targets(
    model: nn.Module,
    stages: tuple[str, ...] | list[str] = DEFAULT_STAGES,
    projections: tuple[str, ...] = ("q_proj", "v_proj"),
) -> list[str]:
    """`stages` 에 해당하는 attention projection 모듈 이름을 모아 반환한다.

    Args:
        model: `Sam3Model` (PEFT 로 감싸기 전).
        stages: `STAGE_PATTERNS` 의 키들.
        projections: 붙일 projection. q/v 만 붙이는 게 표준이고, 파라미터를 더
            쓸 수 있으면 ("q_proj","k_proj","v_proj","o_proj") 로 늘린다.
            CLIP 텍스트 인코더는 o_proj 대신 out_proj 라는 이름을 쓴다.
    """
    unknown = set(stages) - set(STAGE_PATTERNS)
    if unknown:
        raise ValueError(f"알 수 없는 stage: {sorted(unknown)}. 가능한 값: {sorted(STAGE_PATTERNS)}")

    patterns = [re.compile(STAGE_PATTERNS[s]) for s in stages]
    wanted = set(projections)
    if "o_proj" in wanted:
        wanted.add("out_proj")  # CLIP 이름 호환

    targets = [
        name
        for name, module in model.named_modules()
        if isinstance(module, nn.Linear)
        and name.rsplit(".", 1)[-1] in wanted
        and any(p.match(name) for p in patterns)
    ]
    if not targets:
        raise RuntimeError(
            f"stages={stages}, projections={projections} 에 해당하는 모듈이 없습니다. "
            "transformers 버전이 바뀌어 모듈 경로가 달라졌을 수 있습니다."
        )
    return targets


def build_lora_config(
    model: nn.Module,
    stages: tuple[str, ...] | list[str] = DEFAULT_STAGES,
    projections: tuple[str, ...] = ("q_proj", "v_proj"),
    r: int = 16,
    lora_alpha: int = 32,
    lora_dropout: float = 0.05,
    modules_to_save: tuple[str, ...] | list[str] | None = DEFAULT_MODULES_TO_SAVE,
):
    from peft import LoraConfig

    return LoraConfig(
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        bias="none",
        target_modules=find_lora_targets(model, stages, projections),
        modules_to_save=list(modules_to_save) if modules_to_save else None,
    )


def summarize_trainable(model: nn.Module) -> str:
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return f"trainable {trainable:,} / total {total:,} ({100 * trainable / total:.3f}%)"
