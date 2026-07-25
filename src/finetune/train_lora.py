"""
InternVL3-1B LoRA SFT (teacher 라벨 sequence-level distillation).

실행(단일 GPU):
  python train_lora.py --model /models/internvl3-1b --data-root /opt/dlami/nvme/hf-cache/data

실행(멀티 GPU, 예: 4장):
  torchrun --nproc_per_node=4 train_lora.py --model /models/internvl3-1b --data-root ...
"""
import argparse
import torch
from transformers import AutoModel, AutoTokenizer, TrainingArguments, Trainer
from peft import LoraConfig, get_peft_model

from dataset import (
    InternVLVideoSFTDataset, InternVLCollator, IMG_CONTEXT_TOKEN,
)

class InternVLTrainer(Trainer):
    """커스텀 멀티모달 forward에 맞춰 compute_loss 오버라이드."""
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        outputs = model(
            pixel_values=inputs["pixel_values"],
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            image_flags=inputs["image_flags"],
            labels=inputs["labels"],
        )
        loss = outputs.loss
        return (loss, outputs) if return_outputs else loss

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="InternVL3-1B 로컬 경로")
    p.add_argument("--data-root", required=True, help="finevideo/ 를 포함하는 로컬 루트")
    p.add_argument("--split-prefix", default="finevideo/sports")
    p.add_argument("--output-dir", default="./internvl3-1b-lora")
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--grad-accum", type=int, default=8)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    p.add_argument("--lora-dropout", type=float, default=0.05)
    p.add_argument("--max-length", type=int, default=5120)
    return p.parse_args()

def main():
    args = parse_args()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True, use_fast=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModel.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    # 이미지 컨텍스트 토큰 id 설정 (필수)
    model.img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    num_image_token = model.num_image_token  # 448/14 다운샘플 → 256
    print(f"[model] num_image_token per tile = {num_image_token}")

    # 비전 인코더 + projector freeze (LLM에만 LoRA)
    model.vision_model.requires_grad_(False)
    if hasattr(model, "mlp1"):
        model.mlp1.requires_grad_(False)
    model.config.use_cache = False

    # LoRA: LLM attention/MLP 선형층에만
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj"],
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()  # gradient checkpointing + LoRA 호환
    model.print_trainable_parameters()

    dataset = InternVLVideoSFTDataset(
        data_root=args.data_root,
        split_prefix=args.split_prefix,
        tokenizer=tokenizer,
        num_image_token=num_image_token,
        max_length=args.max_length,
    )
    collator = InternVLCollator(pad_token_id=tokenizer.pad_token_id,
                                pixel_dtype=torch.bfloat16)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        bf16=True,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_steps=200,
        save_total_limit=2,
        dataloader_num_workers=4,
        remove_unused_columns=False,   # pixel_values/image_flags 유지
        report_to="none",
        ddp_find_unused_parameters=False,
    )

    trainer = InternVLTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator,
    )
    trainer.train()

    # LoRA 어댑터 + 토크나이저 저장
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)
    print(f"[done] adapter saved to {args.output_dir}")

if __name__ == "__main__":
    main()

