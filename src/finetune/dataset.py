"""
InternVL3 video-SFT dataset.

레이아웃 (로컬, S3와 동일 구조):
  <data_root>/finevideo/sports/manifest.jsonl
  <data_root>/finevideo/sports/<video_id>/frames/frames.json
  <data_root>/finevideo/sports/<video_id>/frames/frame_XXX.jpg
  <data_root>/finevideo/sports/<video_id>/inference/<run_id>.json   # teacher 라벨

각 학습 샘플 = (16 프레임) + (prompt) -> (teacher answer)
"""
import os
import json
import glob
from typing import List, Dict

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as T
from torchvision.transforms.functional import InterpolationMode

# --- InternVL 이미지 정규화 상수 ---
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# --- InternVL 특수 토큰 ---
IMG_START_TOKEN = "<img>"
IMG_END_TOKEN = "</img>"
IMG_CONTEXT_TOKEN = "<IMG_CONTEXT>"
IGNORE_INDEX = -100

# 기본 시스템 메시지 (필요시 교체)
SYSTEM_MESSAGE = "당신은 영상을 보고 한국어로 정확하게 설명하는 유능한 어시스턴트입니다."

def build_transform(input_size: int = 448):
    """스펙: 원본 크기 무관 448x448 리사이즈 = 프레임당 타일 1개."""
    return T.Compose([
        T.Lambda(lambda img: img.convert("RGB")),
        T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])

def _pick_latest_inference(inf_dir: str):
    """영상당 inference json이 여러 개면 created_at 기준 최신 1개."""
    files = glob.glob(os.path.join(inf_dir, "*.json"))
    if not files:
        return None
    best, best_key = None, None
    for f in files:
        try:
            with open(f, "r", encoding="utf-8") as fp:
                data = json.load(fp)
        except Exception:
            continue
        key = data.get("created_at") or str(os.path.getmtime(f))
        if best_key is None or key > best_key:
            best, best_key = data, key
    return best

class InternVLVideoSFTDataset(Dataset):
    def __init__(self, data_root: str, split_prefix: str, tokenizer,
                 num_image_token: int, input_size: int = 448,
                 max_length: int = 5120):
        self.data_root = data_root
        self.split_dir = os.path.join(data_root, split_prefix)
        self.tokenizer = tokenizer
        self.num_image_token = num_image_token
        self.max_length = max_length
        self.transform = build_transform(input_size)

        self.samples = self._build_index()
        if len(self.samples) == 0:
            raise RuntimeError(f"학습 샘플이 없습니다: {self.split_dir}")
        print(f"[dataset] usable samples: {len(self.samples)}")

    def _build_index(self) -> List[Dict]:
        manifest = os.path.join(self.split_dir, "manifest.jsonl")
        samples = []
        with open(manifest, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                vid = rec["video_id"]
                vdir = os.path.join(self.split_dir, vid)

                frames_json = os.path.join(vdir, "frames", "frames.json")
                inf_dir = os.path.join(vdir, "inference")
                if not os.path.exists(frames_json) or not os.path.isdir(inf_dir):
                    continue

                inf = _pick_latest_inference(inf_dir)
                if not inf:
                    continue
                prompt = (inf.get("prompt") or "").strip()
                answer = (inf.get("answer") or "").strip()
                if not prompt or not answer:
                    continue

                with open(frames_json, "r", encoding="utf-8") as fp:
                    fj = json.load(fp)
                frame_keys = fj.get("frames", [])
                if not frame_keys:
                    continue
                # frames.json의 경로는 S3-relative key → 로컬 절대경로로
                frame_paths = [os.path.join(self.data_root, k) for k in frame_keys]
                if not all(os.path.exists(p) for p in frame_paths):
                    continue

                samples.append({
                    "video_id": vid,
                    "frame_paths": frame_paths,
                    "prompt": prompt,
                    "answer": answer,
                })
        return samples

    def _expand_image_tokens(self, text: str) -> str:
        img_block = (IMG_START_TOKEN
                     + IMG_CONTEXT_TOKEN * self.num_image_token
                     + IMG_END_TOKEN)
        return text.replace("<image>", img_block)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        frame_paths = s["frame_paths"]
        num_frames = len(frame_paths)

        # 1) 프레임 로드 & 448x448 (프레임당 1타일) → (num_frames, 3, 448, 448)
        pixel_values = torch.stack([self.transform(Image.open(p)) for p in frame_paths])

        # 2) 프레임 플레이스홀더 + 프롬프트
        video_prefix = "".join(f"Frame{i+1}: <image>\n" for i in range(num_frames))
        question = video_prefix + s["prompt"]
        question = self._expand_image_tokens(question)  # <image> -> <img>...256개...</img>

        # 3) ChatML(Qwen2.5) 템플릿. prefix까지는 마스킹, answer만 학습.
        prefix = (f"<|im_start|>system\n{SYSTEM_MESSAGE}<|im_end|>\n"
                  f"<|im_start|>user\n{question}<|im_end|>\n"
                  f"<|im_start|>assistant\n")
        full = prefix + s["answer"] + "<|im_end|>\n"

        prefix_ids = self.tokenizer(prefix, add_special_tokens=False).input_ids
        full_ids = self.tokenizer(full, add_special_tokens=False).input_ids

        # 길이 제한 (거의 없지만 방어)
        full_ids = full_ids[: self.max_length]
        labels = list(full_ids)
        n_prefix = min(len(prefix_ids), len(full_ids))
        for i in range(n_prefix):
            labels[i] = IGNORE_INDEX  # 이미지 토큰 + 프롬프트 = loss 제외

        input_ids = torch.tensor(full_ids, dtype=torch.long)
        labels = torch.tensor(labels, dtype=torch.long)
        image_flags = torch.ones(num_frames, dtype=torch.long)  # 프레임당 1타일

        return {
            "input_ids": input_ids,
            "labels": labels,
            "pixel_values": pixel_values,   # (num_frames, 3, 448, 448)
            "image_flags": image_flags,     # (num_frames,)
        }

class InternVLCollator:
    """배치: input_ids/labels 우측 패딩, pixel_values/image_flags는 타일 축으로 concat."""
    def __init__(self, pad_token_id: int, pixel_dtype=torch.bfloat16):
        self.pad_token_id = pad_token_id
        self.pixel_dtype = pixel_dtype

    def __call__(self, batch):
        maxlen = max(b["input_ids"].size(0) for b in batch)
        input_ids, labels, attn = [], [], []
        for b in batch:
            ids = b["input_ids"]
            lab = b["labels"]
            pad = maxlen - ids.size(0)
            if pad > 0:
                ids = torch.cat([ids, torch.full((pad,), self.pad_token_id, dtype=torch.long)])
                lab = torch.cat([lab, torch.full((pad,), IGNORE_INDEX, dtype=torch.long)])
            mask = (ids != self.pad_token_id).long()
            # 패딩이 실제 pad_token과 겹칠 수 있으니 길이 기반 마스크로 보정
            mask = torch.zeros(maxlen, dtype=torch.long)
            mask[: b["input_ids"].size(0)] = 1
            input_ids.append(ids); labels.append(lab); attn.append(mask)

        return {
            "input_ids": torch.stack(input_ids),
            "labels": torch.stack(labels),
            "attention_mask": torch.stack(attn),
            "pixel_values": torch.cat([b["pixel_values"] for b in batch], dim=0).to(self.pixel_dtype),
            "image_flags": torch.cat([b["image_flags"] for b in batch], dim=0),
        }
