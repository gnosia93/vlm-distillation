## SAM-3 평가 및 파인튜닝 ##

### 1. facebook/sam3 모델 억세스 권한 획득 ###

facebook/sam3 모델의 경우 gated model 인 관계로, https://huggingface.co/facebook/sam3 방문해서 sam3 모델에 대한 억세스 권한을 요청한다.

### 2. 모델 스모크 테스트 ###
```
python -m venv ~/sam3-venv
source ~/sam3-venv/bin/activate

python -m pip install -U pip
python -m pip install -U "transformers>=4.57" accelerate torch torchvision pillow matplotlib requests

export HF_TOKEN=xxxxx
git clone https://github.com/gnosia93/vlm-distillation.git
cd ~/vlm-distillation/sam/src
python sam3.py
```
[결과]
```
'notebook' 객체 1개 탐지
박스: tensor([[-1.0039e+00,  3.6772e-01,  2.8926e+02,  4.2335e+02]], device='mps:0')
점수: tensor([0.9606], device='mps:0')
저장됨: result.png
```
open 으로 결과 이미지를 확인한다. 
```
open original.jpg result.png
```
![](https://github.com/gnosia93/vlm-distillation/blob/main/images/sam-result.png)


### 3. 고객 데이터로 성능 보기 (폴더 일괄 + 시각화) ###
모델이 우리가 원하는 객체를 제대로 세그멘테이션했는지 확인한다.
```
# sam3_eval.py — 유치원 프레임 폴더에 프롬프트 돌려 결과 시각화/집계
import os, glob, torch, numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from transformers import Sam3Model, Sam3Processor

FRAME_DIR = "/path/to/kindergarten_frames"   # ← 프레임 폴더
OUT_DIR   = "/path/to/sam3_out"
PROMPT    = "child"
THRESH    = 0.5
os.makedirs(OUT_DIR, exist_ok=True)

model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
processor = Sam3Processor.from_pretrained("facebook/sam3")

paths = sorted(glob.glob(os.path.join(FRAME_DIR, "*.jpg")))
summary = []

for p in paths:
    image = Image.open(p).convert("RGB")
    inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs)
    res = processor.post_process_instance_segmentation(
        outputs, threshold=THRESH, mask_threshold=0.5,
        target_sizes=inputs.get("original_sizes").tolist(),
    )[0]

    n = len(res["masks"])
    scores = [float(s) for s in res.get("scores", [])]
    summary.append((os.path.basename(p), n, scores))

    # 마스크 오버레이 저장 (눈으로 확인용)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.imshow(image)
    for m in res["masks"]:
        m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
        ax.imshow(np.ma.masked_where(m == 0, m), alpha=0.5)
    ax.set_title(f"{os.path.basename(p)}  |  '{PROMPT}' x{n}")
    ax.axis("off")
    fig.savefig(os.path.join(OUT_DIR, f"vis_{os.path.basename(p)}.png"),
                bbox_inches="tight", dpi=100)
    plt.close(fig)

# 집계 리포트
print(f"총 {len(paths)}장 처리")
counts = [n for _, n, _ in summary]
print(f"프레임당 평균 탐지: {np.mean(counts):.2f}, 최소 {min(counts)}, 최대 {max(counts)}")
zero = [f for f, n, _ in summary if n == 0]
print(f"아무것도 못 잡은 프레임: {len(zero)}장  (인식 실패 후보 → 눈으로 확인)")

```
* vis_*.png를 열어 **"아이를 제대로 잡나, 놓치나, 엉뚱한 걸 잡나"**를 눈으로 확인.
* 0개로 나온 프레임이 "인식 실패 후보" → 크롭 테스트(그 아이를 크게 잘라 다시 넣어보기)를 하면 입력(해상도) 문제인지 모델(인코더) 문제인지가 판별됨.

### 4. 파인튜닝 ###

앞 단계에서 "확실히 못 잡는다"가 확인되면 그때 파인튜닝을 수행한다. HF라서 PEFT/LoRA를 detector decoder에 붙이는 식으로 가볍게 시작한다. (라벨 데이터 확보가 전제)

* 프롬프트 실험: "child", "person", "kid", "toddler" 등 여러 프롬프트를 바꿔가며 돌려본다. 특정 프롬프트만 잘 되면 그건 인코더가 아니라 텍스트 정렬 이슈라는 신호이다.
* 점수 계산: 내부적으로 final = pred_logits.sigmoid() * presence_logits.sigmoid()로 최종 점수가 나온다. presence token이 "이 개념이 존재하나"를 판단해줘서 negative(없는 것) 처리에 강함.

#### 진단 결과별 대응 ####
- 크롭하면 잘 잡힘 → 입력(해상도) 문제 → 전처리 개선 (파인튜닝 불필요)
- 일부 프롬프트만 됨 → 텍스트 정렬 이슈 → 프롬프트 교체로 해결 (파인튜닝 불필요)
- 그 외 탐지/마스크 품질 → detector/mask decoder에 LoRA
- 크롭해도 못 잡음 → 인코더 문제 (LoRA, 최후)

> [!NOTE]
> 프롬프트 교체 테스트: 문제 되는 이미지에서 단어만 바꿔가며 테스트.
> 일부 단어만 되면 텍스트 정렬 이슈(프롬프트 교체로 해결), 어떤 단어도 안 되면 입력/인코더 문제.

> [!WARNING]
> 정리
> 인코더 LoRA = 원본 freeze + ViT 선형층에 어댑터만 학습. 붙이는 건 PEFT로 표준적.
> 진짜 난관은 학습 loss: SAM3는 DETR식 손실(헝가리안 매칭)이 필요한데 HF forward가 loss를 바로 안 줌 → 직접 구현하거나 Meta > repo 학습코드 사용.
> 그래서 최후의 수단. 입력·프롬프트·디코더를 먼저 소진하고, 정 안 되면 라벨 데이터 갖춰서 Meta repo 기반으로.

## 레퍼런스 ##

* https://huggingface.co/facebook/sam3 
* https://huggingface.co/settings/gated-repos


