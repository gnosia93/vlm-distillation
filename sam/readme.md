### Phase 1: 스모크 테스트 (모델 로드 + 추론 확인) ###

✅ transformers/torch 환경이 제대로 깔렸나
✅ facebook/sam3 모델이 로드되나 (다운로드, 메모리)
✅ 이미지 + 프롬프트 넣으면 에러 없이 출력(마스크/개수)이 나오나

```
# 설치 (SAM3는 2025-11-19에 추가돼서 최신 transformers 필요)
pip install -U "transformers>=4.57" torch pillow requests matplotlib
# 안 되면 소스 설치: pip install git+https://github.com/huggingface/transformers.git
```
```
# sam3_smoke.py — SAM3가 로드되고 추론이 되는지 확인
import requests, torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
processor = Sam3Processor.from_pretrained("facebook/sam3")

# 샘플 이미지 (나중에 유치원 프레임 경로로 교체)
url = "http://images.cocodataset.org/val2017/000000077595.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

PROMPT = "cat"   # ← 텍스트 프롬프트. 유치원이면 "child" / "person" 등
inputs = processor(images=image, text=PROMPT, return_tensors="pt").to(model.device)

with torch.no_grad():
    outputs = model(**inputs)

results = processor.post_process_instance_segmentation(
    outputs, threshold=0.5, mask_threshold=0.5,
    target_sizes=inputs.get("original_sizes").tolist(),
)[0]

print(f"'{PROMPT}' 객체 {len(results['masks'])}개 탐지")
print("박스:", results.get("boxes"))
print("점수:", results.get("scores"))

```

### Phase 2: 고객 데이터로 성능 보기 (폴더 일괄 + 시각화) ###
✅ 마스크가 유치원 아이를 정확히 잡았는지 ?
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
* vis_*.png를 열어 **"아이를 제대로 잡나, 놓치나, 엉뚱한 걸 잡나"**를 눈으로 봅니다.
* 0개로 나온 프레임이 "인식 실패 후보" → 여기에 앞서 얘기한 크롭 테스트(그 아이를 크게 잘라 다시 넣어보기)를 하면 입력(해상도) 문제인지 모델(인코더) 문제인지가 갈립니다.

### Phase 3: 파인튜닝 ###

Phase 2에서 "확실히 못 잡는다"가 확인되면 그때 파인튜닝. HF라서 PEFT/LoRA를 detector decoder에 붙이는 식으로 가볍게 시작합니다. (라벨 데이터 확보가 전제)

실행 전 알아둘 점

* GPU 필요: 848M + 기본 1008px 해상도라 무겁습니다. macOS 로컬보단 **AWS GPU 인스턴스(g5 권장)**에서 돌린다.
* 해상도 조절 가능: 빠르게 보려면 Sam3Config의 image_size를 낮출 수 있지만 정확도 저하 경고가 있으니, 평가 단계에선 기본값 유지 권장.
* 프롬프트 실험: "child", "person", "kid", "toddler" 등 여러 프롬프트를 바꿔가며 돌려본다. 특정 프롬프트만 잘 되면 그건 인코더가 아니라 텍스트 정렬 이슈라는 신호이다.
* 점수 계산: 내부적으로 final = pred_logits.sigmoid() * presence_logits.sigmoid()로 최종 점수가 나온다. presence token이 "이 개념이 존재하나"를 판단해줘서 negative(없는 것) 처리에 강함.

정리하면 Phase 1으로 "돌아가는지" 확인 → Phase 2로 고객 프레임에서 "잘 잡는지" 눈으로 + 개수로 확인 → 실패 확인되면 Phase 3 파인튜닝..




