import requests, torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor

model = Sam3Model.from_pretrained("facebook/sam3", device_map="auto")
processor = Sam3Processor.from_pretrained("facebook/sam3")

# 샘플 이미지 (나중에 유치원 프레임 경로로 교체)
#url = "http://images.cocodataset.org/val2017/000000077595.jpg"
url = "https://edu.chosun.com/site/data/img_dir/2023/02/16/2023021601153_0.jpg"
image = Image.open(requests.get(url, stream=True).raw).convert("RGB")

PROMPT = "teacher"   # ← 텍스트 프롬프트. 유치원이면 "child" / "person" 등
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


import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches

masks  = results["masks"]
boxes  = results["boxes"]
scores = results["scores"]

fig, ax = plt.subplots(figsize=(10, 8))
ax.imshow(image)

for i in range(len(masks)):
    # 마스크 오버레이 (반투명 색)
    m = masks[i]
    m = m.cpu().numpy() if hasattr(m, "cpu") else np.asarray(m)
    m = m.astype(bool)
    color = np.random.rand(3)
    overlay = np.zeros((*m.shape, 4))
    overlay[m] = [color[0], color[1], color[2], 0.5]
    ax.imshow(overlay)

    # 박스 + 점수
    box = boxes[i]
    box = box.cpu().numpy() if hasattr(box, "cpu") else np.asarray(box)
    x1, y1, x2, y2 = box
    ax.add_patch(patches.Rectangle((x1, y1), x2 - x1, y2 - y1,
                                   fill=False, edgecolor="lime", linewidth=2))
    ax.text(x1, y1 - 5, f"{float(scores[i]):.2f}",
            color="lime", fontsize=12, weight="bold")

ax.axis("off")
plt.savefig("result.png", bbox_inches="tight", dpi=120)
print("저장됨: result.png")
image.save('original.jpg')
