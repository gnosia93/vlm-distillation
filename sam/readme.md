### Phase 1: 스모크 테스트 (모델 로드 + 추론 확인) ###

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
