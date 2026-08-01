## SAM-3 진단 / 평가 ##

SAM 3(Segment Anything Model 3)는 Meta가 개발한 세그멘테이션 파운데이션 모델로, 텍스트로 개념을 지정하면 이미지나 영상에서 그 개념에 해당하는 모든 객체를 찾아 분할하고 추적해 준다.

SAM 3의 핵심 능력은 PCS(Promptable Concept Segmentation, 개념 기반 분할)이다. 입력으로는 이미지 또는 영상과 함께 프롬프트를 받는데, 프롬프트는 child나 yellow school bus 같은 짧은 명사구 형태의 텍스트 개념, 참조용 이미지 예시(exemplar), 또는 점·박스·마스크 같은 기하학적 프롬프트가 될 수 있다. 이에 대해 SAM 3는 픽셀 단위 실루엣인 마스크와 바운딩 박스, 각 객체를 구분하는 인스턴스 ID를 출력하며, 영상의 경우 프레임 간 추적 트랙까지 함께 제공한다. 특히 클릭 하나로 객체 하나를 잡던 기존 방식과 달리, 개념에 맞는 모든 인스턴스를 한 번에 찾아낸다는 점이 특징이다. 예를 들어 child라는 프롬프트를 주면 화면 속 아이 전부를 동시에 분할한다.

![](https://github.com/gnosia93/vlm-distillation/blob/main/images/sam3-arch.png)

- Vision Encoder: 이미지를 특징으로 변환 (Detector·Tracker가 공유)
- Detector: 텍스트/예시 프롬프트로 객체 탐지·분할 (DETR 기반, 박스+분류)
- Tracker: 영상에서 시간적 추적 (SAM2 방식) — 아래 세부
  - Prompt Encoder: 점/박스/마스크 인코딩
  - Memory Bank: 과거 프레임 특징·마스크 저장, Mask Decoder가 참조
  - Feedback Loop: 예측 마스크 → 메모리 뱅크 갱신 → 다음 프레임에 활용

SAM(Segment Anything Model) 은 프롬프트를 주면 이미지·영상에서 대상을 픽셀 단위로 분할하는 Meta의 범용 세그멘테이션 모델로 SAM1(이미지, 클릭) → SAM2(영상, 추적) → SAM3(텍스트 개념, open-vocabulary)로 발전하였다.
* SAM 계열 = "무엇이 어디 있나" 분할·탐지·추적 → 출력이 마스크/박스
* VLM = "무슨 일이 일어나나" 이해·서술 → 출력이 텍스트


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




### 4. 문제 진다 및 식별 ###

(1) "잘 판별 못한다"는 결론이 애초에 어떻게 나온 거냐, 그리고 (2) 그게 인코더 탓인지 어떻게 가려내냐. 순서대로 가야 해요. (2)를 하기 전에 (1)이 먼저거든요.

#### 1. "잘 판별 못한다"는 어떻게 측정된 결과인가 ? ####

* 정량적으로 나왔나?: 정답(사람이 라벨한 마스크/박스)이 있는 테스트셋에 SAM을 돌려서, recall(놓친 아이 비율), precision(오탐), 마스크 IoU, 추적 지표(ID 바뀜 횟수) 같은 숫자로 확인한 건가?
* 정성적(눈으로)인가?: 그냥 몇 장 돌려보고 "어, 아이를 놓치네" 하고 느낀 건가?

대부분은 후자(인상)일 가능성이 크다. 그러면 "정말 못 하는 건지, 프롬프트를 잘못 준 건지, 몇몇 어려운 케이스만 그런 건지"가 불분명하다. 그래서 첫 액션은:

실패 사례 프레임을 모으고, 정답 라벨을 붙여 "무엇을 얼마나 못 맞히는지"를 숫자로 만드는 것.


#### 2. 병목이 어디인지 가려내는 진단 테스트 ####
"못 한다"를 확인했으면, 어느 부품 탓인지를 아래 테스트로 좁힌다.

#### ① 에러 유형 분류 (제일 먼저) ####
실패가 어떤 종류인지 봅니다. 이것만으로도 범인이 좁혀진다.

| 실패 양상 | 유력 원인 |
| :--- | :--- |
| **아이를 아예 못 찾음 (탐지 자체 실패)** | 인코더(지각) 또는 입력(작음/저해상도) |
| **찾긴 했는데 마스크/박스가 엉성** | Detector decoder |
| **한 프레임엔 잡는데 다음 프레임에 놓침/ID 바뀜** | Tracker/memory |
| **엉뚱한 걸 아이로 잡음, 개념 혼동** | VL/텍스트 정렬 |


#### ② 프롬프트 교체 테스트 (VL vs 인코더 구분) ####

같은 이미지에 프롬프트만 바꿔본다 : "child", "person", "kid", "toddler"...  
어떤 프롬프트는 되고 어떤 건 안 됨 → 문제는 VL/텍스트 정렬이지 인코더 지각이 아님. (인코더는 봤는데 텍스트 매칭이 안 된 것)
어떤 프롬프트를 줘도 못 찾음 → 인코더나 입력 쪽 의심.

#### ③ 크롭/확대 테스트 (입력 vs 인코더 구분) — 매우 중요 ####

문제의 아이를 크게 크롭해서 고해상도로 넣어본다.  
크롭하면 잘 잡음 → 인코더 가중치 문제 아님! 입력(해상도/작은 객체) 문제. → 전처리(크롭/타일링/해상도↑)로 해결.    
크롭해도 여전히 못 잡음 → 진짜 인코더 지각 문제일 가능성.    
이 테스트가 "입력 문제냐 인코더 문제냐"를 가장 손쉽게 가른다. 유치원 CCTV는 아이가 작게 잡히는 경우가 많아서, 의외로 여기서 입력 문제로 판명되는 경우가 많다.

#### ④ 특징/어텐션 시각화 (인코더가 보고는 있나) ####
인코더의 특징 맵이나 어텐션을 시각화해서, 아이가 있는 영역에 인코더가 반응하는지 본다.   

아이 영역이 특징에서 밋밋/무반응 → 인코더가 지각을 못 하는 것.   
반응은 하는데 최종 출력이 틀림 → 뒷단(decoder/VL) 문제.   

#### ⑤ 리니어 프로브 (특징에 정보가 있나 - 결정적) ####

인코더를 freeze한 채, 그 특징 위에 아주 작은 분류기(선형) 하나만 얹어 "아이 있나 없나"를 학습시켜봅니다.

선형 프로브도 구분 못 함 → 특징 자체에 정보가 없음 = 인코더 병목 확정.   
선형 프로브는 잘 구분함 → 정보는 특징에 있음 = 문제는 디코더/헤드 (튜닝으로 해결 가능).    
이게 "인코더냐 디코더냐"를 가장 명확히 증명하는 방법이다. 

#### ⑥ 어블레이션: 디코더만 튜닝 → 안 되면 인코더 LoRA ####
디코더만 파인튜닝 → 개선되면 디코더가 병목이었음(인코더 무죄).    
정체하면 → 인코더 LoRA 추가 → 확 좋아지면 인코더가 병목이었음.    

#### 진단 순서 정리 (플로우) ####
```
0. 실패 사례 수집 + 정답 라벨 → "무엇을 얼마나 못 하나" 숫자화
1. 에러 유형 분류 (못 찾음? 마스크 엉성? 추적 끊김? 개념 혼동?)
2. 프롬프트 교체   → 되는 프롬프트 있으면 → VL/텍스트 문제
3. 크롭/확대       → 크롭하면 되면 → 입력(해상도) 문제  ★대부분 여기서 갈림
4. 특징/어텐션 시각화 + 리니어 프로브 → 인코더가 지각하나 판정
5. 디코더 튜닝 → 안 되면 인코더 LoRA (어블레이션)
```

#### 원인별 결론과 대응 ####

| 진단 결과 | 원인 | 대응 |
| :--- | :--- | :--- |
| **크롭하면 잘 됨** | 입력 (작은 객체/저해상도) | 전처리: 크롭 / 타일링 / 해상도↑ *(모델 안 바꿈)* |
| **특정 프롬프트만 됨** | VL / 텍스트 정렬 | 프롬프트 개선 or VL 부분 경량 튜닝 |
| **리니어 프로브 잘됨** | 디코더 | Detector Decoder 파인튜닝 |
| **리니어 프로브도 안됨** | 인코더 지각 | 인코더 LoRA / 더 큰 인코더 *(최후의 수단)* |
| **프레임 간 놓침** | Tracker | Tracker Memory / Decoder 튜닝 |



* 먼저: "인식 못한다"가 측정된 결과인지 인상인지 확인 → 실패 사례 + 라벨로 숫자화. 이게 없으면 진단 자체가 공허.
* 그다음: 싼 테스트부터 — 에러 유형 → 프롬프트 교체 → 크롭/확대 → 특징 프로브/리니어 프로브 → 어블레이션.
* 핵심 감별점: 크롭하면 되나?(입력 vs 인코더), 리니어 프로브 되나?(디코더 vs 인코더).
* 유치원 CCTV는 아이가 작게 잡히는 경우가 많아, 인코더가 아니라 입력(해상도) 문제로 판명될 확률이 꽤 높습니다. 이러면 파인튜닝 없이 전처리만으로 해결돼요.
* 그래서 고객에게 확인할 첫 질문은: "인식 못한다는 걸 어떻게 확인하셨어요? 실패한 실제 프레임 샘플을 주실 수 있나요?" 이다. 그 샘플로 위 크롭 테스트만 해봐도 "입력 문제냐 모델 문제냐"가 상당히 갈린다. 샘플 확보가 모든 진단의 출발점이다.


### 5. 파인튜닝 ###

앞 단계에서 "확실히 못 잡는다"가 확인되면 파인튜닝을 수행한다. 파인튜닝은 HF가 아니라 Meta 공식 repo(sam3/train.py) 또는 LoRA 특화 커뮤니티 repo(SAM3_LoRA)로 진행하며, LoRA로 detector decoder 등에 가볍게 시작할 수 있다. (라벨 데이터 확보가 전제)

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
> HF forward 함수는 loss를 자동 반환하지 않음 → 인코더/디코더 튜닝 시 HF만으로는 학습 loss를 바로 얻을 수 없음.  
> DETR 식 손실(헝가리안 매칭)이 필요한데, Meta repo(train.py) 나 SAM3_LoRA에 찾을 수 있음 -> 파인 튜닝시 HF 대신 사용.
> * https://github.com/Sompote/SAM3_LoRA - 이미지 파인 튜닝 샘플 제공 (오브젝트 detection/segmentation)
> * https://github.com/facebookresearch/sam3/blob/main/README_TRAIN.md - 이미지/영상에 대한 파인 튜닝 샘플 제공 ( + 오브젝트 tracking )

## 레퍼런스 ##

* https://huggingface.co/facebook/sam3 
* https://huggingface.co/settings/gated-repos


