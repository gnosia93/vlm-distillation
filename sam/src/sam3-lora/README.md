# SAM3 LoRA 파인튜닝 예제

Meta 의 [SAM 3](https://huggingface.co/facebook/sam3) 을 LoRA 로 파인튜닝하는 동작하는 예제.
COCO 형식 instance segmentation 데이터를 SAM3 의 **Promptable Concept Segmentation (PCS)**
형태로 변환해 학습한다.

## SAM3 파인튜닝의 특이점

일반적인 HF 파인튜닝과 다른 점이 세 가지 있고, 이 예제는 그 세 가지를 다룬다.

**1. `Sam3Model.forward` 는 loss 를 계산하지 않는다.**
`labels` 인자가 없고 반환값은 `pred_masks / pred_boxes / pred_logits / presence_logits / semantic_seg`
뿐이다. SAM3 는 DETR 계열 head 를 쓰므로 Hungarian matching 기반 set-prediction 손실을
직접 붙여야 한다 → `sam3_lora/loss.py`

**2. 학습 샘플의 단위가 "이미지"가 아니라 "(이미지, 개념 1개)" 다.**
SAM3 는 짧은 명사구 하나를 받아 해당하는 **모든** 인스턴스를 찾는다. 따라서 한 이미지에
person 과 car 가 있으면 학습 샘플이 2개가 된다 → `sam3_lora/data.py`

**3. presence head 를 따로 학습해야 한다.**
SAM3 는 인식(이 개념이 이미지에 있나?)과 위치추정(어디에 있나?)을 분리하고, 최종 점수를
`pred_logits.sigmoid() * presence_logits.sigmoid()` 로 만든다. 프롬프트에 해당하는 객체가
하나도 없는 **hard negative** 샘플을 섞어서 presence 를 학습시켜야 false positive 가 줄어든다.

## 설치

```bash
python -m venv .venv && source .venv/bin/activate
pip install "transformers>=5.14" torch torchvision peft scipy pycocotools pillow matplotlib
```

`facebook/sam3` 은 gated repo 다. HF 페이지에서 라이선스에 동의한 뒤:

```bash
hf auth login       # 또는 export HF_TOKEN=...
```

## 빠른 검증 (모델 다운로드 없이)

0.9B gated 체크포인트를 받지 않고 파이프라인 전체를 검증한다. 축소된 랜덤 초기화
`Sam3Config` 로 실제 구조를 그대로 유지한 채 데이터 → LoRA → 손실 → backward →
저장/로드 → merge 까지 9단계를 확인한다.

```bash
python smoke_test.py \
  --images-dir ~/fiftyone/coco-2017/validation/data \
  --annotations ~/fiftyone/coco-2017/validation/labels.json
```

학습 루프 자체도 gated 접근 없이 돌려볼 수 있다:

```bash
python train_lora.py --tiny-debug \
  --images-dir ~/fiftyone/coco-2017/validation/data \
  --annotations ~/fiftyone/coco-2017/validation/labels.json \
  --categories person car --epochs 1 --max-steps 8 --output-dir /tmp/run
```

## 학습

```bash
python train_lora.py \
  --images-dir /data/coco/train2017 \
  --annotations /data/coco/annotations/instances_train2017.json \
  --categories person car bicycle \
  --lora-stages detr_encoder detr_decoder mask_decoder \
  --lora-r 16 --lora-alpha 32 \
  --epochs 10 --batch-size 4 --lr 1e-4 \
  --dtype bfloat16 \
  --output-dir runs/sam3-lora
```

주요 옵션:

| 옵션 | 설명 |
|---|---|
| `--lora-stages` | LoRA 부착 스테이지. `vision text geometry detr_encoder detr_decoder mask_decoder` 중 선택 |
| `--lora-projections` | 기본 `q_proj v_proj`. 예산이 있으면 `q_proj k_proj v_proj o_proj` |
| `--image-size` | 기본 1008 (사전학습 해상도). 낮추면 메모리는 줄지만 정확도가 떨어진다 |
| `--negative-ratio` | hard negative 비율 (기본 0.25). 0 으로 두면 presence 가 학습되지 않는다 |
| `--head-lr` | `modules_to_save` head 용 별도 lr (기본 1e-5). 사전학습 head 파괴 방지 |
| `--no-modules-to-save` | box/presence head 를 얼리고 LoRA 만 학습 |

### 어느 스테이지에 붙일까

기본값은 `detr_encoder + detr_decoder + mask_decoder` 다. 개념 이해와 위치추정을 담당하는
DETR 스택만 적응시키고, 0.9B 중 대부분을 차지하는 ViT backbone 은 건드리지 않는다.

- **COCO 와 비슷한 자연 이미지, 새 카테고리** → 기본값
- **도메인이 크게 다름** (의료/위성/적외선/현미경) → `vision` 추가. backbone feature 자체가
  안 맞으므로 DETR 만 조정해서는 한계가 있다
- **텍스트 프롬프트 어휘가 특수함** (전문용어, 제품명) → `text` 추가
- **박스/포인트 프롬프트 위주로 쓸 것** → `geometry` 추가

`modules_to_save` 로 `box_head`, `presence_head`, `dot_product_scoring` 의 projection 은
통째로 학습한다. 출력 차원이 4, 1 이라 저랭크 근사가 의미가 없고, 새 도메인에서 스케일이
크게 바뀌기 때문이다.

## 추론

```bash
python infer_lora.py \
  --adapter runs/sam3-lora/final \
  --image /data/coco/val2017/000000000139.jpg \
  --text "person" \
  --out result.png
```

`--adapter` 를 생략하면 베이스 모델로 돌아가므로 파인튜닝 전/후를 바로 비교할 수 있다.
`--merge` 를 주면 LoRA 를 베이스 가중치에 흡수해 추론 오버헤드를 없앤다.

## 평가

```bash
# 전
python evaluate.py --images-dir ... --annotations ... --categories person car
# 후
python evaluate.py --images-dir ... --annotations ... --categories person car \
    --adapter runs/sam3-lora/final
```

mask AP@[.50:.95] 와 함께 **presence accuracy** 및 negative 이미지의 평균 false positive
개수를 출력한다. PCS 에서는 "없는 것을 없다고 하는" 능력이 AP 만큼 중요하다.

## 구조

```
sam3-lora/
├── sam3_lora/
│   ├── data.py     COCO → PCS 샘플 변환, hard negative 생성, collate
│   ├── loss.py     Hungarian matching + focal/L1/GIoU/dice/presence 손실
│   └── lora.py     SAM3 모듈 경로 → LoRA 타겟 선택
├── train_lora.py   학습 루프 (warmup+cosine, grad accum, 어댑터 저장)
├── infer_lora.py   추론 + 마스크/박스 시각화
├── evaluate.py     mask AP + presence accuracy
└── smoke_test.py   다운로드 없는 9단계 파이프라인 검증
```

## 손실 구성

| 항목 | 가중치 | 적용 대상 |
|---|---|---|
| `loss_class` | 2.0 | query 200개 전체, sigmoid focal |
| `loss_bbox` | 5.0 | 매칭된 query, L1 (정규화 좌표) |
| `loss_giou` | 2.0 | 매칭된 query, GIoU |
| `loss_mask` | 5.0 | 매칭된 query, sigmoid focal (점 샘플링) |
| `loss_dice` | 5.0 | 매칭된 query, dice |
| `loss_presence` | 1.0 | 이미지당 1개, BCE |

가중치는 DETR/Mask2Former 관례를 따른 출발점이다. 데이터에 맞게 조정한다:
마스크 경계가 뭉개지면 `loss_dice`↑, false positive 가 많으면 `loss_presence`↑.

## 알려진 제약

- **`Sam3Model` 은 gradient checkpointing 을 지원하지 않는다.**
  (`_supports_gradient_checkpointing = False`, 확인: transformers 5.14.1)
  1008px 로 메모리가 부족하면 `--image-size` 를 낮추거나 `--grad-accum` 을 쓴다.
- 마스크 GT 는 모델 출력 해상도(`image_size * 4 / 14`)로 렌더링한다. 아주 작은 객체는
  이 과정에서 사라질 수 있으므로 `--min-area` 로 걸러 낸다.
- deep supervision(중간 decoder layer 손실)은 구현하지 않았다. `outputs.decoder_hidden_states`
  가 모든 layer 를 담고 있으니 필요하면 추가할 수 있다.
- `iscrowd=1` 어노테이션은 기본적으로 제외한다.

## 검증한 환경

transformers 5.14.1 / torch 2.13.0 / peft 0.20.0 / macOS (MPS).
`smoke_test.py` 9단계 전부 통과, 축소 모델 40 epoch overfit 에서 모든 손실 항목 감소 확인.
