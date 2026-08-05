
유치원과 같은 환경에서 사람(아동·교사) 인식 및 행동 분석을 목적으로 SAM-3 기반 모델을 파인튜닝할 때 필요한 데이터 수량과 구축 전략이다.

### 💡 훈련 데이터 준비 ###
유치원 CCTV/카메라 영상은 사람 간 신체 겹침(가림 현상), 다양한 자세, 빠르게 움직이는 아동, 좁은 공간 내 밀집 등의 난관이 많다.

* 최소 PoC (가치 검증)	200장 ~ 500장	- 특정 카메라 구도 1~2곳 기반 기본 성능 확인 - Mask Decoder 위주 학습
* 실전 서비스 적용 수준	1,000장 ~ 3,000장	- 다양한 카메라 각도, 조명, 옷차림, 행동 패턴 수용 - 가림 현상(Occlusion) 극복 가능 수준
* 고도화 (다중 유치원/다양한 구도)	5,000장 이상	- 원거리/근거리, 야외 놀이터, 주간/어두운 실내 등 전천후 대응

### 🎯 왜 일반 파인튜닝보다 데이터가 더 필요한가? (도메인 특수성) ###

* 신체 겹침 및 가림 (Occlusion) - 아이들이 얽혀 놀거나 교사와 교감할 때 신체 일부가 가려지는 경우가 빈번. 가려진 상태에서도 사람 개체를 정확히 분리(Instance Segmentation)하려면 다양한 겹침 프레임 데이터가 많이 확보되어야 한다.
* 다양한 포즈 및 작은 객체 - 엎드리거나, 쪼그려 앉거나, 뛰어다니는 등 일반적인 '서 있는 사람' 데이터셋(COCO 등)과 포즈 분포가 전혀 다르다. 화면 구석의 작은 아동 인스턴스도 인식해야 한다.
* 행동 파악과의 연계 - 행동 파악은 보통 **SAM (사람/신체 영역 분리) + Pose Estimation (관절 추적) + Action Recognition (행동 분류 모델)**을 조합하여 구성된다. SAM은 이 과정에서 '사람 영역을 정확히 따내는 역할'을 하므로, 경계선(Boundary)이 뭉개지지 않는 고품질 마스크 데이터가 핵심이다.
  
### 💡 실무 데이터 구축 및 파인튜닝 전략 ###
#### 1. 프레임 추출 전략 (영상 데이터 특성 활용) ####
* 동영상(CCTV) 원본을 매 프레임 따면 중복이 심하다.
*	초당 1프레임(1 FPS) 이하로 추출하거나, 주요 행동 변화가 있는 구간만 선택 추출하여 중복성을 최대로 낮춘다. (예: 10분 영상에서 중복되지 않는 알짜배기 30~50장 추출)

#### 2. 효율적인 레이블링 (Semi-Auto Masking) ####
* SAM의 기본 zero-shot 성능을 활용하여 1차로 자동 마스크를 생성한 뒤, 사람이 경계선 결함이나 겹쳐진 아이들의 경계만 수정하는 방식으로 레이블링 시간을 80% 이상 단축할 수 있다. 

#### 3. 학습 모델 범위 (LoRA 활용) ####
* 무거운 Image Encoder는 고정(Freeze)하고 Mask Decoder 부분만 LoRA(Low-Rank Adaptation) 방식으로 파인튜닝하는 것을 권장. 데이터가 500~1,000장 수준으로 적어도 과적합 없이 빠른 수렴이 가능.

> [!IMPORTANT]  
> SAM에서 Image Encoder는 Freeze, Mask Decoder만 LoRA로 학습하는 이유:
> 
> - 인코더 Freeze: 파라미터의 90% 이상을 차지해 학습 비용이 크고, 이미 수십억 마스크로 학습돼 시각 특징 추출 능력이 완성형. 건드리면 일반화 성능만 잃음(파괴적 망각).
> - 디코더만 학습: 전체의 1~2%로 가볍고, 우리가 고칠 건 "특징 추출"이 아니라 유치원 상황(가림·앉은 자세)에 맞는 마스크 복원 기준뿐.
> - LoRA 사용: 데이터가 500~1,000장뿐이라 전체 학습은 과적합 확정. 작은 보조 행렬만 학습해 과적합을 줄이고 효율을 극대화.
> 결론: 무거운 인코더는 그대로, 가벼운 디코더에 LoRA만 얹어 최소 자원으로 도메인 적응. 

### 💡 진행 순서 ###

> 1.	우선 카메라 구도별로 대표 이미지 200~300장을 정밀하게 레이블링.
> 2.	Mask Decoder / LoRA 기반 1차 파인튜닝을 진행.
> 3.	아동이 겹치거나 쪼그려 앉아 인식이 깨지는 오류 구간(Hard Negative) 이미지 300~500장을 추가 발굴하여 보완 학습시키는 순서로 진행하는 것이 가장 비용 효율적

## 디코더 파인 튜닝 ##

#### SAM 3 아키텍처 (Segment Anything with Concepts, Meta 2025) ####

SAM 1/2와 가장 큰 차이는 텍스트/예시 프롬프트로 "개념(concept)" 단위 검출이 가능해졌다는 점이고, 구조적으로는 Detector + Tracker 2단 구성이다. 그래서 디코더가 한
개가 아니라 여러 곳에 있다.

```
  ┌───────────────────────────── SAM 3 ─────────────────────────────┐
  │                                                                 │
  │  INPUT: image / video frame          PROMPT                     │
  │         │                            ├─ text (예: "아이", "모자")  │
  │         │                            ├─ image exemplar (박스 예시)│
  │         ▼                            └─ geometry (point/box/mask)│
  │  ┌────────────────────┐                        │                │
  │  │  Vision Encoder    │  ← Perception          │                │
  │  │  (PE ViT)   ❄FROZEN│    Encoder (PE)        ▼                │
  │  └─────────┬──────────┘              ┌──────────────────┐       │
  │            │ image tokens            │  Text Encoder    │       │
  │            │                         │  (PE)     ❄FROZEN│       │
  │            │                         └────────┬─────────┘       │
  │            └────────────┬─────────────────────┘                 │
  │                         ▼                                       │
  │              ┌──────────────────────┐                           │
  │              │   Fusion Encoder     │ (이미지↔텍스트 정렬)           │
  │              └──────────┬───────────┘                           │
  │                         ▼                                       │
  │  ══════════════ DETECTOR (DETR 계열) ══════════════              │
  │  ┌───────────────────────────────────────────────┐              │
  │  │ ★ Detection Decoder  (object queries)         │◄── 디코더 ①   │
  │  │      → box + score                            │              │
  │  │   + Presence Head/Token ("그 개념이 있냐?")      │              │
  │  ├───────────────────────────────────────────────┤              │
  │  │ ★ Mask Head (= Mask Decoder)                  │◄── 디코더 ②   │
  │  │      → 인스턴스별 마스크                           │             │
  │  └───────────────────┬───────────────────────────┘              │
  │                      ▼  (영상일 때만)                              │
  │  ══════════════ TRACKER (SAM 2 계보) ══════════════              │
  │  ┌───────────────────────────────────────────────┐              │
  │  │   Prompt Encoder                              │              │
  │  │   Memory Attention  ◄──  Memory Bank          │              │
  │  │ ★ Mask Decoder (프레임별 마스크 복원)              │◄── 디코더 ③   │
  │  │   Memory Encoder    ──►  Memory Bank          │              │
  │  └───────────────────┬───────────────────────────┘              │
  │                      ▼                                          │
  │     Temporal Disambiguation / Matching (검출↔트랙 연결, ID 유지)    │
  └─────────────────────────────────────────────────────────────────┘
                         ▼
     OUTPUT: 프롬프트 개념에 해당하는 **모든** 인스턴스 마스크 + track ID
```

#### 디코더는 어디인가 ? ####

* ①   │ Detector의 Detection Decoder │ object query → 박스/점수 (있는 것 다 찾기)               
* ②   │ Detector의 Mask Head         │ 박스별 마스크 픽셀 복원                             
* ③   │ Tracker의 Mask Decoder       │ 메모리 참조해 프레임마다 마스크 갱신 (SAM 2의 그 디코더)
  
무거운 쪽은 Vision Encoder(PE ViT) 하나이다. 전체 약 8억대 파라미터 중 대부분이 여기 있고, Detector/Tracker는 이 백본을 공유한다.

#### 유치원 CCTV 파인튜닝에 대입하면 ####

- ❄ Freeze: Vision Encoder(PE), Text Encoder — 앞서 얘기한 이유 그대로 (비용 + 파괴적 망각).
- 🔧 LoRA 대상: ① Detection Decoder, ② Mask Head, ③ Tracker Mask Decoder.
  - "가림/겹침 때문에 애를 놓친다" → ①(+ Presence Head) 쪽 교정.
  - "사람은 찾는데 마스크 경계가 지저분하다" → ②③ 쪽 교정.
  - "프레임 넘어가면 ID가 바뀐다" → ③ + Matching 쪽 교정.
- 실무적으로는 Fusion Encoder의 cross-attention에도 LoRA를 얹는 경우가 많다. ("아이", "선생님" 같은 도메인 어휘와 시각 특징의 정렬을 맞추기 위해).

SAM 2 대비 체감 차이는, 예전엔 "사람을 박스로 찍어줘야" 했던 걸 SAM 3는 "child" 같은 텍스트 한 줄로 프레임 내 전원을 한 번에 잡아준다는 점입니다. 즉 검출기가 내장돼
별도 YOLO 계열 디텍터를 붙일 필요가 줄어든다.

## Text Alignment ##
비전 인코더(Vision Encoder)는 완전히 동결(Freeze)한 상태에서 텍스트와 비전 간 정렬(Alignment) 성능을 끌어올리기 위해서는 텍스트 인코더 자체 또는 비전-텍스트를 이어주는 중간 레이어(Fusion Encoder)만 선택적으로 트레이닝 한다.
SAM 3 구조를 활용하여 텍스트 정렬 부분만 파인튜닝할 수 있는 핵심 기법과 단계별 접근법은 다음과 같다.

### 1. 파라미터 고정(Freezing) 설정 전략 ###
가장 기본이자 핵심은 비전 인코더의 가중치 업데이트를 차단하고, 학습시킬 모듈의 requires_grad만 True로 설정하는 것이다.

*	Vision Encoder (PE ViT): requires_grad = False (완전 동결)
*	Fusion Encoder (Cross-Attention 영역): requires_grad = True (학습)
* Text Encoder (선택 사항):
  * 방법 A (LoRA 적용 - 추천): Text Encoder 전체를 튜닝하면 범용 언어 능력이 파괴(Catastrophic Forgetting)될 수 있으므로, **LoRA(Low-Rank Adaptation)**를 텍스트 인코더에만 붙여서 정렬 성능을 개선
  * 방법 B (Projection Layer만 학습): Text Encoder 자체는 Frozen 상태로 두고, 텍스트 토큰을 Fusion Encoder로 보내는 마지막 선형 투영 레이어(Text Projection Head)만 학습

### 2. 추천하는 3가지 파인튜닝 기법

#### ① LoRA (Low-Rank Adaptation)를 이용한 텍스트 인코더 튜닝 ####
텍스트 인코더의 Self-Attention 레이어(‭$W_q, W_v$‬‭‬ ‭‬)에 LoRA 어댑터를 삽입

⚬	장점: 파라미터 추가량이 전체의 1% 미만으로 극히 적으면서도, 도메인 특화 단어(예: 의료용어, 공장 불량 명칭 등)와 비전 특징 간 정렬 능력을 대폭 끌어올릴 수 있다.

#### ② Fusion Encoder의 Cross-Attention 집중 튜닝 ####
이미지와 텍스트 토큰이 만나는 Fusion Encoder 내부의 Cross-Attention 및 Feed-Forward Network(FFN) 레이어만 가중치를 열어둔다.

⚬	원리: 이미 추출된 강인한 비전 패치 임베딩을 바탕으로, 입력된 텍스트 Query가 비전의 어느 위치를 참조(Attend)해야 하는지 매핑 규칙만 재학습한다.

#### ③ Soft Prompt Tuning / Prefix Tuning (Text Prompt Encoder) ####
텍스트 인코더 전단에 학습 가능한 연속적인 임베딩 벡터(Virtual Tokens)를 붙여 학습하는 방식이다.

⚬	원리: "아이", "모자"라는 단어 앞에 학습 가능한 프롬프트 토큰 ‭$[P_1][P_2]\dots$‬‭‬‭‬‭‬‭‬‭‬‭‬를 주입하여, 비전 특징과 잘 맞물리도록 텍스트 표현을 유연하게 변형한다.

### 3. 손실 함수(Loss) 구성 방안 ###

텍스트 정렬(Alignment) 향상이 목적이므로 다음과 같은 손실 함수 조합을 사용한다.

1.	Contrastive Loss / Region-Text Alignment Loss:

⚬	입력한 텍스트 프롬프트 벡터와 Detector/Mask Head에서 추출된 해당 인스턴스 영역의 비전 벡터 간 코사인 유사도를 높이도록 학습한다.

2.	Presence Loss (Presence Head):

⚬	이미지 내에 해당 텍스트 프롬프트 개념이 실제 존재하는지 판단하는 분류 손실(BCE Loss)을 함께 학습시켜 거짓 긍정(False Positive)을 줄인다.



