_* ICN Region_
![](https://github.com/gnosia93/vlm-distillation/blob/main/images/gpu-compare-2.png)

* [p6-b300.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/p6/) - USD 142.416(_us-east-1_) - NVIDIA BW Tensor Core GPU x 8개 (GPU당 268GB HBM3e) -> 8
* [p6-b200.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/p6/) - USD 113.9328(_us-east-1_) - NVIDIA BW Tensor Core GPU x 8개 (GPU당 180GB HBM3e) -> 8
* [p5e.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/p5/) - USD 87.34848 - NVIDIA H200 Tensor Core GPU x 8개 (GPU당 141GB HBM3) -> 8
* [p5.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/p5/) - USD 75.9552 - NVIDIA H100 x 8개 (GPU당 80GB HBM3) -> 1/8
* <img width="12" height="12" alt="download" src="https://github.com/user-attachments/assets/6e5eca91-aaae-4c52-a43d-6f5b24b1dff8" /> [p4d.24xlarge](https://aws.amazon.com/ko/ec2/instance-types/p4/) - USD 30.41028 - NVIDIA A100 Tensor Core GPU x 8개 (GPU당 40GB HBM2) -> 8
* <img width="12" height="12" alt="download" src="https://github.com/user-attachments/assets/6e5eca91-aaae-4c52-a43d-6f5b24b1dff8" /> [g7e.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/g7e/) - USD 40.74921 - NVIDIA RTX PRO 6000 Blackwell x 8개 (GPU당 96GB) -> 1/2/4/8
* [g6e.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/g6e/) - USD 37.04468 - NVIDIA L40S x 8개 (GPU당 48GB) -> 1/4/8
* [g6.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/g6/) - USE $16.416 - NVIDIA L4 x 8개 (GPU당 24GB) -> 1/4/8
* [g5.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/g5/) - USD 20.02809 - NVIDIA A10G x 8개 (GPU당 24GB) -> 1/4/8
* [g4dn.metal](https://aws.amazon.com/ko/ec2/instance-types/g4/) - USD 9.624 - NVIDIA T4 x 4개 (GPU당 16 GB) -> 1/4/Metal 8

![](https://github.com/gnosia93/vlm-distillation/blob/main/images/blank-space.png)


### G 시리즈 스팩 비교 ###

| 패밀리 | GPU | 아키텍처 | GPU 메모리 | 연산성능(TFLOPS) | 대역폭 | 주요특징 | 비고 |
|--------|-----|----------|-------|----|------|------- |-------|
| g4dn | 텐서코어 2세대 - [NVIDIA T4](https://www.nvidia.com/ko-kr/data-center/tesla-t4/) | Turing | 16GB |FP32-8.1, FP16-65, INT8-130| GDDR6 320 GB/s |INT 4/8 지원 | 이미지 분류, 객체 탐지(CNN) 및 음성 인식과 같은 기계 학습 모델의 배포 |
| <img width="12" height="12" alt="download" src="https://github.com/user-attachments/assets/6e5eca91-aaae-4c52-a43d-6f5b24b1dff8" /> g5 | 텐서코어 3세대 - [NVIDIA A10G](https://www.nvidia.com/ko-kr/data-center/products/a10-gpu/) | Ampere | 24GB | FP32-31.2, FP16-125, INT8-250 | **GDDR6 600 GB/s** | + TF32, BF16 지원, **Flash Attention 1/2** | **G4dn 대비 최대 3배 더 높은 추론 성능 제공**, 최대 100Gbps의 네트워크 대역폭, 최대 7.6TB의 로컬 NVMe SSD 스토리지 지원  |
| <img width="12" height="12" alt="download" src="https://github.com/user-attachments/assets/6e5eca91-aaae-4c52-a43d-6f5b24b1dff8" /> g6 | 텐서코어 4세대 - [NVIDIA L4](https://www.nvidia.com/ko-kr/data-center/l4/) | Ada Lovelace | 24GB | FP32-30.3, FP16-242, FP8-485| GDDR6 300 GB/s | + **FP8 및 트랜스포머 엔진** | **G4dn 인스턴스 대비 추론 성능 2배 향상**, 자연어 처리(NLP)·언어 번역·비디오 및 이미지 분석·음성 인식·개인화를 위한 머신러닝(ML) 모델 지원, 최대 100Gbps의 네트워크 대역폭 및 최대 7.52TB의 로컬 NVMe SSD 스토리지 지원 |
| g6e | 텐서코어 4세대 - [NVIDIA L40S](https://www.nvidia.com/ko-kr/data-center/l40s/) | Ada Lovelace | 48GB | FP32-91.6, FP16-733, FP8-1.4P | GDDR6 864 GB/s |  |**G5 인스턴스 대비 최대 2.5배 더 나은 성능 제공**, 최대 130억(13B) 파라미터 LLM / 디퓨전(Diffusion) 모델 배포, 최대 400Gbps의 네트워크 대역폭 지원, 최대 7.6TB의 로컬 NVMe SSD 스토리지 제공.|
| g7 | 텐서코어 5세대 - [NVIDIA RTX PRO 4500 BW](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4500/) | Blackwell | 32GB | | |  |.  |
| g7e | 텐서코어 5세대 - [NVIDIA RTX PRO 6000 BW](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) | Blackwell | 96GB | FP32-120, FP16-1P, FP8-2P | **GDDR7 ~1.6 TB/s** | + **Flash Attention 3/4, FP4** | **G6e 인스턴스 대비 최대 2.3배의 추론 성능 성능 향상**, G6e 인스턴스 대비 최대 4배의 GPU 간 통신 대역폭 및 4배의 EFA(Elastic Fabric Adapter) 네트워크 대역폭 제공, EFA를 통해 최대 1600Gbps의 네트워크 대역폭 지원, 최대 15.2TB의 로컬 NVMe SSD 스토리지 제공. |

![](https://github.com/gnosia93/vlm-distillation/blob/main/images/blank-space.png)


### P 시리즈 스팩 비교 ###

EC2의 P 계열은 딥러닝 학습·추론과 HPC에 특화된 GPU 가속 인스턴스로, 세대마다 최신 NVIDIA 데이터센터 GPU를 탑재.

| 패밀리 | GPU 아키텍처 | 아키텍처 | GPU 메모리 | 연산성능(TFLOPS) | 대역폭 | 주요특징 | 비고 |
|--------|--------|-----|-----------|------------------|--------|---------|------|
| P4e / P4de | 텐서코어 3세대 - [NVIDIA A100](https://www.nvidia.com/ko-kr/data-center/a100/) | Ampere | 40 or 80GB | FP32-19.5, FP16-312, INT8-624 | HBM2e 1.6 or 2.0 TB/s | NVLink 600GB/s, EFA 400Gbps | **Flash Attention 1/2**  |
| P5 | 텐서코어 4세대 - [NVIDIA H100](https://www.nvidia.com/ko-kr/data-center/h100/) | Hopper | 80GB | FP32-67, FP16-1.9P FP8-3.9P | HBM3 3.35 TB/s | NVLink 900GB/s, EFA 3.2Tbps | P4 대비 최대 4배 연산성능, 100B+ 모델훈련, **+ Flash Attention 3**  |
| P5e / P5en | 텐서코어 4세대 - [NVIDIA H200](https://www.nvidia.com/ko-kr/data-center/h200/) | Hopper | 141GB | FP32-67, FP16-1.9P FP8-3.9P  | HBM3 4.8 TB/s | NVLink 900GB/s, EFA 3.2Tbps  | - |
| P6-B200 | 텐서코어 5세대 - [NVIDIA BW B200](https://www.nvidia.com/ko-kr/data-center/hgx/) | Blackwell | 180GB | FP32-600, FP16-2.25P FP8-4.5P, FP4-9P | HBM3e 7.7 TB/s | NVLink 1.8 TB/s, EFA 3.2Tbps  | P5en 인스턴스에 비해 최대 2.25배 높은 연산성능, **+ Flash Attention 4**   |
| P6-B300 | 텐서코어 5세대 - [NVIDIA BW B300](https://www.nvidia.com/ko-kr/data-center/hgx/) | Blackwell | 268GB | FP32-600, FP16-2.25P FP8-4.5P, FP4-9P  | HBM3e 7.7 TB/s | NVLink 1.8 TB/s, EFA 6.4Tbps |  -  |

![](https://github.com/gnosia93/vlm-distillation/blob/main/images/blank-space.png)

### FlashAttention ###


| 버전 | 대상 GPU | 핵심 혁신 | 해결한 병목 |
|------|----------|-----------|-------------|
| FA1 (2022) | A100 (Ampere) | Tiling + online softmax + recompute | HBM I/O, O(N²) 메모리 |
| FA2 (2023) | A100 등 | 병렬화·작업분할 개선, non-matmul 감소 | 낮은 GPU 활용률 |
| FA3 (2024) | H100 (Hopper) | 비동기 실행, warp specialization, FP8 | 연산·데이터이동 오버랩 |
| FA4 (2025~26) | B200 (Blackwell) | 알고리즘·커널 co-design, exp 근사, CuTe-DSL | 비대칭 스케일링(SFU·공유메모리) |

**"exact attention을 유지하면서, 매 세대 GPU에서 새로 생긴 병목을 찾아 최적화한다"**, FA1~2는 메모리 I/O, FA3는 Hopper의 비동기성, FA4는 Blackwell에서 텐서코어만 빨라지고 나머지는 안 따라온 불균형 제거.

![](https://github.com/gnosia93/vlm-distillation/blob/main/images/fa-evol-kr.png)

#### 1. 플래시어텐션이 등장한 배경 (문제점) ####
트랜스포머 모델의 어텐션 연산은 입력 시퀀스의 길이(‭$N$‬)가 길어질 때, 메모리 사용량과 연산량이 시퀀스 길이의 제곱(‭N^2‬)에 비례하여 폭발적으로 증가하는 문제를 가지고 있습니다. (이를 ‭O(N^2)‬‭‬‭‬‭‬ 복잡도라고 합니다.)
이로 인해 두 가지 주요 병목 현상이 발생합니다.

1.	`메모리 병목 (HBM 쓰기)`: 기본 어텐션 연산은 중간 계산 결과물인 어텐션 행렬(‭N x N‬‭‬)을 GPU의 느린 메인 메모리(HBM)에 저장했다가 다시 읽어와야 합니다. 시퀀스가 길어지면 이 행렬이 너무 커져 HBM 용량을 초과하거나, 데이터를 읽고 쓰는 속도가 연산 속도를 따라가지 못하는 병목이 발생합니다.
2.	`연산 병목`: 메모리 입출력에 걸리는 시간이 실제 코어에서 연산하는 시간보다 훨씬 길어져 GPU의 강력한 연산 능력을 제대로 활용하지 못합니다.

#### 2. 플래시어텐션의 핵심 원리 ####
플래시어텐션은 이러한 문제를 해결하기 위해 하드웨어 레벨(GPU의 메모리 계층 구조)을 고려하여 어텐션 알고리즘을 재설계했습니다.

`핵심 아이디어 1`: 타일링 (Tiling) 및 로컬 연산
* 원리: 어텐션 연산에 필요한 행렬(‭Q, K, V‭‬‭‬)을 한꺼번에 처리하지 않고, 조각(Tile) 단위로 쪼개어 가져옵니다.
* 효과: 이 조각 데이터를 GPU의 **느린 HBM(메인 메모리)에서 빠른 SRAM(내부 메모리)**으로 가져온 뒤, SRAM 안에서 최종 결과가 나올 때까지 연산을 마치고 HBM에는 최종 결과물만 한 번에 씁니다. 이를 통해 HBM 읽기/쓰기 횟수를 최소화합니다.

`핵심 아이디어 2`: 온라인 소프트맥스 (Online Softmax)
* 원리: 어텐션 행렬 전체를 미리 계산해서 메모리에 저장하지 않습니다. 타일링된 조각 데이터 안에서 소프트맥스(Softmax) 계산에 필요한 중간 통계값(최댓값, 합계)만 업데이트하며 로컬에서 소프트맥스 연산을 수행합니다.
* 효과: ‭N x N‬‭‬ 크기의 거대한 중간 어텐션 행렬을 HBM에 저장할 필요를 완전히 없애, 메모리 사용량을 ‭N^2‬에서 ‭N‬ 수준으로 줄입니다.

#### 3. 플래시어텐션의 발전사 (v1~v4) ####

플래시어텐션은 하드웨어의 발전에 맞춰 계속 진화해 왔습니다.

* FlashAttention-1 (2022): 타일링과 온라인 소프트맥스를 도입하여 HBM 병목을 해결하고 메모리 사용량을 크게 줄임. (A100 등 암페어 아키텍처 최적화)
* FlashAttention-2 (2023): 1버전의 연산 효율이 낮았던 점을 개선. 일하는 단위(Warp)를 효율적으로 나누고 불필요한 스케일링 연산을 줄여 GPU 활용률을 극대화함.
* FlashAttention-3 (2024): Hopper (H100) 하드웨어의 비동기 하드웨어 특성(TMA, Tensor Cores)을 극대화하기 위해 재설계. 데이터 로드와 연산을 완전히 겹치게(Overlap) 만들어 노는 시간(Stall)을 완전히 없앰.
* FlashAttention-4 (2026): Blackwell (B200) 아키텍처의 비대칭 스케일링 문제(연산은 엄청 빨라졌으나 지수함수 연산 및 메모리 대역폭은 정체)를 해결하기 위해 지수함수(‭e^x‬) 계산 자체를 쉬운 소프트웨어 방식으로 분산 처리(에뮬레이션)하고 새로운 텐서 메모리(TMEM)를 활용함


### Price ###

* https://instances.vantage.sh/aws/ec2/p4d.24xlarge?currency=USD&region=ap-northeast-2
