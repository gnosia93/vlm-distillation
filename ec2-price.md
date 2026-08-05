_* ICN Region_
![](https://github.com/gnosia93/vlm-distillation/blob/main/images/gpu-compare-2.png)

* [p5e.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/p5/) - USD 87.34848 - NVIDIA H200 Tensor Core GPU x 8개 (GPU당 141GB HBM3e) -> 8
* [p5.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/p5/) - USD 75.9552 - NVIDIA H100 x 8개 (GPU당 80GB HBM3) -> 1/8
* <img width="12" height="12" alt="download" src="https://github.com/user-attachments/assets/6e5eca91-aaae-4c52-a43d-6f5b24b1dff8" /> [p4d.24xlarge](https://aws.amazon.com/ko/ec2/instance-types/p4/) - USD 30.41028 - NVIDIA A100 Tensor Core GPU x 8개 (GPU당 40GB HBM2) -> 8
* <img width="12" height="12" alt="download" src="https://github.com/user-attachments/assets/6e5eca91-aaae-4c52-a43d-6f5b24b1dff8" /> [g7e.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/g7e/) - USD 40.74921 - NVIDIA RTX PRO 6000 Blackwell x 8개 (GPU당 96GB) -> 1/2/4/8
* [g6e.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/g6e/) - USD 37.04468 - NVIDIA L40S x 8개 (GPU당 48GB) -> 1/4/8
* [g6.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/g6/) - USE $16.416 - NVIDIA L4 x 8개 (GPU당 24GB) -> 1/4/8
* [g5.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/g5/) - USD 20.02809 - NVIDIA A10G x 8개 (GPU당 24GB) -> 1/4/8
* [g4dn.metal](https://aws.amazon.com/ko/ec2/instance-types/g4/) - USD 9.624 - NVIDIA T4 x 4개 (GPU당 16 GB) -> 1/4/Metal 8

### G 시리즈 스팩 비교 ###

| 패밀리 | GPU | 아키텍처 | GPU 메모리 | 연산성능(TFLOPS) | 대역폭 | 주요특징 | 비고 |
|--------|-----|----------|-------|----|------|------- |-------|
| g4dn | 텐서코어 2세대 - [NVIDIA T4](https://www.nvidia.com/ko-kr/data-center/tesla-t4/) | Turing | 16GB |FP32-8.1, FP16-65, INT8-130| GDDR6 320 GB/s |INT 4/8 지원 | 이미지 분류, 객체 탐지(CNN) 및 음성 인식과 같은 기계 학습 모델의 배포 |
| <img width="12" height="12" alt="download" src="https://github.com/user-attachments/assets/6e5eca91-aaae-4c52-a43d-6f5b24b1dff8" /> g5 | 텐서코어 3세대 - [NVIDIA A10G](https://www.nvidia.com/ko-kr/data-center/products/a10-gpu/) | Ampere | 24GB | FP32-31.2, FP16-125, INT8-250 | **GDDR6 600 GB/s** | + TF32, BF16 지원, **Flash Attention 1/2** | **G4dn 대비 최대 3배 더 높은 추론 성능 제공**, 최대 100Gbps의 네트워크 대역폭, 최대 7.6TB의 로컬 NVMe SSD 스토리지 지원  |
| <img width="12" height="12" alt="download" src="https://github.com/user-attachments/assets/6e5eca91-aaae-4c52-a43d-6f5b24b1dff8" /> g6 | 텐서코어 4세대 - [NVIDIA L4](https://www.nvidia.com/ko-kr/data-center/l4/) | Ada Lovelace | 24GB | FP32-30.3, FP16-242, FP8-485| GDDR6 300 GB/s | + **FP8 및 트랜스포머 엔진** | **G4dn 인스턴스 대비 추론 성능 2배 향상**, 자연어 처리(NLP)·언어 번역·비디오 및 이미지 분석·음성 인식·개인화를 위한 머신러닝(ML) 모델 지원, 최대 100Gbps의 네트워크 대역폭 및 최대 7.52TB의 로컬 NVMe SSD 스토리지 지원 |
| g6e | 텐서코어 4세대 - [NVIDIA L40S](https://www.nvidia.com/ko-kr/data-center/l40s/) | Ada Lovelace | 48GB | FP32-91.6, FP16-733, FP8-1.4P | GDDR6 864 GB/s |  |**G5 인스턴스 대비 최대 2.5배 더 나은 성능 제공**, 최대 130억(13B) 파라미터 LLM / 디퓨전(Diffusion) 모델 배포, 최대 400Gbps의 네트워크 대역폭 지원, 최대 7.6TB의 로컬 NVMe SSD 스토리지 제공.|
| g7 | 텐서코어 5세대 - [NVIDIA RTX PRO 4500 BW](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-4500/) | Blackwell | 32GB | | |  |.  |
| g7e | 텐서코어 5세대 - [NVIDIA RTX PRO 6000 BW](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/) | Blackwell | 96GB | FP32-120, FP16-1P, FP8-2P | **GDDR7 ~1.6 TB/s** | + **Flash Attention 3/4, FP4** | **G6e 인스턴스 대비 최대 2.3배의 추론 성능 성능 향상**, G6e 인스턴스 대비 최대 4배의 GPU 간 통신 대역폭 및 4배의 EFA(Elastic Fabric Adapter) 네트워크 대역폭 제공, EFA를 통해 최대 1600Gbps의 네트워크 대역폭 지원, 최대 15.2TB의 로컬 NVMe SSD 스토리지 제공. |


### P 시리즈 스팩 비교 ###

> EC2의 P 계열은 딥러닝 학습·추론과 HPC에 특화된 GPU 가속 인스턴스로, 세대마다 최신 NVIDIA 데이터센터 GPU를 탑재한다.
> 연산성능·대역폭은 **GPU 1개 기준의 NVIDIA 공표 피크값**이며(별도 표기 제외), 정밀도(FP16/FP8/FP4)와 sparsity 적용 여부에 따라 달라진다.

| 패밀리 | GPU 아키텍처 | GPU 메모리 | 연산성능(TFLOPS) | 대역폭 | 주요특징 | 비고 |
|--------|-------------|-----------|------------------|--------|---------|------|
| **P4** | NVIDIA A100 (Ampere) | 40 / 80GB | FP16 312 | 1.6 / 2.0 TB/s | 이전 세대 학습 표준 | p4d=40GB, p4de=80GB |
| **P5** | NVIDIA H100 (Hopper) | 80GB | FP8 1,979 / FP16 989 | 3.35 TB/s | 대규모 LLM 학습·추론 |  |
| **P5e / P5en** | NVIDIA H200 (Hopper) | 141GB | FP8 1,979 / FP16 989 | 4.8 TB/s | H100과 연산 동일, 메모리·대역폭 강화 | |
| **P6-B200** | NVIDIA Blackwell B200 | 180GB | FP4 9,000 / FP8 4,500 | 7.7 TB/s | P5en 대비 메모리 대역폭 +60%, EFAv4 최대 3.2Tbps |  |
| **P6-B300** | NVIDIA Blackwell Ultra (B300) | 268GB | B200 대비 약 1.5배 | 7.7 TB/s | EFA 6.4Tbps, ENA 300Gbps, B200 대비 네트워크 2배 |  |

#### 각주 ####
1. NVIDIA 단일 GPU 원사양은 B200 = 192GB, Blackwell Ultra(B300) ≈ 288GB HBM3e. AWS는 P6-B200을 "총 1,440GB", P6-B300을 "총 2.1TB"로 안내하며, 이는 인스턴스에서 노출/할당되는 값 기준이라 원사양 단순 합산과 차이가 있다.
2. P6e 계열의 연산성능은 GPU 1개가 아니라 **UltraServer(최대 72 GPU) 전체** 기준이다. 나머지 P4~P6은 GPU 1개 기준이므로 직접 비교 시 유의.

#### 접미사 의미 ####
- `e` : 메모리(HBM) 강화 버전 (예: P5e = H200)
- `n` : 네트워킹 강화 버전
- `GB`: Grace CPU + Blackwell GPU 슈퍼칩(NVL72)
- `UltraServer`: 여러 노드를 NVLink로 묶어 다수 GPU를 하나처럼 사용하는 구성





### Price ###

* https://instances.vantage.sh/aws/ec2/p4d.24xlarge?currency=USD&region=ap-northeast-2
