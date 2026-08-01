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

| 패밀리 | GPU | 아키텍처 | GPU 메모리 | 대역폭 | 비고 |  |
|--------|-----|----------|-----------|------|------- |------- |
| g4dn | NVIDIA T4 | Turing | 16GB | GDDR6 320 GB/s |INT 8/4 | 이미지 분류, 객체 탐지(CNN) 및 음성 인식과 같은 기계 학습 모델의 배포 |
| g5g | NVIDIA T4G | Turing (ARM/Graviton2) | 16GB | | ARM 기반, T4급 |  |
| g5 | NVIDIA A10G | Ampere | 24GB | GDDR6 600 GB/s | Flash Attention 2 | **G4dn 대비 최대 3배 더 높은 추론 성능 제공**, 최대 100Gbps의 네트워크 대역폭, 최대 7.6TB의 로컬 NVMe SSD 스토리지 지원  |
| g6 | NVIDIA L4 | Ada Lovelace | 24GB | GDDR6 300 GB/s | 전력효율 좋음 | **G4dn 인스턴스 대비 추론 성능 2배 향상**, 자연어 처리(NLP)·언어 번역·비디오 및 이미지 분석·음성 인식·개인화를 위한 머신러닝(ML) 모델 지원, 최대 100Gbps의 네트워크 대역폭 및 최대 7.52TB의 로컬 NVMe SSD 스토리지 지원 |
| g6e | NVIDIA L40S | Ada Lovelace | 48GB | GDDR6 864 GB/s | 고성능/대용량 |**G5 인스턴스 대비 최대 2.5배 더 나은 성능 제공**, 최대 130억(13B) 파라미터 LLM / 디퓨전(Diffusion) 모델 배포, 최대 400Gbps의 네트워크 대역폭 지원, 최대 7.6TB의 로컬 NVMe SSD 스토리지 제공.|
| g7 | NVIDIA RTX PRO 4500 Blackwell | Blackwell | 32GB | | G6 대비 메모리·대역폭 향상 |.  |
| g7e | NVIDIA RTX PRO 6000 Blackwell | Blackwell | 96GB | GDDR7 ~1.6 TB/s | Flash Attention 3/4, FP 4/8 |Amazon Elastic Compute Cloud (Amazon EC2) G7e instances, accelerated by NVIDIA RTX PRO™ 6000 Blackwell Server Edition GPUs, deliver cost-effective performance for generative AI inference workloads and the highest performance for spatial computing workloads. These instances offer 2x the GPU memory (96 GB), 1.85x the GPU memory bandwidth, up to 4x the inter-GPU communication bandwidth and 4x the Elastic Fabric Adapter (EFA) networking bandwidth compared to G6e instances. G7e instances offer up to 2.3x inference performance compared to G6e. Customers can use G7e instances to deploy large language models (LLMs), agentic AI models, multimodal generative AI models and physical AI models. Additionally, G7e instances can be used to accelerate a broad range of workloads including spatial computing and scientific computing workloads. G7e instances feature up to 8 NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs with 768 GB of total GPU memory (96 GB of memory per GPU) and 5th generation Intel Xeon Scalable (Emerald Rapids) processors. G7e instances support up to 192 vCPUs, up to 1600 Gbps of networking bandwidth with EFA, up to 2 TiB of system memory, and up to 15.2 TB of local NVMe SSD storage.  |


### Price ###

* https://instances.vantage.sh/aws/ec2/p4d.24xlarge?currency=USD&region=ap-northeast-2
