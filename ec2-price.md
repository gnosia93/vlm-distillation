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

| 패밀리 | GPU | 아키텍처 | GPU 메모리 | 비고 |  |
|--------|-----|----------|-----------|------|------- |
| g4dn | NVIDIA T4 | Turing | 16GB | INT 8/4 |  |
| g5g | NVIDIA T4G | Turing (ARM/Graviton2) | 16GB | ARM 기반, T4급 |  |
| g5 | NVIDIA A10G | Ampere | 24GB | Flash Attention 2 |Amazon EC2 G5 인스턴스는 NVIDIA GPU 기반 인스턴스 중에서 최신 세대로, 다양한 그래픽 집약적 사용 사례와 기계 학습 사용 사례에서 사용할 수 있습니다. Amazon EC2 G4dn 인스턴스와 비교할 때 그래픽 집약적 애플리케이션 및 기계 학습 추론에서 최대 3배 더 높은 성능을 제공하고, 기계 학습 훈련에서 최대 3.3배 더 높은 성능을 제공합니다. 원격 워크스테이션, 비디오 렌더링 및 게임과 같은 그래픽 집약적인 애플리케이션에 G5 인스턴스를 사용하여 고화질 그래픽을 실시간으로 생성할 수 있습니다. G5 인스턴스를 기계 학습에 사용하면 비용 효율적인 고성능 인프라에서 자연어 처리, 컴퓨터 비전 및 추천 엔진 사용 사례를 위한 더 크고 더 복잡한 모델을 훈련하고 배포할 수 있습니다. G5 인스턴스는 최대 8개의 NVIDIA와 2세대 AMD EPYC 프로세서를 갖추고 있습니다. 또한 최대 192개의 vCPU, 최대 100Gbps의 네트워크 대역폭 및 최대 7.6TB의 로컬 NVMe SSD 스토리지를 지원합니다.  |
| g6 | NVIDIA L4 | Ada Lovelace | 24GB | 전력효율 좋음 |Amazon EC2 G6 instances powered by NVIDIA L4 Tensor Core GPUs can be used for a wide range of graphics-intensive and machine learning use cases. The G6 instances offer 2x better performance for deep learning inference and graphics workloads compared to EC2 G4dn instances. G6 instances also introduce sizes with fractionalized GPU offerings for ML inference and graphics workloads that cannot fully utilize the NVIDIA L4 GPUs. Customers can use G6 instances for deploying ML models for natural language processing, language translation, video and image analysis, speech recognition, and personalization as well as graphics workloads, such as creating and rendering real-time, cinematic-quality graphics and game streaming. G6 instances feature up to 8 NVIDIA L4 Tensor Core GPUs with 24 GB of memory per GPU and fractionalized GPU sizes with as little as 1/8 of an L4 GPU with 3 GB of GPU memory. They also support up to 192 vCPUs, up to 100 Gbps of network bandwidth, and up to 7.52 TB of local NVMe SSD storage.  |
| g6e | NVIDIA L40S | Ada Lovelace | 48GB | 고성능/대용량 |Amazon EC2 G6e instances powered by NVIDIA L40S Tensor Core GPUs are the most cost-efficient GPU instances for deploying generative AI models and the highest performance GPU instances for spatial computing workloads. They offer 2x higher GPU memory (48 GB), and 2.9x faster GPU memory bandwidth compared to G6 instances. G6e instances deliver up to 2.5x better performance compared to G5 instances. Customers can use G6e instances to deploy large language models (LLMs) with up to 13B parameters and diffusion models for generating images, video, and audio. Additionally, the G6e instances will unlock customers’ ability to create larger, more immersive 3D simulations and digital twins for spatial computing workloads using NVIDIA Omniverse. G6e instances feature up to 8 NVIDIA L40S Tensor Core GPUs with 384 GB of total GPU memory (48 GB of memory per GPU) and third generation AMD EPYC processors. They also support up to 192 vCPUs, up to 400 Gbps of network bandwidth, up to 1.536 TB of system memory, and up to 7.6 TB of local NVMe SSD storage. |
| g7 | NVIDIA RTX PRO 4500 Blackwell | Blackwell | 32GB | G6 대비 메모리·대역폭 향상 |.  |
| g7e | NVIDIA RTX PRO 6000 Blackwell | Blackwell | 96GB | 최상위, 대형 LLM/멀티모달 |
Amazon Elastic Compute Cloud (Amazon EC2) G7e instances, accelerated by NVIDIA RTX PRO™ 6000 Blackwell Server Edition GPUs, deliver cost-effective performance for generative AI inference workloads and the highest performance for spatial computing workloads. These instances offer 2x the GPU memory (96 GB), 1.85x the GPU memory bandwidth, up to 4x the inter-GPU communication bandwidth and 4x the Elastic Fabric Adapter (EFA) networking bandwidth compared to G6e instances. G7e instances offer up to 2.3x inference performance compared to G6e.
Customers can use G7e instances to deploy large language models (LLMs), agentic AI models, multimodal generative AI models and physical AI models. Additionally, G7e instances can be used to accelerate a broad range of workloads including spatial computing and scientific computing workloads.
G7e instances feature up to 8 NVIDIA RTX PRO 6000 Blackwell Server Edition GPUs with 768 GB of total GPU memory (96 GB of memory per GPU) and 5th generation Intel Xeon Scalable (Emerald Rapids) processors. G7e instances support up to 192 vCPUs, up to 1600 Gbps of networking bandwidth with EFA, up to 2 TiB of system memory, and up to 15.2 TB of local NVMe SSD storage.  |


### Price ###

* https://instances.vantage.sh/aws/ec2/p4d.24xlarge?currency=USD&region=ap-northeast-2
