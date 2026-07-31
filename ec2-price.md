_Seoul_
![](https://github.com/gnosia93/vlm-distillation/blob/main/images/gpu-comparison.png)

* [p5e.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/p5/) - USD 87.34848 - NVIDIA H200 Tensor Core GPU x 8개 (GPU당 141GB HBM3e) -> 8
* [p5.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/p5/) - USD 75.9552 - NVIDIA H100 x 8개 (GPU당 80GB HBM3) -> 1/8
* <img width="12" height="12" alt="download" src="https://github.com/user-attachments/assets/6e5eca91-aaae-4c52-a43d-6f5b24b1dff8" /> [p4d.24xlarge](https://aws.amazon.com/ko/ec2/instance-types/p4/) - USD 30.41028 - NVIDIA A100 Tensor Core GPU x 8개 (GPU당 40GB HBM2) -> 8
* <img width="12" height="12" alt="download" src="https://github.com/user-attachments/assets/6e5eca91-aaae-4c52-a43d-6f5b24b1dff8" /> [g7e.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/g7e/) - USD 40.74921 - NVIDIA RTX PRO 6000 Blackwell x 8개 (GPU당 96GB) -> 1/2/4/8
* [g6e.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/g6e/) - USD 37.04468 - NVIDIA L40S x 8개 (GPU당 48GB) -> 1/4/8
* [g5.48xlarge](https://aws.amazon.com/ko/ec2/instance-types/g5/) - USD 20.02809 - NVIDIA A10G x 8개 (GPU당 24GB) -> 1/4/8
* [g4dn.12xlarge](https://aws.amazon.com/ko/ec2/instance-types/g4/) - USD 4.812 - NVIDIA T4 x 4개 (GPU당 16 GB) -> 1/4/Metal 8

### G 시리즈 스팩 비교 ###

| 패밀리 | GPU | 아키텍처 | GPU 메모리 | 비고 |
|--------|-----|----------|-----------|------|
| g4dn | NVIDIA T4 | Turing | 16GB | NVIDIA Turing, INT 8/4 |
| g5g | NVIDIA T4G | Turing (ARM/Graviton2) | 16GB | ARM 기반, T4급 |
| g5 | NVIDIA A10G | Ampere | 24GB | NVIDIA Ampere, Flash Attention 2 |
| g6 | NVIDIA L4 | Ada Lovelace | 24GB | 전력효율 좋음 |
| g6e | NVIDIA L40S | Ada Lovelace | 48GB | 고성능/대용량 |
| g7 | NVIDIA RTX PRO 4500 Blackwell | Blackwell | 32GB | G6 대비 메모리·대역폭 향상 |
| g7e | NVIDIA RTX PRO 6000 Blackwell | Blackwell | 96GB | 최상위, 대형 LLM/멀티모달 |


* price - https://instances.vantage.sh/aws/ec2/p4d.24xlarge?currency=USD&region=ap-northeast-2
