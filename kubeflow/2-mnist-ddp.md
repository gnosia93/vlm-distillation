## MNIST 분산학습 (DDP) ##


### 1. 사전 준비 ###

### 1-1. EFA 지원 GPU 인스턴스 확인 ###
```bash
aws ec2 describe-instance-types \
    --filters Name=network-info.efa-supported,Values=true \
    --query "InstanceTypes[?GpuInfo.Gpus!=null].InstanceType" \
    --output text | sort
```

[결과]
```

```

### 1-2. MNIST 데이터 준비 ###

```
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

export AWS_REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)
export ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
export CLUSTER_NAME="vlm-distillation"
export BUCKET=vlm-data-${ACCOUNT_ID}-${AWS_REGION}

echo -e "\n-------------------------------------"
echo "AWS_REGION: $AWS_REGION"
echo "ACCOUNT_ID: $ACCOUNT_ID"
echo "CLUSTER_NAME: $CLUSTER_NAME"
echo "BUCKET: $BUCKET"
```
MNIST 데이터를 다운로드 받아서 S3 에 업로드 한다. 
```bash
cd 
git clone https://github.com/gnosia93/vlm-distillation.git
cd ~/vlm-distillation/kubeflow/src

pip install torch torchvision boto3
python3 upload_mnist_to_s3.py --s3-bucket $BUCKET --s3-prefix mnist/raw

aws s3 ls s3://$BUCKET/mnist/raw/
```
[결과]
```
2026-07-26 01:59:03    1648877 t10k-images-idx3-ubyte.gz
2026-07-26 01:59:03       4542 t10k-labels-idx1-ubyte.gz
2026-07-26 01:59:03    9912422 train-images-idx3-ubyte.gz
2026-07-26 01:59:03      28881 train-labels-idx1-ubyte.gz
```



### 2. 학습 코드 도커라이징 & 푸시 ###

도커 데몬을 설치한다.
```
sudo dnf install -y docker
sudo systemctl start docker
sudo systemctl enable docker
sudo usermod -aG docker $USER
newgrp docker
```

ecr 레포지토리를 생성하고 로그인 한다. 
```bash
cd ddp

export ECR=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
echo "ECR: $ECR"

aws ecr create-repository --repository-name mnist-ddp --region $AWS_REGION
aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR
```
[결과]
```
WARNING! Your password will be stored unencrypted in /home/ec2-user/.docker/config.json.
Configure a credential helper to remove this warning. See
https://docs.docker.com/engine/reference/commandline/login/#credentials-store

Login Succeeded
```

도커 이미지 빌드에 사용되는 Dockerfile 의 내용을 확인한다. 
```
cat Dockerfile
```
[결과]
```
# CUDA runtime base so the same image works on GPU nodes; falls back to CPU
# (gloo) automatically when no GPU is present. Swap for python:3.10-slim if
# you only ever run on CPU and want a smaller image.
# FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime
FROM public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.48-soci

WORKDIR /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 멀티노드/단일노드 multi-GPU 공용 학습 스크립트 (LOCAL_RANK 기준 통합).
COPY train.py .

# torchrun/env-based launch — Kubeflow injects MASTER_ADDR/PORT, RANK, WORLD_SIZE.
ENTRYPOINT ["python", "train.py"]
```

도커 이미지를 빌드하여 ecr 에 푸시한다.
```
docker build --platform linux/amd64 -t $ECR/mnist-ddp:v1.0.0 .
docker push $ECR/mnist-ddp:v1.0.0
```
[결과]
```
The push refers to repository [499514681453.dkr.ecr.ap-northeast-2.amazonaws.com/mnist-ddp]
787103d0bc13: Pushed 
1855cb852b5a: Pushed 
81014ae93b9a: Pushed 
5f70bf18a086: Pushed 
0f646da89c67: Pushed 
e75c063b91c2: Pushing [=======>                                           ]  1.123GB/7.591GB
a63da41ef05f: Pushed 
0b9c994b0484: Pushed 
```

ecr 에 푸시된 이미지를 확인한다.
```
aws ecr describe-images --repository-name mnist-ddp --region $AWS_REGION
```
[결과]
```
{
    "imageDetails": [
        {
            "registryId": "499514681453",
            "repositoryName": "mnist-ddp",
            "imageDigest": "sha256:cfc1c18db69546240237381ef3d854e4bdbd725f6239e9eb466207d7f1ac92c5",
            "imageTags": [
                ""v1.0.0"
            ],
            "imageSizeInBytes": 3737181538,
            "imagePushedAt": "2026-07-26T02:31:51.462000+00:00",
            "imageManifestMediaType": "application/vnd.docker.distribution.manifest.v2+json",
            "artifactMediaType": "application/vnd.docker.container.image.v1+json",
            "imageStatus": "ACTIVE"
        }
    ]
}
```
> [!WARNING]
> **이미지 태그 관리**  
> 훈련 소스 코드가 변경되면 Docker 빌드 시 태그를 증가시켜야 한다 ($ECR/mnist-ddp:v1.0.0 → $ECR/mnist-ddp:v1.0.1).
> 같은 태그로 덮어쓰면, ECR의 이미지는 갱신되지만 노드에 캐시된 기존 이미지가 재사용되어 변경 사항이 반영되지 않은 채 실행된다.
> 이는 고정 태그(latest가 아닌)의 기본 imagePullPolicy가 IfNotPresent이기 때문이다.
> Kubernetes는 태그 이름만 보고 판단하므로, 태그가 동일하면 "이미 있음"으로 간주해 레지스트리에서 새로 받지 않는다.
> 이렇게 하지 않으면, ecr 이미지는 업데이트 되나, 실행시 기존 이미지로 실행이 된다.  

### 3. S3 접근 설정 (IRSA) ###

Training job은 MNIST 데이터를 S3에서 읽어오고, 훈련 중 생성되는 체크포인트를 다시 S3에 저장한다. 그런데 Kubernetes 파드는 기본적으로 S3에 접근할 권한이 없다. 이 읽기/쓰기 권한을 부여하기 위해 IAM Roles for Service Accounts(IRSA)를 설정한다.

_아래 블록 전체를 복사하여 실행한다._
```
(
  set -euo pipefail

  # 필수 환경변수 검사 — 없으면 여기서 즉시 중단 (아래 명령 실행 안 함)
  : "${BUCKET:?BUCKET 환경변수가 설정되지 않았습니다. export BUCKET=... 먼저 실행하세요}"

  # 1. 클러스터에 OIDC 공급자 연결 (최초 1회)
  eksctl utils associate-iam-oidc-provider --cluster "$CLUSTER_NAME" --region "$AWS_REGION" --approve

  # 2. S3 접근 IAM 정책 생성 (버킷 읽기 + 체크포인트 쓰기)
  cat > s3-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": ["s3:GetObject", "s3:PutObject", "s3:ListBucket"],
      "Resource": [
        "arn:aws:s3:::$BUCKET",
        "arn:aws:s3:::$BUCKET/*"
      ]
    }
  ]
}
EOF

  aws iam create-policy \
    --policy-name mnist-s3-access \
    --policy-document file://s3-policy.json

  # 3. IAM Role 생성 + 서비스어카운트에 연결
  eksctl create iamserviceaccount \
    --cluster "$CLUSTER_NAME" --region "$AWS_REGION" \
    --namespace mnist \
    --name mnist-sa \
    --attach-policy-arn "arn:aws:iam::$ACCOUNT_ID:policy/mnist-s3-access" \
    --override-existing-serviceaccounts \
    --approve
)
```
지정한 정책(mnist-s3-access)을 붙인 IAM Role을 eksctl이 새로 생성하고, 그 Role ARN을 mnist-sa 서비스어카운트에 자동으로 연결한다. 이때 mnist 네임스페이스가 없으면 함께 생성되며, 이 서비스어카운트의 스코프는 mnist 네임스페이스로 한정된다.
따라서 이 SA의 S3 권한을 사용하려는 파드(즉 TrainJob)도 반드시 mnist 네임스페이스에 있어야 하며, 파드 스펙에 serviceAccountName: mnist-sa를 지정해야 실제로 권한이 적용된다.

서비스어카운트에 IAM Role이 연결됐는지 확인한다. 
```
kubectl get sa mnist-sa -n mnist -o yaml | grep role-arn
```
[결과]
```
eks.amazonaws.com/role-arn: arn:aws:iam::499514681453:role/eksctl-vlm-distillation-addon-iamserviceaccou-Role1-02ETyaUCb8Aj
```

### 4. TrainJob 실행 ###

```bash
export IMAGE_URI=$ECR/mnist-ddp:v1.0.0
envsubst '$IMAGE_URI $BUCKET' < trainjob-mnist.yaml | kubectl apply -f -

kubectl get trainjob -n mnist
kubectl describe trainjob mnist-ddp -n mnist

kubectl get pods -n mnist            
```
[결과]
```
NAME        STATE   AGE
mnist-ddp           3m12s

NAME                       READY   STATUS              RESTARTS   AGE
mnist-ddp-node-0-0-sbqmz   0/1     ContainerCreating   0          2m7s
mnist-ddp-node-0-1-zzsrt   0/1     ContainerCreating   0          2m7s
```

### 5. 모니터링 & 결과 검증 ###

* eks-node-viewer - GPU 노드를 확인    
`vs-code 에서 새로운 터미널을 하나 열고 eks-node-viewer 실행`    
![](https://github.com/gnosia93/vlm-distillation/blob/main/images/eks-nodeviewer.png)


* k9s - 파드 정보 확인      
`vs-code 에서 새로운 터미널을 하나 열고 k9s 실행 -> 숫자키 '1' 을 누름`    
![](https://github.com/gnosia93/vlm-distillation/blob/main/images/k9s-1.png)

* rank 0 파드 로그 스트리밍   
`파드 이름은 kubectl get pods 로 확인`    
```bash
kubectl logs -n mnist mnist-ddp-node-0-0-sbqmz -f
```
[결과]
```

```
> [!NOTE]
> 노드간의 통신은 .. EFA ...


### 6. Job 정리 / 재실행 ###

TrainJob은 이름으로 식별되며, spec.trainer 등 핵심 필드가 불변(immutable)이다. 따라서 같은 이름으로 다시 실행하려면 기존 Job을 먼저 삭제한 뒤 재생성해야 한다. 삭제하지 않고 수정된 매니페스트를 다시 적용하면 다음 오류가 발생한다.
`The TrainJob "mnist-ddp" is invalid: spec.trainer: Invalid value: "object": field is immutable`
TrainJob은 Kubernetes Job처럼 한 번 실행되는 작업으로 설계되어, 생성 후 실행 스펙 변경을 막는다. 그래서 이미지 태그나 커맨드를 바꿔 다시 돌릴 때는 항상 삭제 → 재생성 패턴을 쓴다.

```bash
kubectl delete trainjob mnist-ddp -n mnist     # 이전 작업 삭제 후 

envsubst '$IMAGE_URI $BUCKET' < trainjob-mnist.yaml | kubectl apply -f -      # 재실행
```

### 7. 싱글 노드 멀티 GPU로 실행하기 ###

분산 훈련의 성능은 GPU 간 통신(allreduce) 경로에 크게 좌우된다. 같은 GPU 수라도 이 통신이 어디를 거치느냐에 따라 속도가 달라진다. 경로를 빠른 순서로 보면 다음과 같다.

| 구성 | 통신 경로 | 상대 속도 |
|------|-----------|-----------|
| 싱글 노드 멀티 GPU | 노드 내부 (NVLink / PCIe P2P, 미지원 시 SHM) | 가장 빠름 |
| 멀티 노드 + EFA | 노드 간 고속 RDMA 네트워크 | 빠름 |
| 멀티 노드 + 일반 네트워크 | 노드 간 TCP/Socket (이더넷) | 가장 느림 |


즉 필요한 GPU 수가 한 노드 안에 들어간다면(대개 ≤8개), 여러 노드에 나눠 배치하는 것보다 한 노드에 몰아넣는 편이 대체로 빠르다. 노드 내부 통신(NVLink/PCIe/공유 메모리)은 노드 간 네트워크보다 지연이 훨씬 낮고, 잘 튜닝된 EFA조차 노드 내부 경로를 넘어서기는 어렵기 때문이다. 반대로 일반 네트워크(TCP)로 노드를 나누면 통신이 병목이 되어, GPU를 늘려도 기대만큼 빨라지지 않는 경우가 많다.

따라서 스케일링 전략은 이렇게 정리할 수 있다.
* 한 노드로 충분한 규모 → 싱글 노드 멀티 GPU (노드 내부 통신, 가장 유리)
* 한 노드 용량(최대 8 GPU)을 초과 → 멀티 노드로 확장하되, 이때는 반드시 EFA를 사용해 노드 간 통신 페널티를 최소화 (일반 TCP는 지양)

(확인필요) 참고로 g7e는 GPUDirect P2P(GPU Direct RDMA) 를 지원해 노드 내 멀티-GPU 성능에 유리하고, 노드당 1, 2, 4, 8 갯수의 GPU 를 지원한다. 

```
kubectl delete trainjob mnist-ddp -n mnist     # 이전 작업 삭제 후 

export IMAGE_URI=$ECR/mnist-ddp:v1.0.0
envsubst '$IMAGE_URI $BUCKET' < trainjob-mnist-1n2g.yaml | kubectl apply -f -

kubectl get pods -n mnist       
```
[결과]
```
NAME                            READY   STATUS      RESTARTS   AGE
mnist-ddp-1n2g-node-0-0-zgh6w   0/1     Completed   0          14m
```
`SHM/direct/direct` 통신을 확인하기 위해 로그를 조회한다. 
```
kubectl logs -n mnist-ddp-1n2g-node-0-0-zgh6w -f
```
[결과]

```

```

## AWS 딥러닝 컨테이너 이미지 ##

AWS는 공식 딥러닝 컨테이너(DLC) 이미지를 제공하고 있다. 특히 멀티 노드 GPU 분산 학습 시 필수적인 EFA(Elastic Fabric Adapter) 관련 설정이 미리 완료되어 있어, AWS DLC 이미지를 활용하면 손쉽게 환경을 구축할 수 있다.

* [Deep Learning Containers](https://gallery.ecr.aws/deep-learning-containers/pytorch-training)

![](https://github.com/gnosia93/vlm-distillation/blob/main/images/pytorch-gpu-dlc-2.png)

`public.ecr.aws/deep-learning-containers/pytorch-training:2.9.0-gpu-py312-cu130-ubuntu22.04-ec2-v1.25-soci`

* pytorch-training - PyTorch 모델 학습(Training) 전용 DLC 이미지.
* 2.9.0 - PyTorch v2.9.0 버전이 적용. (이전 2.8.0 대비 최신 PyTorch 기능 및 버그 수정 적용)
* gpu - NVIDIA GPU 전용 이미지.
* py312 - Python 3.12 환경.
* cu130 - CUDA 13.0 버전이 내장. (이전 cu129보다 최신 CUDA toolkit 탑재)
* ubuntu22.04 - 베이스 OS는 Ubuntu 22.04 LTS.
* ec2 - AWS EC2 및 EKS 환경, EFA 네트워크, AWS 전용 NCCL 플러그인에 최적화.
* v1.25 - AWS DLC release/patch 버전.
* soci - SOCI 지표가 적용되어 있어, 대용량 이미지임에도 빠른 지연 로딩(Lazy Loading)으로 파드를 초고속 스타트.

`public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.48-soci`


| 항목 | PyTorch 2.8.0 이미지 | PyTorch 2.9.0 이미지 | 주요 영향 및 특징 |
| :--- | :--- | :--- | :--- |
| **PyTorch 버전** | v2.8.0 | v2.9.0 | 최신 PyTorch API, Compiler 성능 및 분산 학습(FSDP2 등) 버그 수정 포함 |
| **CUDA Toolkit** | v12.9 (cu129) | v13.0 (cu130) | CUDA 13.0의 Shared Memory Spilling 등 메모리 처리 개선 및 최신 GPU 명령어 지원 |
| **최소 호스트 드라이버** | R535 / R550 이상 | R580 이상 | cu130 사용 시 노드(EC2)에 탑재된 NVIDIA 드라이버 요구 스펙이 더 높음 |
| **AWS DLC 패치 버전** | v1.48 | v1.25 | v2.8.0 이미지가 상대적으로 장기간 패치 및 검증되어 패치 횟수(v1.48)가 높음 |
| **Python / OS / SOCI** | 동일 (Python 3.12 / Ubuntu 22.04 / SOCI 지원) | 동일 (Python 3.12 / Ubuntu 22.04 / SOCI 지원) | 동일 환경 및 동일한 지연 로딩 최적화 적용 |


