## MNIST 분산학습 (DDP) ##

### 1. MNIST 데이터 준비 ###

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
FROM pytorch/pytorch:2.3.1-cuda12.1-cudnn8-runtime

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

* eks-node-viewer - GPU 노드를 확인. 
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

### 6. Job 정리 / 재실행 ###

TrainJob 은 이름으로 구분되므로, 다시 돌리려면 삭제후 재 실행해야 한다.
삭제하지 않고 재실행하는 경우 `The TrainJob "mnist-ddp" is invalid: spec.trainer: Invalid value: "object": field is immutable` 
와 같은 오류가 발생한다. 

```bash
kubectl delete trainjob mnist-ddp -n mnist     # 이전 작업 삭제 후 
envsubst '$IMAGE_URI $BUCKET' < trainjob-mnist.yaml | kubectl apply -f -      # 재실행
```

### 7. 스케일 조정 ###

`trainjob-mnist.yaml`에서 numNodes, numProcPerNode, nvidia.com/gpu 항목을 수정한 후 실행한다. 
```yaml
  trainer:
    numNodes: 1
    numProcPerNode: "2"
    resourcesPerNode:
      limits:
        nvidia.com/gpu: 2
```
