## MNIST 분산학습 (DDP) ##

### 1. MNIST 데이터 준비 ###

```
TOKEN=$(curl -s -X PUT "http://169.254.169.254/latest/api/token" -H "X-aws-ec2-metadata-token-ttl-seconds: 21600")

export REGION=$(curl -s -H "X-aws-ec2-metadata-token: $TOKEN" http://169.254.169.254/latest/meta-data/placement/region)
export ACCOUNT_ID=$(aws sts get-caller-identity --query 'Account' --output text)
export BUCKET=vlm-data-${ACCOUNT_ID}-${REGION}

echo -e "\n-------------------------------------"
echo "REGION: $REGION"
echo "ACCOUNT_ID: $ACCOUNT_ID"
echo "BUCKET: $BUCKET"
```
MNIST 데이터를 다운로드 받아서 S3 에 업로드 한다. 
```bash
cd 
git clone https://github.com/gnosia93/vlm-distillation.git
cd ~/vlm-distillation/kubeflow/src

pip install torch torchvision boto3
python upload_mnist_to_s3.py --s3-bucket my-datasets --s3-prefix mnist/raw

aws s3 ls s3://my-datasets/mnist/raw/
```
train-images-idx3-ubyte.gz 등 4개 파일이 보이면 통과


### 2. 학습 코드 도커라이징 ###

```bash
export AWS_REGION=ap-northeast-2
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export ECR=$ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

aws ecr create-repository --repository-name mnist-ddp --region $AWS_REGION || true

aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR

docker build --platform linux/amd64 -t $ECR/mnist-ddp:latest .
docker push $ECR/mnist-ddp:latest

aws ecr describe-images --repository-name mnist-ddp --region $AWS_REGION
```

### 3. S3 접근 설정 ###

쿠버테이스 파드가 S3 에 접근하여 Read/Write 과 같은 작업을 수행하기 위해서는 IRSA(IAM Role for Sservice Account)설정이 필요하다 
```
export CLUSTER=my-eks-cluster
export AWS_REGION=ap-northeast-2
export ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
export NAMESPACE=kubeflow-user
export SA_NAME=mnist-trainer
export BUCKET=my-datasets

# 1. 클러스터에 OIDC 공급자 연결 (최초 1회)
eksctl utils associate-iam-oidc-provider --cluster $CLUSTER --region $AWS_REGION --approve

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

# 3. IAM Role 생성 + 서비스어카운트에 연결 (role-arn annotation 자동 부여)
eksctl create iamserviceaccount \
  --cluster $CLUSTER --region $AWS_REGION \
  --namespace $NAMESPACE \
  --name $SA_NAME \
  --attach-policy-arn arn:aws:iam::$ACCOUNT_ID:policy/mnist-s3-access \
  --approve
```

확인 — 서비스어카운트에 IAM Role이 연결됐는지:
```
kubectl get sa $SA_NAME -n $NAMESPACE -o yaml | grep role-arn
# eks.amazonaws.com/role-arn: arn:aws:iam::<계정ID>:role/... 이 보이면 성공
```

IRSA는 서비스 어카운트를 만들어 놓은 것일 뿐이고, TrainJob 파드가 그 서비스어카운트(mnist-trainer)를 실제로 쓰도록 지정해야 적용됩니다. 

- YAML (trainjob-mnist.yaml): 파드 템플릿에 serviceAccountName: mnist-trainer 추가
- SDK (run_trainjob_sdk.py): TrainJob이 이 SA를 쓰도록 지정 (필드명은 설치 버전의 스키마 확인 필요)

> [!NOTE]
> 동작원리 
> 파드 안의 boto3가 서비스어카운트에 붙은 토큰으로 IAM Role을 assume 하여 임시 자격증명 획득하므로,
> 코드/매니페스트에 AWS 키를 넣지 않아도 된다. 

### 4. TrainJob 실행 ###

```bash
kubectl apply -f trainjob-mnist.yaml
```

아래 명령으로 실행 여부를 확인한다. 
```bash
kubectl get trainjob -n kubeflow-user
kubectl get pods -n kubeflow-user            # numNodes 만큼 파드가 뜬다
```

### 5. 스케일 조정 ###

SDK 예 (단일노드 2GPU로 다시 실행):
```bash
python run_trainjob_sdk.py --num-nodes 1 --num-proc-per-node 2 --gpus-per-node 2
```

YAML이라면 `trainjob-mnist.yaml`에서:
```yaml
  trainer:
    numNodes: 1
    numProcPerNode: "2"
    resourcesPerNode:
      limits:
        nvidia.com/gpu: 2
```

### 6. 모니터링 & 결과 검증 ###

```bash
# rank 0 파드 로그 스트리밍 (파드 이름은 kubectl get pods 로 확인)
kubectl logs -n kubeflow-user <rank0-pod> -f
```
정상 로그 예:
```
start: rank=0/2 local_rank=0 backend=nccl device=cuda:0
[rank 0] epoch 1 [0/60000] loss=2.3011
...
모델 저장 완료: /data/mnist_ddp.pt
체크포인트 업로드: s3://my-datasets/checkpoints/mnist_ddp.pt
```

**→ 확인**:
- 로그에서 loss가 epoch마다 감소
- 체크포인트 업로드 확인:
  ```bash
  aws s3 ls s3://my-datasets/checkpoints/
  ```

### 7. Job 정리 및 재실행 ###

TrainJob 은 이름으로 구분되므로, 다시 돌리려면 삭제후 재 실행해야 한다.

```bash
kubectl delete -f trainjob-mnist.yaml
```

> [!NOTE]
> trainjob 명령어
> * 잡 확인 - kubectl get trainjob                       
> * 잡 삭제 - kubectl delete trainjob llama-3-8b        
> * 잡 상세 - kubectl describe trainjob llama-3-8b
