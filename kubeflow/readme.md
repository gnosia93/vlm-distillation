
## Kubeflow로 MNIST 분산학습 (DDP) ##

### 1. Kubeflow Trainer 설치 ###

```bash
# Trainer(V2) 설치 확인 — 이 CRD 가 있어야 한다
kubectl get crd trainjobs.trainer.kubeflow.org

# 사용 가능한 런타임 확인 → 3.8 의 runtime_ref 에 넣을 이름
kubectl get clustertrainingruntime

# SDK 필드명은 버전마다 다를 수 있으니 스키마 확인
kubectl explain trainjob.spec.trainer

# GPU 노드 확인
kubectl get nodes -o json | grep nvidia.com/gpu
```

**→ 확인**: `trainjobs.trainer.kubeflow.org` CRD가 존재하고, `clustertrainingruntime`
목록에 torch 계열 런타임(예: `torch-distributed`)이 하나 이상 보이면 통과.

> ⚠️ V2는 API가 안정화 전이다. 위에서 확인한 **런타임 이름·필드명을 이 실습 내내 그대로 사용**한다.


### 2. MNIST 데이터 준비 ###

```bash
pip install torch torchvision boto3
python upload_mnist_to_s3.py --s3-bucket my-datasets --s3-prefix mnist/raw

aws s3 ls s3://my-datasets/mnist/raw/
```
train-images-idx3-ubyte.gz 등 4개 파일이 보이면 통과


### 3. 이미지 빌드 & 푸시 ###

```bash
# ecr 로그인 추가 ..

docker build --platform linux/amd64 -t YOUR_REGISTRY/mnist-ddp:latest .
docker push YOUR_REGISTRY/mnist-ddp:latest
```

ecr 레지스트리 조회.


### 4. S3 접근 IRSA 생성 ###
;; pod 가 s3 접근이 가능하다록 한다.


### 5. TrainJob 실행 ###

#### Python SDK 활용 ####

```bash
pip install kubeflow

python run_trainjob_sdk.py \
  --num-nodes 2 --num-proc-per-node 1 \
  --image YOUR_REGISTRY/mnist-ddp:latest \
  --runtime torch-distributed \
  --s3-bucket my-datasets --s3-prefix mnist/raw
```

#### YAML 활용 ####

```bash
kubectl apply -f trainjob-mnist.yaml
```

#### 실행 확인 ####
```bash
kubectl get trainjob -n kubeflow-user
kubectl get pods -n kubeflow-user            # numNodes 만큼 파드가 뜬다
```

---

### 6. 스케일 조정 ###

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

---

### 7. 모니터링 & 결과 검증 ###

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

### 8. 정리 & 재실행 ###

TrainJob은 동명 재적용이 안 되므로, 다시 돌리려면 먼저 삭제한다.
```bash
# YAML 로 만든 경우
kubectl delete -f trainjob-mnist.yaml

# SDK 로 만든 경우 (job_name 은 실행 시 출력됨)
kubectl delete trainjob <job-name> -n kubeflow-user
```
