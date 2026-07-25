
# 3. PyTorch DDP 기반 MNIST 모델 학습 (Kubeflow Trainer / V2)

> **Part 2. Advanced (EKS 병렬 스케일아웃)** 의 세 번째 장.
> 앞 장(2. K8s Job 기반 데이터 병렬 생성)에서 다룬 "독립 병렬"과 달리,
> 이 장은 프로세스 간 **통신이 필요한 분산 학습(DDP)** 을 Kubeflow Trainer(V2)로 실행한다.

**범례**: `[개념]` = 읽고 이해 · `[실습]` = 직접 실행 · `→ 확인` = 다음 단계로 넘어가기 전 검증

---

## 3.1 [개념] Job vs DDP — 왜 여기선 DDP인가

| | K8s Job (2장) | PyTorch DDP (이 장) |
|---|---|---|
| 병렬 방식 | 독립 병렬 (embarrassingly parallel) | 동기 병렬 (all-reduce) |
| 프로세스 간 통신 | **없음** — 각자 다른 데이터 처리 | **있음** — 매 step 그래디언트 동기화 |
| 실패 처리 | 하나 죽어도 나머지 진행 | 하나 죽으면 전체 재시작 필요 |
| 도구 | K8s `Job` (completions/parallelism) | Kubeflow Trainer `TrainJob` |

핵심: **데이터 생성**은 서로 몰라도 되니 Job으로 뿌리면 되지만, **모델 학습**은 모든
노드가 같은 모델을 유지해야 하므로 매 step 그래디언트를 평균(all-reduce)한다.
이 통신을 대신 처리해주는 것이 PyTorch DDP이고, 그 실행 인프라가 Kubeflow Trainer다.

---

## 3.2 [개념] Kubeflow Trainer(V2) 한눈에

V1(`PyTorchJob`)을 써봤다면 아래 차이만 알면 된다.

| | V1 (PyTorchJob) | V2 (Kubeflow Trainer) |
|---|---|---|
| API | `kubeflow.org/v1` | `trainer.kubeflow.org/v1alpha1` |
| CRD | `PyTorchJob` (프레임워크별) | `TrainJob` (통합) |
| 노드 표현 | `Master` + `Worker` **역할** | `numNodes` **숫자만** |
| rank 0 | "Master" 파드 | 이름 없음 — 그냥 rank 0 |
| 주 인터페이스 | YAML | **Python SDK** (YAML도 가능) |
| 런타임 | 매니페스트에 파드 spec 직접 작성 | `ClusterTrainingRuntime` 참조 |

> **왜 역할이 사라졌나**: DDP는 모든 노드가 대칭으로 동작하는 all-reduce 구조라
> "master가 지시한다"는 표현이 실제와 맞지 않았다. V2는 `numNodes` 숫자로만 규모를 선언한다.

---

## 3.3 [실습] 사전 준비 — 클러스터 확인

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

---

## 3.4 [실습] MNIST 데이터 준비 — S3에 배치

> 실무에선 2장에서 병렬 생성한 데이터가 입력이 되지만, 본 실습은 공개 MNIST를 사용한다.
> 학습 파드가 인터넷을 타지 않도록 **미리 S3에 올려둔다.**

로컬(또는 점프박스)에서 한 번만:
```bash
pip install torch torchvision boto3
export AWS_ACCESS_KEY_ID=...  AWS_SECRET_ACCESS_KEY=...  AWS_DEFAULT_REGION=ap-northeast-2
python upload_mnist_to_s3.py --s3-bucket my-datasets --s3-prefix mnist/raw
```

**→ 확인**:
```bash
aws s3 ls s3://my-datasets/mnist/raw/
# train-images-idx3-ubyte.gz 등 4개 파일이 보이면 통과
```

---

## 3.5 [실습] 분산 학습 코드 확인 (train.py)

`train.py`는 멀티노드/단일노드 공용이다. 핵심 3곳만 이해하면 된다.

**① 분산 초기화** — Trainer가 주입한 환경변수를 torch가 그대로 읽는다:
```python
backend = "nccl" if torch.cuda.is_available() else "gloo"
dist.init_process_group(backend=backend)   # MASTER_ADDR/RANK/WORLD_SIZE 자동 사용
```

**② 데이터 분할** — `DistributedSampler`가 노드별로 겹치지 않게 샘플을 나눈다:
```python
sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
```

**③ S3 다운로드 — 스케일 무관 핵심 로직** (3.9에서 숫자만 바꿔도 되는 이유):
```python
if local_rank == 0:                     # 각 파드의 대표 프로세스만 다운로드
    download_from_s3(...)
if is_distributed():
    dist.barrier()                      # 같은 파드의 나머지 프로세스는 대기
```
> `LOCAL_RANK` 기준이라 파드가 몇 개든 각 파드가 딱 한 번씩만 받는다. 이 덕분에
> 멀티노드/단일노드 전환 시 **코드를 고칠 필요가 없다.**

**→ 확인**: 코드 수정은 없다. 위 3곳의 의도만 이해하고 넘어간다.

---

## 3.6 [실습] 이미지 빌드 & 푸시

```bash
cd kubeflow-mnist-ddp

# Mac(Apple Silicon)에서 빌드하면 --platform 을 명시 (EKS 노드는 보통 amd64)
docker build --platform linux/amd64 -t YOUR_REGISTRY/mnist-ddp:latest .
docker push YOUR_REGISTRY/mnist-ddp:latest
```

**→ 확인**: `docker push` 가 성공하고, 레지스트리(ECR 등)에 `mnist-ddp:latest` 태그가 보이면 통과.

---

## 3.7 [실습] S3 자격증명 Secret 생성

```bash
kubectl create namespace kubeflow-user --dry-run=client -o yaml | kubectl apply -f -

kubectl create secret generic s3-credentials -n kubeflow-user \
  --from-literal=AWS_ACCESS_KEY_ID=... \
  --from-literal=AWS_SECRET_ACCESS_KEY=... \
  --from-literal=AWS_DEFAULT_REGION=ap-northeast-2
```
> 운영 환경(EKS)에서는 정적 키 대신 **IRSA**(서비스어카운트 기반 IAM)를 권장한다.

**→ 확인**: `kubectl get secret s3-credentials -n kubeflow-user` 로 존재 확인.

---

## 3.8 [실습] TrainJob 실행 — 두 가지 방법

### 3.8.1 Python SDK (권장 · 메인)

V2의 1급 인터페이스. `run_trainjob_sdk.py`의 값을 본인 환경으로 바꾼 뒤:
```bash
pip install kubeflow
python run_trainjob_sdk.py \
  --num-nodes 2 --num-proc-per-node 1 \
  --image YOUR_REGISTRY/mnist-ddp:latest \
  --runtime torch-distributed \
  --s3-bucket my-datasets --s3-prefix mnist/raw
```

### 3.8.2 YAML (참고)

SDK 없이 매니페스트로도 가능. `trainjob-mnist.yaml`의 `image`/`runtimeRef`/`S3_*` 교체 후:
```bash
kubectl apply -f trainjob-mnist.yaml
```

**→ 확인**:
```bash
kubectl get trainjob -n kubeflow-user
kubectl get pods -n kubeflow-user            # numNodes 만큼 파드가 뜬다
```

---

## 3.9 [실습] 스케일 조정 — 숫자만 바꿔 확장

**여기가 이 장의 하이라이트.** 코드(train.py)는 그대로 두고 두 숫자만 바꾼다.

| 목표 구성 | num_nodes | num_proc_per_node | gpus_per_node | WORLD_SIZE |
|---|---|---|---|---|
| 멀티노드 2대 × 1GPU | 2 | 1 | 1 | 2 |
| 단일노드 1대 × 2GPU | **1** | **2** | **2** | 2 |
| 멀티노드 2대 × 2GPU | 2 | 2 | 2 | 4 |

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

> **왜 코드 변경이 없나**: (1) DDP는 `WORLD_SIZE`/`RANK`를 환경변수로 읽어 자동 적응하고,
> (2) S3 다운로드가 `local_rank == 0` 기준이라 파드 수와 무관하게 항상 옳기 때문.
> — 단일노드는 한 파드에 local_rank 0가 하나라 1회, 멀티노드는 파드마다 1회씩 받는다.

**→ 확인**: 재실행한 Job의 파드 수가 `numNodes`와 일치하고, 로그에 `WORLD_SIZE`가
기대값으로 찍히면 통과.

---

## 3.10 [실습] 모니터링 & 결과 검증

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

---

## 3.11 [실습] 정리 & 재실행

TrainJob은 동명 재적용이 안 되므로, 다시 돌리려면 먼저 삭제한다.
```bash
# YAML 로 만든 경우
kubectl delete -f trainjob-mnist.yaml

# SDK 로 만든 경우 (job_name 은 실행 시 출력됨)
kubectl delete trainjob <job-name> -n kubeflow-user
```

---

## 부록. V1(PyTorchJob) 과의 매핑

기존 V1 자료를 참고하거나, 클러스터가 아직 V1일 때를 위한 대응표.

| 개념 | V1 (`pytorchjob-mnist.yaml`) | V2 (`trainjob-mnist.yaml`) |
|---|---|---|
| 리소스 종류 | `kind: PyTorchJob` | `kind: TrainJob` |
| 노드 2대 | `Master.replicas:1` + `Worker.replicas:1` | `numNodes: 2` |
| 노드당 GPU | replica별 `resources.limits` | `resourcesPerNode.limits` / `numProcPerNode` |
| 실행 확인 | `kubectl get pytorchjob` | `kubectl get trainjob` |
| 실행 방법 | YAML `kubectl apply` | SDK `TrainerClient.train` (또는 YAML) |

> `train.py`·`Dockerfile`·`upload_mnist_to_s3.py`·`s3-secret.yaml`은 V1/V2 공용이다.
> 매니페스트/실행 방법만 다르다.
