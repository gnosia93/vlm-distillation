
## Kubeflow 의 이해 ##

### 1. Kubeflow 란? ###

Kubeflow는 쿠버네티스 위에서 머신러닝 워크플로를 실행하기 위한 오픈소스 플랫폼으로, "ML을 쿠버네티스 네이티브하게 돌리자"는 것이 핵심 목표다.

모델을 만드는 과정은 하나의 작업이 아니라 여러 단계로 이어진다 — 데이터 전처리, 모델 학습, 하이퍼파라미터 튜닝, 모델서빙, 파이프라인 관리 등, 이걸 각자 다른 도구로 따로 돌리면 관리가 어렵다. Kubeflow는 이 단계들을 쿠버네티스 리소스로 표준화해서, 컨테이너로 패키징하고 클러스터의 자원(CPU/GPU)을 필요한 만큼 확장해 쓸 수 있게 해준다.

Kubeflow는 하나의 프로그램이 아니라 여러 컴포넌트의 묶음이다. 대표적으로:

* Kubeflow Pipelines - ML 워크플로를 DAG로 정의·실행    
* Katib - 하이퍼파라미터 튜닝 - AutoML
* KServe - 학습된 모델 서빙     
* Kubeflow Trainer (구 Training Operator) - 분산 학습 실행 (이 실습에서 사용)
  
즉 우리가 이 실습에서 쓰는 것은 Kubeflow 전체가 아니라, 그중 분산 학습을 담당하는 Trainer 컴포넌트다.

### 2. Kubeflow Trainer 란? ###

Kubeflow Trainer는 분산 학습(distributed training) 작업을 쿠버네티스 위에서 실행·관리해주는 컴포넌트다. PyTorch DDP, TensorFlow, XGBoost 같은 프레임워크의 분산 학습을 클러스터에 띄우는 일을 담당한다. (이전 이름은 Training Operator이며, PyTorchJob·TFJob 등을 제공하던 것이 이것이다.)

Trainer가 하는 일을 한 문장으로 요약하면 이렇다 :

▎ 여러 노드에 학습 프로세스를 띄우고, 서로를 찾아 연결하고, 환경변수(RANK, WORLD_SIZE, MASTER_ADDR)를 주입하고, 실패를 감시하는 성가신 클러스터 작업을 대신 처리해준다.

원래 분산 학습을 직접 돌리려면 각 노드에서 torchrun을 실행하며 --nnodes, --node-rank, --rdzv-endpoint 같은 인자를 노드마다 다르게 지정해야 한다. 노드가 많아질수록 이 조율은 번거롭고 실수가 잦다. Trainer는 이것을 선언적으로 바꾼다 — 사용자는 "몇 개의 노드에, 어떤 이미지로, 무슨 명령을 돌릴지"만 선언하고, 프로세스 기동과 노드 연결은 Trainer가 맡는다.

Trainer는 두 세대가 있다.

- V1 (Training Operator): 프레임워크마다 별도 리소스(PyTorchJob, TFJob…)를 쓰고, 노드를 Master/Worker 역할로 나눠 지정한다.
- V2 (Kubeflow Trainer): 프레임워크와 무관하게 TrainJob 하나로 통합했고, 역할 개념 없이 numNodes 숫자만으로 규모를 선언한다. Python SDK가 주 인터페이스이며, 공통 설정은 ClusterTrainingRuntime 으로 재사용한다.

Trainer는 학습 코드를 대체하지 않는다. 모델 학습 로직은 여전히 우리가 짠 PyTorch DDP 코드(train.py) 이고, Trainer는 그 코드를 여러 파드에 걸쳐 띄우고 연결해주는 실행 인프라일 뿐이다. 즉 train.py는 어디서든 그대로 두고, 실행만 Trainer에게 맡기는 구조다.

### 3/ 분산 학습의 3계층 ###

분산 학습은 세 개의 층으로 나뉘어서 동작한다.

`① 코드 계층 (PyTorch DDP)` — 모델을 어떻게 여러 프로세스가 나눠 학습하고 그래디언트를 동기화할지는 PyTorch DDP로
구현한다. 이 train.py는 어디서 돌리든 바뀌지 않는다. EC2에서 돌리든 쿠버네티스에서 돌리든 동일한 코드다.

`② 런처 계층 (torchrun)` — DDP 프로세스를 실제로 띄우고 각 프로세스에 RANK·WORLD_SIZE·MASTER_ADDR 같은 환경변수를
주입하는 역할은 torchrun이 담당한다. train.py가 이 값들을 읽어 dist.init_process_group()만 호출하면 되는 것도
torchrun이 뒤에 있기 때문이다.

`③ 오케스트레이션 계층` — torchrun을 "누가, 몇 개의 노드에, 어떻게 연결해서" 실행하느냐가 여기서 갈린다. 그리고 EC2와
쿠버네티스의 차이는 바로 이 층에만 있다.
  - EC2: 사용자가 각 노드에 직접 torchrun train.py를 실행한다. 이때 --nnodes, --node-rank, --rdzv-endpoint 같은 노드별
인자도 직접 지정.
  - 쿠버네티스 + Kubeflow: Kubeflow Trainer가 이 일을 대신한다. numNodes 숫자만 알려주면, 내부적으로 각 파드에서
torchrun train.py를 조립해 실행하고 노드 간 주소 연결까지 자동으로 처리.

즉 Kubeflow가 torchrun을 대체하는 것이 아니라, torchrun을 여러 파드에 걸쳐 자동으로 세팅·실행해주는 상위 오케스트레이터다. 같은 torchrun이 도는데, "누가 그것을 띄우느냐"만 다르다.

### 4. Kubeflow Trainer 설치 ###

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


kubectl delete trainjob <job-name> -n kubeflow-user
```
