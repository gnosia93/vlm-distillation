
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

### 3. 분산 학습의 3계층 ###

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


## 4. Kubeflow Trainer 설치 ##

* https://github.com/kubeflow/trainer/releases
```
sudo dnf install git -y
export VERSION=v2.2.1
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/manager?ref=${VERSION}"
```
30 초 정도 지난 후에 클러스터 트레이닝런타임을 설치한다. 
```
kubectl apply --server-side -k "https://github.com/kubeflow/trainer.git/manifests/overlays/runtimes?ref=${VERSION}"
kubectl get clustertrainingruntimes
```
[결과]
```
NAME                     AGE
deepspeed-distributed    9s
mlx-distributed          9s
torch-distributed        9s
torchtune-llama3.2-1b    9s
torchtune-llama3.2-3b    9s
torchtune-qwen2.5-1.5b   9s
```

**_(Optional)_** torch-distributed 런타임을 아래와 같이 수정한다. 
```
kubectl edit clustertrainingruntime torch-distributed 
```
```
apiVersion: trainer.kubeflow.org/v1alpha1
kind: ClusterTrainingRuntime
metadata:
  name: torch-distributed
spec:
  template:
    spec:
      # ❌ shareProcessNamespace: true   → 제거 (사이드카/디버깅용, 학습엔 불필요)
      # ❌ hostIPC: true                 → 제거 (아래 /dev/shm emptyDir로 대체)
      containers:
        - name: node
          securityContext:
            # ❌ privileged: true        → 제거 (EFA device plugin으로 대체)
            capabilities:
              add: ["IPC_LOCK"]          # ✅ 유지 - RDMA 메모리 핀(필수)
          resources:
            limits:
              vpc.amazonaws.com/efa: 1   # ✅ EFA 디바이스를 플러그인으로 노출
              nvidia.com/gpu: 8          # 인스턴스 GPU 수에 맞게
            requests:
              vpc.amazonaws.com/efa: 1
              nvidia.com/gpu: 8
          volumeMounts:
            - name: dshm
              mountPath: /dev/shm        # ✅ NCCL 공유메모리
      volumes:
        - name: dshm
          emptyDir:
            medium: Memory               # 호스트 IPC 대신 메모리 볼륨
            sizeLimit: 16Gi              # 모델/배치에 맞게 조정
```
* ClusterTrainingRuntime은 플랫폼 관리자가 관리하는 공용 템플릿이고, 실제 학습을 실행하는 TrainJob에서 필요한 부분만 덮어씁니다. 이때 덮어쓰는 방법이 필드의 성격에 따라 두 갈래로 나뉩니다. 학습의 핵심 파라미터는 안정적으로 관리되어야 하는 반면, 파드 배치나 스토리지 같은 인프라 설정은 유연하게 바뀔 수 있어야 하기 때문입니다
```
① spec.trainer (Trainer API) — node(trainer) 컨테이너의 핵심 값
image, command, args, resources(GPU/EFA 개수), numNodes, env
이건 반드시 Trainer API로만 덮어써야 함

② spec.runtimePatches (RuntimePatches API) — 그 외 파드/컨테이너 스펙
volumes, volumeMounts, nodeSelector, tolerations, serviceAccount, securityContext, labels, annotations
이 API는 예전의 podTemplateOverrides를 대체한 것으로, 여러 주체(사용자·Kueue·admission webhook)가 각자 이름(manager) 아래 패치를 기여하고 이를 병합하는 다중 소유(multi-owner) 모델을 씁니다. 설치된 Trainer 버전에 따라 runtimePatches(신규)일 수도, podTemplateOverrides(구)일 수도 있습니다.
```
* privileged: true 는 호스트 시스템의 모든 리소스(디바이스)와 커널 기능에 대한 완전한 접근 권한을 부여하는 설정이다.(보안상 이 설정은 하지 말아야 한다)

    
> [!NOTE]
> _멀티 노드 분산 학습이 아닌 싱글 노드 분산 학습(예: 하나의 노드 안에서 GPU 1~8장으로 훈련)의 경우, 노드 간 조율이 필요 없으므로 Kubeflow Trainer를 설치할 필요가 없다. 이때는 하나의 노드안에서 torchrun --nproc_per_node=<GPU 수>만으로 노드 내 프로세스를 띄우고 NCCL이 GPU 간 통신을 처리하면 충분하다._ 


아래 명령어로 설치 여부를 확인한다. 
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

