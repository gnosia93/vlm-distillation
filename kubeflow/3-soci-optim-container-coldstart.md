*** 아래 내용은 아직 검증이 되지 않은 상태로, 테스트가 필요하다 (테스트 예정) ***

## Seekable OCI lazy loading으로 GPU 파드 기동 가속하기 ##

* 문제: 노드는 90초 만에 뜨는데, 파드는 10분째 대기 중

EKS에서 GPU 워크로드를 스케일아웃하다 보면 이상한 지점을 만나게 됩니다. Karpenter가 새 GPU 노드를 프로비저닝하고 클러스터에 조인시키는 데까지는 몇 분이면 충분한데, 정작 파드가 ContainerCreating에서 한참을 머무릅니다. 로그를 열어보면 범인은 대부분 하나, 이미지 pull입니다.

우리가 훈련에 쓰는 이미지를 떠올려 보죠. PyTorch + CUDA + 각종 라이브러리가 들어간 Deep Learning Container(DLC)는 가볍게 잡아도 수 GB, 모델 가중치까지 번들하면 수십 GB까지 갑니다. 문제는 기존 컨테이너 런타임의 동작 방식입니다.
```
[기존 방식] 이미지 전체 pull 완료 → 압축 해제 → 그제서야 컨테이너 시작
```
10GB짜리 이미지를 통째로 내려받고 풀어야 컨테이너가 뜹니다. AWS 자체 데이터로도 큰 이미지에서는 이미지 pull이 전체 컨테이너 기동 시간의 75% 이상을 차지한다고 합니다. 오토스케일로 노드가 떴다 사라지는 환경에서는 이 콜드 스타트가 매번 반복되니, 스케일아웃의 반응성을 통째로 갉아먹는 셈이죠.

그런데 생각해보면 컨테이너를 시작하는 데 이미지의 모든 파일이 당장 필요한 건 아닙니다. 여기서 SOCI가 등장합니다.
**SOCI(Seekable OCI, "소치"로 읽습니다)**는 AWS가 만든 오픈소스 기술로, 컨테이너 이미지를 lazy loading(지연 로딩) 하게 해줍니다. 핵심 아이디어는 단순합니다.
```
[SOCI 방식] 컨테이너 즉시 시작 → 실제로 필요한 파일 블록만 그때그때 가져옴
```
원리는 이렇습니다. 컨테이너 이미지 레이어는 보통 gzip으로 압축된 tar라, 파일 하나만 꺼내려 해도 전체를 풀어야 합니다(random access가 안 됨). SOCI는 이미지와 별도로 인덱스를 만들어서, 압축된 레이어 안의 특정 위치로 바로 seek해서 필요한 부분만 읽을 수 있게 합니다. 그래서 이름이 "Seekable OCI"예요.

중요한 건 원본 이미지를 변형하지 않는다는 점입니다. 인덱스만 부가적으로 얹기 때문에, 우리가 쓰던 태그 뒤에 -soci 같은 접미사가 붙는 형태로 배포됩니다. 예를 들면:
```
public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.48-soci
```
이 태그의 -soci가 바로 "이 이미지에는 SOCI 인덱스가 딸려 있다"는 표시입니다. 이미 인덱스가 있으니, 우리는 노드가 이걸 활용하도록 세팅만 해주면 됩니다.

참고로 SOCI는 실험적 기술이 아닙니다. Amazon ECS Fargate와 EKS 프로덕션에서 2023년부터 lazy load 요청을 처리해왔고, EKS Auto Mode는 GPU 인스턴스에 SOCI의 parallel pull 모드를 기본으로 씁니다.

SOCI는 AMI에 내장된 기능이 아니라, 노드에서 돌아가는 데몬(soci-snapshotter-grpc) + containerd 설정입니다. OS가 AL2023이든 Bottlerocket이든 결국 세 가지를 해줘야 합니다.

* 스냅샷터 데몬 설치
* containerd가 그 스냅샷터를 쓰도록 설정
* kubelet의 image service를 스냅샷터 소켓으로 지정

### Karpenter EC2NodeClass에 SOCI 붙이기 ###
우리 워크샵은 Karpenter로 GPU 노드를 프로비저닝합니다. 아래와 같이 userData 섹션을 추가하는 경우, 
Karpenter는 AL2023 노드에서 우리가 준 userData를 자기가 생성하는 클러스터 조인 설정과 병합합니다. 덕분에 조인용 설정은 신경 쓸 필요 없이, SOCI 관련 부분만 넣으면 됩니다.

```
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: gpu
spec:
  role: "eksctl-KarpenterNodeRole-${CLUSTER_NAME}"
  amiSelectorTerms:
    - alias: al2023@latest        # GPU 파드 요청 시 accelerated 변형 자동 선택
  subnetSelectorTerms:
    - tags:
        karpenter.sh/discovery: "${CLUSTER_NAME}"
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: "${CLUSTER_NAME}"
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 600Gi         # 대형 DLC를 감당할 넉넉한 루트 볼륨
        volumeType: gp3

  ##### 아래 userData 부분을 기존 설정에 추가합니다.  #####
  userData: |
    MIME-Version: 1.0
    Content-Type: multipart/mixed; boundary="BOUNDARY"

    --BOUNDARY
    Content-Type: application/node.eks.aws

    ---
    apiVersion: node.eks.aws/v1alpha1
    kind: NodeConfig
    spec:
      kubelet:
        config:
          # kubelet의 이미지 서비스를 SOCI 소켓으로 지정
          imageServiceEndpoint: unix:///run/soci-snapshotter-grpc/soci-snapshotter-grpc.sock
      containerd:
        config: |
          [proxy_plugins.soci]
            type = "snapshot"
            address = "/run/soci-snapshotter-grpc/soci-snapshotter-grpc.sock"
            [proxy_plugins.soci.exports]
              root = "/var/lib/soci-snapshotter-grpc"
          [plugins."io.containerd.grpc.v1.cri".containerd]
            snapshotter = "soci"
            # lazy load 정보를 스냅샷터에 전달하기 위해 필수
            disable_snapshot_annotations = false

    --BOUNDARY
    Content-Type: text/x-shellscript; charset="us-ascii"

    #!/bin/bash
    set -euo pipefail
    ARCH=$(uname -m | sed s/aarch64/arm64/ | sed s/x86_64/amd64/)
    VERSION="0.15.0"     # 최신 안정 버전으로 교체
    ARCHIVE=soci-snapshotter-${VERSION}-linux-${ARCH}.tar.gz
    cd /tmp
    curl -sSL -o $ARCHIVE \
      https://github.com/awslabs/soci-snapshotter/releases/download/v${VERSION}/${ARCHIVE}
    tar xzf ./$ARCHIVE -C /usr/local/bin soci-snapshotter-grpc

    # 프라이빗 ECR 자격증명을 CRI keychain으로 캐싱
    mkdir -p /etc/soci-snapshotter-grpc
    cat > /etc/soci-snapshotter-grpc/config.toml <<'CFG'
    [cri_keychain]
    enable_keychain = true
    image_service_path = "/run/containerd/containerd.sock"
    CFG

    # systemd 서비스 등록 및 기동
    curl -sSL -o /etc/systemd/system/soci-snapshotter.service \
      https://raw.githubusercontent.com/awslabs/soci-snapshotter/v${VERSION}/soci-snapshotter.service
    systemctl daemon-reload
    systemctl enable --now soci-snapshotter

    --BOUNDARY--
```
이 userData가 하는 일을 풀어보면:

* 첫 번째 블록(NodeConfig): containerd에 soci라는 proxy 스냅샷터를 등록하고, 기본 스냅샷터로 지정합니다. kubelet도 이미지 작업을 SOCI 소켓으로 보내도록 바꿉니다.
* 두 번째 블록(셸 스크립트): 노드 부팅 시 soci-snapshotter-grpc 바이너리를 내려받아 설치하고, 프라이빗 레지스트리 인증 설정을 만든 뒤, systemd 서비스로 데몬을 띄웁니다.

즉 블록을 통째로 교체하는 게 아니라, 기존 EC2NodeClass에 이 userData 한 덩어리를 얹는 것으로 끝납니다.

#### 제대로 붙었는지 확인하기 ####
스냅샷터는 Kubernetes 입장에서 직접 보이지 않습니다. kubectl로는 확인할 방법이 없어요. 대신 두 가지로 검증합니다.

* 첫째, 파드 기동 시간입니다. SOCI가 잘 붙었다면 같은 이미지의 파드가 눈에 띄게 빨리 뜹니다. 이게 가장 체감되는 지표죠.
* 둘째, 확실히 하려면 노드에 들어가 SOCI 파일시스템 마운트를 확인합니다. SSM으로 노드에서 다음을 실행하면 됩니다.
```
findmnt --source soci
```
[결과]
```
TARGET                                          SOURCE FSTYPE         OPTIONS
/var/lib/soci-snapshotter-grpc/.../fs           soci   fuse.rawBridge rw,nosuid,...
...
```

세팅을 마쳤는데 가속이 안 된다면, 십중팔구 아래 둘 중 하나입니다.

1. 이미지에 SOCI 인덱스가 있어야 한다. 노드에 스냅샷터를 깔아도 이미지에 인덱스가 없으면 lazy load가 안 되고, 조용히 일반 pull로 fallback합니다(동작은 하지만 가속은 없음). 다행히 우리가 쓰는 -soci 태그 DLC는 인덱스가 이미 포함돼 있어 이 조건을 만족합니다. 직접 빌드한 이미지라면 soci create로 인덱스를 만들어 ECR에 push해야 합니다.

2. 프라이빗 레지스트리 자격증명이 스냅샷터에 전달돼야 한다. 위 설정의 [cri_keychain] 블록이 그 역할입니다. 프라이빗 ECR을 쓰는데 이 부분이 빠지면 pull 자체가 실패합니다.



### 실제로 얼마나 빨라지나: Before / After 측정 ###

* 대상 이미지: pytorch-training:2.8.0-gpu-py312-cu129-... (약 __GB)
* 인스턴스 타입: __________ (예: g5.12xlarge)
* 측정 구간: 파드가 스케줄된 시점부터 컨테이너가 Running이 될 때까지
* 비교 조건: SOCI 미적용(일반 pull) vs SOCI 적용(lazy loading)

#### 콜드 스타트 시간 비교 ####
이미지 크기	SOCI 미적용 (일반 pull)	SOCI 적용 (lazy loading)	단축율

__ GB	__ 분 __ 초	__ 분 __ 초	__ %

__ GB	__ 분 __ 초	__ 분 __ 초	__ %

#### 기동 단계별 소요 시간 분해 ####

콜드 스타트를 단계별로 쪼개 보면 어디서 시간이 줄었는지 명확해집니다.

단계	SOCI 미적용	SOCI 적용	비고

노드 프로비저닝 (Karpenter)	__ 초	__ 초	SOCI와 무관, 동일

노드 조인 (bootstrap)	__ 초	__ 초	SOCI와 무관, 동일

이미지 pull / 준비	__ 초	__ 초	← SOCI가 줄이는 구간

컨테이너 시작 ~ Running	__ 초	__ 초	워크로드 의존

총합	__ 초	__ 초	

#### 측정 방법 ####

1) 파드 이벤트 타임스탬프로 측정 (가장 간단)

```
# 파드의 스케줄 → Pulling → Started 이벤트 시각 확인
kubectl describe pod <pod-name> | grep -A2 -E "Scheduled|Pulling|Pulled|Started"

# 또는 이벤트를 시간순으로
kubectl get events --field-selector involvedObject.name=<pod-name> \
  --sort-by='.lastTimestamp'
```
* Pulling → Pulled 사이 간격이 이미지 pull 시간, Scheduled → Started 전체가 콜드 스타트입니다.

2) 반복 측정 (편차 제거)

콜드 스타트는 네트워크·레지스트리 상태에 따라 편차가 있으니, 같은 조건을 최소 3~5회 반복해 평균과 최소/최대를 함께 기록하는 게 좋습니다. 노드가 확실히 새로 뜨도록 매 측정 전 기존 노드를 정리(scale to 0)하고 시작하세요.

```
# 측정마다 새 노드에서 시작되도록 배포 삭제 후 재생성
kubectl delete deployment <name>
# (노드가 스케일다운될 때까지 대기)
kubectl apply -f deployment.yaml
```

3) 공정한 비교를 위한 조건 통일

* 같은 인스턴스 타입, 같은 AZ, 같은 이미지
* SOCI 미적용 측정 시엔 캐시된 이미지가 없는 완전한 콜드 상태에서 (레이어 캐시가 남아있으면 비교가 왜곡됨)
* 같은 시간대에 측정 (레지스트리 부하 변수 최소화)

```
약 __GB DLC 이미지 기준, 콜드 스타트가 __분 __초 → __분 __초로 약 __% 단축되었습니다. 오토스케일로 GPU 노드가 자주 교체되는 환경에서, 스케일아웃당 __분을 절약하는 효과입니다.
```

#### 시간 측정 자동화 스크립트 ####
측정 자동화 bash 스크립트입니다. 파드를 여러 번 띄우면서 매번 새 노드에서 콜드 스타트되도록 강제하고, 이벤트 타임스탬프를 수집해 평균/최소/최대까지 표로 뽑아줍니다
```
#!/usr/bin/env bash
#
# measure-cold-start.sh
# SOCI 적용 전/후 컨테이너 콜드 스타트 시간을 측정한다.
#
# 측정 구간:
#   - schedule -> pulling : 스케줄 후 pull 시작까지
#   - pull (Pulling->Pulled): 이미지 pull(=SOCI가 줄이는 구간)
#   - total (created->Running): 전체 콜드 스타트
#
# 매 반복마다 파드가 돌던 노드를 삭제해 '완전한 콜드 상태'에서 다시 측정한다.
#
# 사용법:
#   ./measure-cold-start.sh \
#       -i public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.48-soci \
#       -n 5 \
#       -g "nvidia.com/gpu"        # (선택) GPU 리소스 요청 키
#
set -euo pipefail

# ----------------------------- 기본값 -----------------------------
IMAGE=""
ITERATIONS=3
NAMESPACE="default"
GPU_RESOURCE=""              # 예: "nvidia.com/gpu" (비우면 GPU 요청 없음)
NODE_SELECTOR_KEY=""         # 예: "karpenter.sh/nodepool"
NODE_SELECTOR_VAL=""         # 예: "gpu"
POD_NAME="soci-bench"
WAIT_TIMEOUT=1200            # Running 대기 최대 초 (20분)
FRESH_NODE=true              # 매 반복 새 노드 강제 여부

usage() {
  grep '^#' "$0" | sed 's/^#//'
  exit 1
}

# ----------------------------- 인자 파싱 -----------------------------
while getopts "i:n:N:g:k:v:t:F" opt; do
  case "$opt" in
    i) IMAGE="$OPTARG" ;;
    n) ITERATIONS="$OPTARG" ;;
    N) NAMESPACE="$OPTARG" ;;
    g) GPU_RESOURCE="$OPTARG" ;;
    k) NODE_SELECTOR_KEY="$OPTARG" ;;
    v) NODE_SELECTOR_VAL="$OPTARG" ;;
    t) WAIT_TIMEOUT="$OPTARG" ;;
    F) FRESH_NODE=false ;;      # -F 주면 노드 재사용(웜) 측정
    *) usage ;;
  esac
done

[[ -z "$IMAGE" ]] && { echo "ERROR: -i <image> 필수"; usage; }
command -v kubectl >/dev/null || { echo "ERROR: kubectl 없음"; exit 1; }
command -v jq >/dev/null      || { echo "ERROR: jq 없음 (brew install jq)"; exit 1; }

# ------------------- RFC3339 -> epoch (macOS/Linux 호환) -------------------
to_epoch() {
  local ts="$1"
  [[ -z "$ts" || "$ts" == "null" ]] && { echo ""; return; }
  # GNU date 먼저 시도, 실패하면 BSD(macOS) date
  if date -d "$ts" +%s >/dev/null 2>&1; then
    date -d "$ts" +%s
  else
    # BSD date: 소수점/타임존 제거 후 파싱
    local clean="${ts%.*}"; clean="${clean%Z}"
    date -j -u -f "%Y-%m-%dT%H:%M:%S" "$clean" +%s 2>/dev/null || echo ""
  fi
}

# ----------------------------- 파드 매니페스트 -----------------------------
render_pod() {
  local resources_block="" selector_block=""
  if [[ -n "$GPU_RESOURCE" ]]; then
    resources_block=$(cat <<YAML
    resources:
      limits:
        ${GPU_RESOURCE}: "1"
YAML
)
  fi
  if [[ -n "$NODE_SELECTOR_KEY" ]]; then
    selector_block=$(cat <<YAML
  nodeSelector:
    ${NODE_SELECTOR_KEY}: "${NODE_SELECTOR_VAL}"
YAML
)
  fi

  cat <<YAML
apiVersion: v1
kind: Pod
metadata:
  name: ${POD_NAME}
  namespace: ${NAMESPACE}
  labels:
    app: soci-bench
spec:
  restartPolicy: Never
  terminationGracePeriodSeconds: 0
${selector_block}
  containers:
  - name: bench
    image: ${IMAGE}
    command: ["sleep", "infinity"]
${resources_block}
YAML
}

# ----------------------------- 정리 함수 -----------------------------
cleanup_pod() {
  kubectl delete pod "$POD_NAME" -n "$NAMESPACE" --ignore-not-found --wait=false >/dev/null 2>&1 || true
}

delete_node() {
  local node="$1"
  [[ -z "$node" || "$node" == "null" ]] && return
  echo "  -> 노드 삭제(콜드 상태 강제): $node"
  # Karpenter면 nodeclaim까지 지워야 인스턴스가 종료됨
  local nc
  nc=$(kubectl get nodeclaim -o json 2>/dev/null \
        | jq -r --arg n "$node" '.items[]? | select(.status.nodeName==$n) | .metadata.name' \
        | head -n1 || true)
  kubectl delete node "$node" --ignore-not-found --wait=false >/dev/null 2>&1 || true
  [[ -n "$nc" ]] && kubectl delete nodeclaim "$nc" --ignore-not-found --wait=false >/dev/null 2>&1 || true
}

trap cleanup_pod EXIT

# ----------------------------- 결과 저장 배열 -----------------------------
declare -a A_SCHED A_PULL A_TOTAL

# ----------------------------- 측정 루프 -----------------------------
echo "================================================================"
echo " SOCI 콜드 스타트 측정"
echo "   image      : $IMAGE"
echo "   iterations : $ITERATIONS"
echo "   fresh node : $FRESH_NODE"
echo "================================================================"
printf "%-5s %-14s %-14s %-14s\n" "RUN" "SCHED->PULL" "PULL(이미지)" "TOTAL"
echo "----------------------------------------------------------------"

for ((run=1; run<=ITERATIONS; run++)); do
  cleanup_pod
  sleep 5

  render_pod | kubectl apply -f - >/dev/null

  # Running 될 때까지 대기
  if ! kubectl wait --for=condition=Ready "pod/${POD_NAME}" \
        -n "$NAMESPACE" --timeout="${WAIT_TIMEOUT}s" >/dev/null 2>&1; then
    echo "  [run $run] 타임아웃 또는 실패, 스킵"
    NODE=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o jsonpath='{.spec.nodeName}' 2>/dev/null || true)
    cleanup_pod
    [[ "$FRESH_NODE" == true ]] && delete_node "$NODE"
    continue
  fi

  # --- 타임스탬프 수집 ---
  POD_JSON=$(kubectl get pod "$POD_NAME" -n "$NAMESPACE" -o json)
  NODE=$(echo "$POD_JSON" | jq -r '.spec.nodeName')

  CREATED=$(echo "$POD_JSON" | jq -r '.metadata.creationTimestamp')
  SCHEDULED=$(echo "$POD_JSON" | jq -r '.status.conditions[]? | select(.type=="PodScheduled") | .lastTransitionTime')
  STARTED=$(echo "$POD_JSON" | jq -r '.status.containerStatuses[0].state.running.startedAt')

  # Pulling / Pulled 이벤트 시각
  EV_JSON=$(kubectl get events -n "$NAMESPACE" \
              --field-selector "involvedObject.name=${POD_NAME}" -o json)
  PULLING=$(echo "$EV_JSON" | jq -r '[.items[] | select(.reason=="Pulling")][0].firstTimestamp // empty')
  PULLED=$(echo  "$EV_JSON" | jq -r '[.items[] | select(.reason=="Pulled")][0].firstTimestamp // empty')

  # --- epoch 변환 ---
  e_sched=$(to_epoch "$SCHEDULED")
  e_pulling=$(to_epoch "$PULLING")
  e_pulled=$(to_epoch "$PULLED")
  e_created=$(to_epoch "$CREATED")
  e_started=$(to_epoch "$STARTED")

  # --- 구간 계산 (실패한 구간은 - 로 표시) ---
  d_sched="-"; d_pull="-"; d_total="-"
  [[ -n "$e_sched" && -n "$e_pulling" ]] && d_sched=$(( e_pulling - e_sched ))
  [[ -n "$e_pulling" && -n "$e_pulled" ]] && d_pull=$(( e_pulled - e_pulling ))
  [[ -n "$e_created" && -n "$e_started" ]] && d_total=$(( e_started - e_created ))

  printf "%-5s %-14s %-14s %-14s\n" "$run" "${d_sched}s" "${d_pull}s" "${d_total}s"

  [[ "$d_sched" != "-" ]] && A_SCHED+=("$d_sched")
  [[ "$d_pull"  != "-" ]] && A_PULL+=("$d_pull")
  [[ "$d_total" != "-" ]] && A_TOTAL+=("$d_total")

  # 다음 측정을 위해 노드 정리
  cleanup_pod
  if [[ "$FRESH_NODE" == true ]]; then
    delete_node "$NODE"
    echo "  (새 노드 프로비저닝 대기 중...)"
    sleep 20
  fi
done

# ----------------------------- 통계 -----------------------------
stats() {
  local -n arr=$1
  local n=${#arr[@]}
  (( n == 0 )) && { echo "N/A"; return; }
  local sum=0 min=${arr[0]} max=${arr[0]}
  for v in "${arr[@]}"; do
    sum=$(( sum + v ))
    (( v < min )) && min=$v
    (( v > max )) && max=$v
  done
  # 평균 소수 1자리
  local avg
  avg=$(awk "BEGIN{printf \"%.1f\", $sum/$n}")
  echo "avg ${avg}s / min ${min}s / max ${max}s (n=$n)"
}

echo "================================================================"
echo " 요약"
echo "----------------------------------------------------------------"
printf "  SCHED->PULL    : %s\n" "$(stats A_SCHED)"
printf "  PULL(이미지)   : %s   <- SOCI가 줄이는 구간\n" "$(stats A_PULL)"
printf "  TOTAL          : %s\n" "$(stats A_TOTAL)"
echo "================================================================"
echo "TIP: SOCI 미적용/적용 각각 실행 후 PULL(이미지) 평균을 비교하세요."
```

* 사용 방법
```
chmod +x measure-cold-start.sh

./measure-cold-start.sh \
  -i public.ecr.aws/deep-learning-containers/pytorch-training:2.8.0-gpu-py312-cu129-ubuntu22.04-ec2-v1.48-soci \
  -n 5 \
  -g "nvidia.com/gpu" \
  -k "karpenter.sh/nodepool" -v "gpu-nosoci"
```
* -i	이미지 (필수)	-i public.ecr.aws/.../pytorch:...-soci
* -n	반복 횟수	-n 5
* -N	네임스페이스	-N ml
* -g	GPU 리소스 키	-g "nvidia.com/gpu"
* -k/-v	nodeSelector 키/값	-k karpenter.sh/nodepool -v gpu
* -t	Running 대기 타임아웃(초)	-t 1800
* -F	노드 재사용(웜 측정)	붙이면 노드 안 지움



### 마치며 ###
큰 이미지를 다루는 ML 워크로드에서 콜드 스타트의 진짜 병목은 노드 프로비저닝이 아니라 이미지 pull입니다. SOCI는 "이미지를 다 받고 시작"하던 방식을 "필요한 것만 그때그때 seek"하는 방식으로 바꿔서 이 병목을 줄여줍니다.

#### 핵심을 세 줄로 요약하면: ####

* SOCI는 AMI 기능이 아니라 노드의 스냅샷터 데몬 + containerd 설정이다. AMI를 바꾼다고 켜지지 않는다.
* Karpenter에서는 EC2NodeClass를 교체하는 게 아니라 spec.userData를 추가해서 붙인다.
* 이미지에 SOCI 인덱스가 있고(-soci 태그) 실행 환경이 스냅샷터를 지원할 때만 실제로 가속된다.

오토스케일로 GPU 노드가 수시로 떴다 지는 환경이라면, 이 한 번의 세팅이 매 스케일아웃마다 몇 분씩을 돌려줍니다. 특히 대형 VLM 이미지를 여러 노드에 뿌리는 우리 파이프라인에서는 그 효과가 곧바로 체감될 겁니다.

## 레퍼런스 ##

* https://aditmodi.hashnode.dev/soci-snapshotter-on-eks-eliminating-the-8-minute-gpu-cold-start-nobody-talks-about
* [Seekable OCI: Lazy-Loading Container Images via Range-Request Indexing](https://arxiv.org/abs/2607.06868)





   
 



