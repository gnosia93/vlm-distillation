### s3 버킷 구조 ##
```
s3://<bucket>/
├── raw/                                  # ① 원본 CCTV 영상 (자동 업로드)
│   └── {camera_id}/{date}/{video_id}.mp4
│
├── frames/                               # ② 샘플링 결과 (CPU/Graviton 단계)
│   └── {video_id}/
│       ├── frames.json                   #    프레임 목록·메타
│       ├── frame_0001.jpg
│       ├── ...
│       └── events/                       #    이벤트 클립 (event_clips.py)
│           ├── event_000/
│           │   ├── frame_00.jpg ... frame_15.jpg
│           └── event_001/ ...
│
├── jobs/                                 # ③ 배치 인퍼런스 작업 정의
│   └── {job_id}/                         #    job_id = 날짜/실행ID 등
│       ├── manifest.jsonl                #    처리할 아이템 전체 목록
│       ├── shards/                       #    pre-shard (Indexed Job용)
│       │   ├── shard_00000.jsonl         #    각 Pod가 자기 것만 읽음
│       │   └── shard_00001.jsonl
│       └── _SUCCESS                      #    입력 준비 완료 마커 (트리거용)
│
├── outputs/                              # ④ 인퍼런스 결과 (멱등성 핵심)
│   └── {job_id}/
│       ├── {item_id}.json                #    결정적 경로 → "있으면 skip"
│       └── _SUCCESS                      #    잡 완료 마커
│
└── models/                               # ⑤ 모델 가중치 (선택)
    └── internvl3-1b/ , sam3/ ...

```



### 카펜터 GPU 노드풀 ###
```
apiVersion: karpenter.sh/v1
kind: NodePool
metadata: { name: gpu-spot }
spec:
  template:
    spec:
      requirements:
        - key: karpenter.sh/capacity-type
          operator: In
          values: ["spot", "on-demand"]             # Spot 우선, 없으면 OnDemand
        - key: node.kubernetes.io/instance-type
          operator: In
          values:                                   # 다중 GPU 타입
            - "g5.4xlarge"
            - "g5.12xlarge"
            - "g5.48xlarge"
            - "g6.4xlarge"
            - "g6.12xlarge"
            - "g6.48xlarge"        
      nodeClassRef: { group: karpenter.k8s.aws, kind: EC2NodeClass, name: gpu }
      taints:
        - key: nvidia.com/gpu
          effect: NoSchedule
  limits: { nvidia.com/gpu: "100" }
  disruption:
    consolidationPolicy: WhenEmpty
    consolidateAfter: 1m        # 비면 1분 후 노드 정리 → scale to 0

---
apiVersion: karpenter.k8s.aws/v1
kind: EC2NodeClass
metadata:
  name: gpu
spec:
  role: "eksctl-KarpenterNodeRole-${CLUSTER_NAME}"
  amiSelectorTerms:
    # Required; when coupled with a pod that requests NVIDIA GPUs or AWS Neuron
    # devices, Karpenter will select the correct AL2023 accelerated AMI variant
    # see https://aws.amazon.com/ko/blogs/containers/amazon-eks-optimized-amazon-linux-2023-accelerated-amis-now-available/
    # EKS GPU Optimized AMI: NVIDIA 드라이버와 CUDA 런타임만 포함된 가벼운 이미지 (Karpenter가 자동으로 선택 가능) 가 설치됨.
    # 특정 DLAMI 가 필요한 경우 - name : 필드에 정의해야 함. 
    - alias: al2023@latest
  subnetSelectorTerms:                          # 대상 서브넷을 태그로 찾으므로, 필요한 만큼의 서브넷(AZ)에 태깅한다.. (us-east-1의 경우 6개 AZ)
    - tags:
        karpenter.sh/discovery: vlm-distillation 
  securityGroupSelectorTerms:
    - tags:
        karpenter.sh/discovery: vlm-distillation 
  blockDeviceMappings:
    - deviceName: /dev/xvda
      ebs:
        volumeSize: 600Gi
        volumeType: gp3
```

Karpenter는 내부적으로 이미 price-capacity-optimized(가격+용량 균형, 중단 최소화 지향)를 전략을 사용하여 EC2 인스턴스를 프로비저닝 한다. 
**capacity-type + instance-type + AZ 조합이므로 충분한 인스턴스를 확보할 수 있다.** 

### [Weighted NodePools](https://karpenter.sh/docs/concepts/scheduling/#weighting-nodepools) ###

Karpenter의 NodePool weight는 1~100 사이 정수라서 우선순위를 최대 100단계까지 표현할 수 있으며(미지정 시 0), 실무에서는 100·50·10처럼 간격을 넓게 두어 나중에
중간 단계를 끼워넣을 수 있게 하는 것이 관례이다. 또한 NodePool 개수 자체에는 하드 제한이 없어, weight로 우선순위를 매긴 여러 NodePool을 원하는 만큼 폴백
체인으로 구성할 수 있다.
```
# ── NodePool 1: g7/g6/g5 우선 ──────────────────────────────
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
name: gpu-preferred
spec:
weight: 100                       # 먼저 시도
template:
  metadata:
    labels:
      gpu: "true"
  spec:
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
    requirements:
      - key: karpenter.k8s.aws/instance-family
        operator: In
        values: ["g7", "g6", "g5"]
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["spot", "on-demand"]     # spot도 쓰려면 "spot" 추가
      - key: kubernetes.io/arch
        operator: In
        values: ["amd64"]
    nodeClassRef:
      group: karpenter.k8s.aws
      kind: EC2NodeClass
      name: default               # 실제 EC2NodeClass 이름으로 교체
    expireAfter: 720h
disruption:
  consolidationPolicy: WhenEmptyOrUnderutilized
  consolidateAfter: 1m
---
# ── NodePool 2: g4dn 폴백 ─────────────────────────────────
apiVersion: karpenter.sh/v1
kind: NodePool
metadata:
name: gpu-fallback
spec:
weight: 10                        # g7/g6/g5 안 되면 여기로
template:
  metadata:
    labels:
      gpu: "true"
  spec:
    taints:
      - key: nvidia.com/gpu
        value: "true"
        effect: NoSchedule
    requirements:
      - key: karpenter.k8s.aws/instance-family
        operator: In
        values: ["g4dn"]
      - key: karpenter.sh/capacity-type
        operator: In
        values: ["spot", "on-demand"]
      - key: kubernetes.io/arch
        operator: In
        values: ["amd64"]
    nodeClassRef:
      group: karpenter.k8s.aws
      kind: EC2NodeClass
      name: default
    expireAfter: 720h
disruption:
  consolidationPolicy: WhenEmptyOrUnderutilized
  consolidateAfter: 1m
```
인스턴스 할당 시도 순위는  g7/g6/g5 spot → g7/g6/g5 on-demand → g4dn spot → g4dn on-demand 순이다. 


### Indexed Job (배치 인퍼런스, Spot 내성) ###
```
apiVersion: batch/v1
kind: Job
metadata: { name: vlm-batch-infer }
spec:
  completions: 100          # 총 100 샤드
  parallelism: 20           # 동시에 20개 Pod
  completionMode: Indexed
  backoffLimitPerIndex: 3   # 각 샤드는 3번까지 재시도
  maxFailedIndexes: 5       # 최종 실패한 샤드가 5개를 넘으면 → Job 전체 중단(Failed) 
  podFailurePolicy:
    rules:
      - action: Ignore                        # Spot 중단은 재시도 카운트 제외
        onPodConditions:
          - type: DisruptionTarget
  template:
    spec:
      restartPolicy: Never                    # 컨테이너 실패 → 그 Pod는 실패로 끝남 → Job이 새 Pod 생성해서 재시도 / podFailurePolicy 는 Never 일때만 동작
      serviceAccountName: infer-sa            # S3 접근 IRSA
      nodeSelector:
        karpenter.sh/nodepool: gpu-spot       # ← 이 NodePool의 노드에만 스케줄
      tolerations:
        - key: nvidia.com/gpu
          effect: NoSchedule                  # GPU 노드 taint 허용
      containers:
        - name: worker
          image: <ECR>/vlm-infer:latest
          resources:
            limits: { nvidia.com/gpu: 1 }
          env:
            - name: TOTAL_SHARDS
              value: "100"
            # JOB_COMPLETION_INDEX 는 Indexed Job이 자동 주입
```

### worker 로직 ###
```
import os
idx   = int(os.environ["JOB_COMPLETION_INDEX"])   # 내 샤드 번호 (자동 주입)
total = int(os.environ["TOTAL_SHARDS"])

items = load_manifest_from_s3()          # 전체 아이템 목록
mine  = items[idx::total]                # 내 샤드만 (stride 분할)

model = load_model()                     # A10G/L4 무관, CUDA면 동작
for it in mine:
    if output_exists_in_s3(it["id"]):    # 멱등성: 이미 한 건 skip
        continue
    result = run_inference(model, it)
    write_to_s3(it["id"], result)        # 결정적 출력 경로
```


### 실행하기 ### 
```
# 1) 아이템 manifest를 S3에 올림 (producer)
# 2) Job apply
kubectl apply -f vlm-batch-infer.yaml
# 3) Karpenter가 GPU Spot 노드 자동 provision → Pod 실행
kubectl get pods -w
# 4) 완료 후 노드 자동 축소 확인
kubectl get nodes
```

### 정리 ###

* 고정 배치면: Karpenter GPU NodePool(Spot·다중타입·scale-to-0) + Indexed Job(샤드) + 멱등 워커. 
* Spot 내성: podFailurePolicy로 중단 무시 + backoffLimitPerIndex + 워커 멱등성.
* 비용 최적화: 끝나면 Karpenter가 노드를 0으로 내림.
* 동적 유입인 경우 → SQS + KEDA로 확장 필요.
