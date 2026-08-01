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
```

arpenter는 EC2 Fleet처럼 allocationStrategy 필드를 직접 노출하지 않습니다. 내부적으로 이미 price-capacity-optimized(가격+용량 균형, 중단 최소화 지향)를 씁니다. 그래서 "capacity 위주"의 실질적 레버는 필드 설정이 아니라 인스턴스 타입/AZ 다양화예요.

핵심: Karpenter의 Spot 전략은 이미 price-capacity-optimized
AWS 문서 기준 price-capacity-optimized는 **"중단 가능성이 가장 낮은 풀 + 가능한 낮은 가격"**을 함께 봅니다 — 대부분의 Spot 워크로드에 권장되는 전략.
Karpenter는 후보 인스턴스 타입들을 EC2 Fleet에 넘길 때 이 전략으로 provisioning → 이미 capacity(중단 최소화)를 고려하고 있어요.
순수 capacity-optimized(가격 무시, 최심 풀만)는 Karpenter가 별도 필드로 안 열어줍니다.







### Indexed Job (배치 인퍼런스, Spot 내성) ###
```
apiVersion: batch/v1
kind: Job
metadata: { name: vlm-batch-infer }
spec:
  completions: 100          # 총 100 샤드
  parallelism: 20           # 동시에 20개 Pod
  completionMode: Indexed
  backoffLimitPerIndex: 3   # 샤드별 재시도 3회
  maxFailedIndexes: 5
  podFailurePolicy:
    rules:
      - action: Ignore                        # Spot 중단은 재시도 카운트 제외
        onPodConditions:
          - type: DisruptionTarget
  template:
    spec:
      restartPolicy: Never
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

* 고정 배치면: Karpenter GPU NodePool(Spot·다중타입·scale-to-0) + Indexed Job(샤드) + 멱등 워커. SQS 불필요.
* Spot 내성: podFailurePolicy로 중단 무시 + backoffLimitPerIndex + 워커 멱등성.
* 비용: 끝나면 Karpenter가 노드를 0으로 내림.
* (동적 유입이면 → SQS + KEDA로 확장)
