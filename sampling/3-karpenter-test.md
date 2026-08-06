폴백이 실제로 동작하는지 테스트하는 방법으로 핵심은 **"우선순위 NodePool을 일부러 프로비저닝 불가 상태로 만들고, 폴백 NodePool로 넘어가는지 확인"** 한다.

### 방법 1: 일부러 용량 부족을 유발 (실전에 가장 가까움) ###

가장 확실한 방법은 g7/g6/g5를 잡을 수 없는 조건으로 만들어 폴백을 강제 트리거하는 것이다.

#### A. weight 높은 NodePool을 아예 못 뜨게 막기 ####

g7/g6/g5 NodePool의 requirements를 존재하지 않는/불가능한 조건으로 잠깐 바뀬더, 예를 들어:
```
# gpu-preferred 에 임시로 추가 → 매칭되는 인스턴스 0개
- key: karpenter.k8s.aws/instance-family
  operator: In
  values: ["g7"]        # g7이 아직 GA 아니거나 해당 리전에 없으면 자연스럽게 0개
```

또는 확실하게 막으려면:
```
- key: node.kubernetes.io/instance-type
  operator: In
  values: ["g5.99xlarge"]   # 존재하지 않는 타입 → 절대 못 뜸
```

이 상태에서 GPU Pod를 배포 → g4dn(fallback)으로 노드가 뜨는지 확인.

#### B. limits로 막기 (RI 시나리오 테스트) ####
```
# gpu-preferred
spec:
  weight: 100
  limits:
    nvidia.com/gpu: 0     # 이 NodePool은 GPU 0개까지만 → 즉시 폴백
```
→ Pod 배포 시 곧바로 g4dn으로 넘어가야 함.

#### 테스트용 Pod ####
```
apiVersion: apps/v1
kind: Deployment
metadata:
  name: gpu-test
spec:
  replicas: 1
  selector:
    matchLabels: { app: gpu-test }
  template:
    metadata:
      labels: { app: gpu-test }
    spec:
      tolerations:
        - key: nvidia.com/gpu
          operator: Equal
          value: "true"
          effect: NoSchedule
      containers:
        - name: cuda
          image: nvidia/cuda:12.4.0-base-ubuntu22.04
          command: ["sh", "-c", "sleep 3600"]
          resources:
            limits:
              nvidia.com/gpu: 1
```

#### 확인 명령어 ####

```
# 1. Karpenter가 어떤 NodePool로 NodeClaim을 만들었는지
kubectl get nodeclaim -o wide
kubectl get nodeclaim -o jsonpath='{range .items[*]}{.metadata.name}{"\t"}{.metadata.labels.karpenter\.sh/nodepool}{"\n"}{end}'

# 2. 실제 뜬 노드의 인스턴스 타입 / NodePool 확인
kubectl get nodes -L node.kubernetes.io/instance-type,karpenter.sh/nodepool

# 3. Pod가 어느 노드에 갔는지
kubectl get pod -l app=gpu-test -o wide

# 4. 폴백 판단 근거 = Karpenter 컨트롤러 로그 (가장 중요)
kubectl logs -n kube-system -l app.kubernetes.io/name=karpenter -f
```

#### 로그에서 봐야 할 것 ####

Karpenter 컨트롤러 로그에서:
- found provisionable pod(s) → Pod를 인지
- 어떤 NodePool을 선택했는지 (nodepool 필드)
- weight 높은 NodePool이 왜 스킵됐는지 (예: no instance type satisfied requirements, insufficient capacity, limit 도달 등)

