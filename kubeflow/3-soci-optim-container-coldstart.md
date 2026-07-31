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

### 제대로 붙었는지 확인하기 ###
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

### 마치며 ###
큰 이미지를 다루는 ML 워크로드에서 콜드 스타트의 진짜 병목은 노드 프로비저닝이 아니라 이미지 pull입니다. SOCI는 "이미지를 다 받고 시작"하던 방식을 "필요한 것만 그때그때 seek"하는 방식으로 바꿔서 이 병목을 줄여줍니다.

#### 핵심을 세 줄로 요약하면: ####

* SOCI는 AMI 기능이 아니라 노드의 스냅샷터 데몬 + containerd 설정이다. AMI를 바꾼다고 켜지지 않는다.
* Karpenter에서는 EC2NodeClass를 교체하는 게 아니라 spec.userData를 추가해서 붙인다.
* 이미지에 SOCI 인덱스가 있고(-soci 태그) 실행 환경이 스냅샷터를 지원할 때만 실제로 가속된다.

오토스케일로 GPU 노드가 수시로 떴다 지는 환경이라면, 이 한 번의 세팅이 매 스케일아웃마다 몇 분씩을 돌려줍니다. 특히 대형 VLM 이미지를 여러 노드에 뿌리는 우리 파이프라인에서는 그 효과가 곧바로 체감될 겁니다.

## 레퍼런스 ##

* https://aditmodi.hashnode.dev/soci-snapshotter-on-eks-eliminating-the-8-minute-gpu-cold-start-nobody-talks-about
* Seekable OCI: Lazy-Loading Container Images via Range-Request Indexing - https://arxiv.org/abs/2607.06868





   
 



