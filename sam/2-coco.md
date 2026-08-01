## CVAT 라벨링 ##

CVAT는 무료 + SAM 보조 라벨링 + COCO export를 다 갖춰서 프레임 라벨링에 적합하다.

### 1. 설치 (자체 호스팅, SAM 포함) ###

* cvat - Computer Vision Annotation Tool. 컴퓨터 비전용 데이터 라벨링(주석) 웹 도구로 이미지·영상에 정답(박스·마스크 등)을 사람이 표시해서 학습용 데이터셋을 만드는 데 사용된다.
```
git clone https://github.com/cvat-ai/cvat
cd cvat
docker compose up -d

관리자 계정 생성
docker exec -it cvat_server bash -ic 'python3 ~/manage.py createsuperuser'
→ 브라우저에서 http://<서버IP>:8080 접속.
```

* nuctl - CVAT에서는 SAM, YOLO, Mask R-CNN 같은 DL 모델들을 각각 "Nuclio 서버리스 함수"로 패키징해서 돌린다. 즉 SAM으로 클릭하면 마스크 생성 같은 자동 주석 기능이 내부적으로 Nuclio 함수 호출로 동작한다. nuctl은 그걸 관리하는 CLI 도구이다.
```
wget https://github.com/nuclio/nuclio/releases/download/<VERSION>/nuctl-<VERSION>-linux-amd64
chmod +x nuctl-<VERSION>-linux-amd64
sudo ln -sf $(pwd)/nuctl-<VERSION>-linux-amd64 /usr/local/bin/nuctl

docker compose -f docker-compose.yml \
  -f components/serverless/docker-compose.serverless.yml up -d

./serverless/deploy_gpu.sh serverless/pytorch/facebookresearch/sam/nuclio

nuctl get functions
pth-facebookresearch-sam-... 함수가 ready 상태면 OK.
```


```
5. UI에서 SAM으로 라벨링
Task 생성 → 유치원 프레임 업로드, 라벨 child(mask/polygon)
프레임 열기 → 좌측 AI Tools → Interactors → Segment Anything 선택
아이 클릭 → 마스크 초안 → positive/negative 점으로 보정 → child 확정
검수(놓친 것 추가) → Export dataset → COCO 1.0


6. (영상 추적 라벨링이면) SAM2 Tracker
만약 영상 추적용 masklet 라벨(2단계 영상 FT용)까지 만들 거면, CVAT의 Segment Anything 2 Tracker 함수를 씁니다. 단, 이건 추적 상태 저장용 Redis가 추가로 필요해요 (cvat_redis_ondisk 재사용 가능). deploy 시 인자로 지정합니다. 이걸 쓰면 한 프레임 라벨 → 여러 프레임 자동 추적으로 masklet을 빠르게 만들 수 있어요.
```

### 2. 프로젝트·작업 생성 + 라벨 정의 ###
* Project 생성 → 라벨 추가: 이름 child, 타입 Mask 또는 Polygon (세그멘테이션이니)
* Task 생성 → 유치원 프레임(event_clips에서 뽑은 것) 업로드
* 이미지 묶음으로 올리거나, 영상을 통째로 올려 프레임 자동 분할도 가능

### 3. SAM으로 반자동 라벨링 (핵심 가속) ###
* 프레임 열기 → 왼쪽 도구에서 AI Tools / Interactor 선택 → SAM(Segment Anything) 지정
* 아이 몸을 클릭 → SAM이 마스크 초안 생성
* 필요하면 positive/negative 점 추가로 다듬기
* 라벨 child 지정 → 확정
* 프레임의 모든 아이에 반복
이렇게 하면 폴리곤을 손으로 찍는 것보다 훨씬 빠르다.

### 4. 검수 (중요) ###
* SAM이 놓친 아이를 사람이 추가 (파인튜닝의 진짜 목표는 이 놓친 케이스들)
* 마스크 경계 보정
* child 없는 프레임도 일부 그대로 둠 (hard negative용)

### 5. COCO로 Export ###
* Task/Project → Export dataset → 포맷 COCO 1.0 선택 (세그멘테이션 포함됨)
* 다운로드하면 annotations/instances_*.json 형태로 나온다.
  
### 6. SAM3_LoRA 레이아웃으로 재구성 ###
CVAT export 구조와 SAM3_LoRA 기대 구조가 조금 다르니 정리해야 한다.

```
data/
├── train/
│   ├── (이미지들)
│   └── _annotations.coco.json    ← CVAT의 instances_*.json 을 이 이름으로
└── valid/
    ├── (이미지들)
    └── _annotations.coco.json
```
* CVAT가 준 instances_default.json → _annotations.coco.json으로 이름 변경
* 이미지와 JSON을 train/valid로 분리 (8:2). CVAT에서 task를 train/valid 두 개로 나눠 export하거나, 한 번에 받고 스크립트로 split
* categories의 이름이 child인지 확인 (프롬프트로 쓰임). 


> [!NOTE]
> 영상 직접 업로드 + 추적 보간: CVAT는 영상에서 한 프레임 라벨 후 다음 프레임으로 보간/추적하는 기능도 있어, 연속 프레임 라벨링을 더 줄일 수 있다. (트래킹 데이터에 유용).
> train/valid 분리는 라벨링 전에 정해두면 편함 (같은 영상 프레임이 train·valid에 섞이면 누수 위험 → 영상 단위로 분리 권장).

## Roboflow Universe ##

* https://app.roboflow.com/login 에서 회원 가입 -> Public Plan 선택 -> Create Workspace 
* Settings → API Key 복사 (rf_xxxx)

### Python SDK 로 데이터 다운로드 받기 ###
```
pip install roboflow
```
```
from roboflow import Roboflow
rf = Roboflow(api_key="rf_xxxx")           # 본인 API 키
project = rf.workspace("워크스페이스명").project("프로젝트명")
dataset = project.version(1).download("coco-segmentation")   # 또는 "coco"
print(dataset.location)   # 다운로드된 경로
```
* 워크스페이스/프로젝트/버전 이름은 데이터셋마다 다르니, 각 데이터셋 페이지의 코드 스니펫을 복사해서 쓰는 게 정확하다.
