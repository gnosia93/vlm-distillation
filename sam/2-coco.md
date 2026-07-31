
CVAT는 무료 + SAM 보조 라벨링 + COCO export를 다 갖춰서 프레임 라벨링에 적합하다.



### 1. 설치 (자체 호스팅, SAM 포함) ###
```
git clone https://github.com/cvat-ai/cvat.git
cd cvat
docker compose up -d          # 기본 CVAT 기동 → http://localhost:8080
```
```
# SAM 등 AI 자동주석 함수는 Nuclio(serverless)로 별도 배포
# (CVAT 문서의 "Automatic annotation / serverless" 절차 따라 SAM 함수 deploy)
SAM 함수 배포 절차·이름은 CVAT 버전마다 달라지니, 현재 CVAT 공식 문서의 AI Tools/serverless 섹션을 확인 필요.
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
