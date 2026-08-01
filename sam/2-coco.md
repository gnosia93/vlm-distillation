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

### 2. UI에서 SAM으로 라벨링 ### 

* Task 생성 → 영상 프레임 업로드, 라벨 child(mask/polygon)
* 프레임 열기 → 좌측 AI Tools → Interactors → Segment Anything 선택
* 오브젝트 클릭 → 마스크 초안 → positive/negative 점으로 보정 → 오브젝트 확정
* 검수(놓친 것 추가) → Export dataset → COCO 1.0


#### (영상 추적 라벨링이면) SAM2 Tracker ####
만약 영상 추적용 masklet 라벨(2단계 영상 FT용)까지 만들 거면, CVAT의 Segment Anything 2 Tracker 함수를 사용한다. 단, 이건 추적 상태 저장용 Redis가 별도로 필요하다. (cvat_redis_ondisk 재사용 가능). 이걸 쓰면 한 프레임 라벨 → 여러 프레임 자동 추적으로 masklet을 빠르게 만들 수 있다.
```

> [!NOTE]
> 영상 직접 업로드 + 추적 보간: CVAT는 영상에서 한 프레임 라벨 후 다음 프레임으로 보간/추적하는 기능도 있어, 연속 프레임 라벨링을 더 줄일 수 있다. (트래킹 데이터에 유용).
> train/valid 분리는 라벨링 전에 정해두면 편함 (같은 영상 프레임이 train·valid에 섞이면 누수 위험 → 영상 단위로 분리 권장).


## COCO 데이터셋 준비 ##

본 워크샵에서는 이미 라벨링 된 COCO 데이터셋을 활용한다. COCO 데이터셋을 다운로드 받기 위해, 파이썬 가상환경을 생성하고 관련 패키지를 설치한다. 
```
python3 -m venv .venv
source .venv/bin/activate
pip install fiftyone pycocotools
```
데이터 셋을 다운로드 받기 위해서 스크립트를 실행한다. 데이터셋은 ~/fiftyone/coco-2017 디렉토리에 저장된다.  
```
cat <<'EOF' > coco.py
import fiftyone as fo
import fiftyone.zoo as foz

# 1. 기존 데이터셋이 꼬였을 수 있으니 새 이름으로 로드
dataset = foz.load_zoo_dataset(
    "coco-2017",
    split="validation",
    label_types=["segmentations"],
    max_samples=50,
    dataset_name="coco-working-contours"
)

# 2. Polylines 변환
for sample in dataset:
    if sample.ground_truth:
        # filled=False로 테두리선만 생성
        sample["polygons"] = sample.ground_truth.to_polylines(
            tolerance=0,
            filled=False
        )

        # ground_truth의 인스턴스 마스크 제거 → 박스 안 색칠 없이 바운딩 박스만 표시
        for detection in sample.ground_truth.detections:
            detection.mask = None
            detection.mask_path = None

        sample.save()

# 3. 앱 실행 (설정값 강제 지정 없이 기본 렌더링으로 띄움)
session = fo.launch_app(dataset, port=5151)
session.wait()
EOF

python coco.py
```
[결과]
```
Downloading split 'validation' to '/Users/automake/fiftyone/coco-2017/validation' if necessary
Downloading annotations to '/Users/automake/fiftyone/coco-2017/tmp-download/annotations_trainval2017.zip'
  64% |█████████████████████████████████████████████████████████████████████████████/-------------------------------------------|    1.2Gb/1.9Gb [14.3s elapsed, 6.2s remaining, 132.4Mb/s]
```

#### 웹 브라이저 UI ####
![](https://github.com/gnosia93/vlm-distillation/blob/main/images/coco-webui.png)
