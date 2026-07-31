
영상 → [배경차분 MOG2, 3fps로 훑기] → 모션 점수
      → [히스테리시스 + 쿨다운 + 프리/포스트롤] → 이벤트 구간 [start,end]
      → [병합/최소길이 필터]
      → 각 구간에서 16장 균일 추출 → event_000/, event_001/ ...
      → VLM(클립) 또는 SAM-3(시퀀스) 입력

```
python -m pip install opencv-python numpy
# 스크립트 상단 VIDEO_PATH, OUT_DIR 수정 후
python event_clips.py
```
[결과]
```
영상 길이: 3600.0s, fps: 30.0
감지된 이벤트: 8개
  event_000:   42.0s ~   58.0s (16.0s) → 16장 저장 → .../event_000
  event_001:  310.0s ~  322.0s (12.0s) → 16장 저장 → .../event_001
  ...
완료. 총 8개 클립, 각 최대 16장.
```

#### 튜닝 포인트 (현장 맞춤) ####
```
파라미터	조정 방향
START_THRESH / END_THRESH	오탐 많으면 ↑, 이벤트 놓치면 ↓
COOLDOWN_SEC	한 사건이 자꾸 쪼개지면 ↑
MERGE_GAP_SEC	가까운 이벤트 합치고 싶으면 ↑
MOTION_FPS	순간 사건 놓치면 ↑ (비용도 ↑)
FRAMES_PER_CLIP	VLM 입력 규격에 맞춤(16)
```
