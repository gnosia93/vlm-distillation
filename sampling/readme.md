
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
