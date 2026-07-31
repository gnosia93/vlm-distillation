# event_clips.py
# CCTV 영상 → 배경차분 모션 감지 → 이벤트 구간[start,end] → 구간당 N장 추출
# 모션 감지는 CPU 연산이라 GPU 불필요 (Graviton 등 CPU 노드에 적합)

import os
import cv2
import numpy as np

# ============ 설정 ============
VIDEO_PATH      = "/path/to/cctv.mp4"     # ← 입력 영상
OUT_DIR         = "/path/to/event_clips"  # ← 출력 폴더
MOTION_FPS      = 3        # 모션 분석 샘플 레이트 (초당 몇 장 볼지)
RESIZE_W        = 480      # 모션 분석용 축소 폭 (속도↑, 정확도 영향 적음)

START_THRESH    = 0.020    # 전경 비율 > 이 값 → 이벤트 시작 (히스테리시스 높은 쪽)
END_THRESH      = 0.008    # 전경 비율 < 이 값 → 종료 후보 (낮은 쪽)
COOLDOWN_SEC    = 2.0      # 조용함이 이 시간 지속되면 이벤트 종료
PRE_ROLL_SEC    = 1.0      # 시작 앞에 포함할 여유
POST_ROLL_SEC   = 1.0      # 종료 뒤에 포함할 여유
MIN_EVENT_SEC   = 1.0      # 이보다 짧은 이벤트는 버림
MERGE_GAP_SEC   = 1.5      # 이보다 가까운 이벤트는 하나로 병합

FRAMES_PER_CLIP = 16       # 이벤트 구간당 뽑을 프레임 수 (VLM용)
# =============================

def detect_events(video_path):
    """1단계: 모션 감지로 이벤트 구간(초 단위) 리스트 반환"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"영상 열기 실패: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total / fps
    stride = max(1, round(fps / MOTION_FPS))   # 모션 분석 간격(프레임)

    bg = cv2.createBackgroundSubtractorMOG2(history=200, varThreshold=25,
                                            detectShadows=False)

    events = []
    state = "IDLE"
    start_t = 0.0
    quiet_t = 0.0
    prev_t = 0.0
    idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if idx % stride != 0:      # coarse 샘플링 (매 stride번째만 분석)
            idx += 1
            continue

        t = idx / fps
        small = cv2.resize(frame, (RESIZE_W, int(RESIZE_W * frame.shape[0] / frame.shape[1])))
        fg = bg.apply(small)
        score = float((fg > 0).sum()) / fg.size    # 전경 픽셀 비율 = 모션 점수

        if state == "IDLE":
            if score > START_THRESH:
                start_t = max(0.0, t - PRE_ROLL_SEC)
                state = "IN_EVENT"
                quiet_t = 0.0
        else:  # IN_EVENT
            if score < END_THRESH:
                quiet_t += (t - prev_t)
                if quiet_t >= COOLDOWN_SEC:
                    end_t = min(duration, t + POST_ROLL_SEC)
                    if end_t - start_t >= MIN_EVENT_SEC:
                        events.append([start_t, end_t])
                    state = "IDLE"
            else:
                quiet_t = 0.0

        prev_t = t
        idx += 1

    # 영상 끝에서 이벤트 진행 중이면 마감
    if state == "IN_EVENT":
        end_t = duration
        if end_t - start_t >= MIN_EVENT_SEC:
            events.append([start_t, end_t])

    cap.release()
    return merge_events(events), fps, duration

def merge_events(events):
    """가까운 이벤트 병합"""
    if not events:
        return []
    merged = [events[0]]
    for s, e in events[1:]:
        if s - merged[-1][1] <= MERGE_GAP_SEC:
            merged[-1][1] = e            # 병합
        else:
            merged.append([s, e])
    return merged

def extract_clip(video_path, start_t, end_t, n_frames, out_dir):
    """2단계: 이벤트 구간에서 N장 균일 추출 → 저장"""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # 구간 내 균일 타임스탬프 → 프레임 인덱스
    times = np.linspace(start_t, end_t, n_frames)
    os.makedirs(out_dir, exist_ok=True)
    saved = 0
    for i, t in enumerate(times):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(t * fps))
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imwrite(os.path.join(out_dir, f"frame_{i:02d}.jpg"), frame)
        saved += 1
    cap.release()
    return saved

def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    events, fps, duration = detect_events(VIDEO_PATH)

    print(f"영상 길이: {duration:.1f}s, fps: {fps:.1f}")
    print(f"감지된 이벤트: {len(events)}개")
    if not events:
        print("이벤트 없음 (정적 영상이거나 임계값 조정 필요)")
        return

    for i, (s, e) in enumerate(events):
        clip_dir = os.path.join(OUT_DIR, f"event_{i:03d}")
        n = extract_clip(VIDEO_PATH, s, e, FRAMES_PER_CLIP, clip_dir)
        print(f"  event_{i:03d}: {s:6.1f}s ~ {e:6.1f}s ({e-s:.1f}s) → {n}장 저장 → {clip_dir}")

    print(f"\n완료. 총 {len(events)}개 클립, 각 최대 {FRAMES_PER_CLIP}장.")
    print("→ 각 event 폴더의 프레임을 VLM(16장 클립) 또는 SAM-3(시퀀스)에 투입")

if __name__ == "__main__":
    main()
