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
