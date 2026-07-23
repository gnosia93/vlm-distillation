#!/usr/bin/env bash
set -euo pipefail

BUCKET="${BUCKET:?BUCKET env var must be set}"
VIDEO_ID="${1:?video id required}"
PREFIX="finevideo/sports/${VIDEO_ID}"
N_FRAMES=16
WORK=$(mktemp -d)
trap 'rm -rf "${WORK}"' EXIT   # 중간에 실패해도 임시 폴더 자동 정리

# 1) 원본 영상 다운로드
aws s3 cp "s3://${BUCKET}/${PREFIX}/video.mp4" "${WORK}/video.mp4"

# 2) 균일 샘플링
DURATION=$(ffprobe -v error -show_entries format=duration -of csv=p=0 "${WORK}/video.mp4")
FPS=$(echo "scale=6; ${N_FRAMES} / ${DURATION}" | bc)
mkdir -p "${WORK}/frames"
ffmpeg -y -i "${WORK}/video.mp4" \
  -vf "fps=${FPS},scale=448:448:force_original_aspect_ratio=decrease,pad=448:448:(ow-iw)/2:(oh-ih)/2" \
  -frames:v ${N_FRAMES} -q:v 2 \
  "${WORK}/frames/frame_%03d.jpg"

# 2.5) frames.json 생성 (인퍼런스 입력 명세)
DURATION_INT=$(printf "%.0f" "${DURATION}")   # 268.44 -> 268 (정수)

# 프레임들을 S3 키(전체 경로)로 나열
FRAME_KEYS=$(cd "${WORK}/frames" && ls frame_*.jpg | sort \
  | sed "s#^#${PREFIX}/frames/#")

# 샘플링 설정 해시 (설정이 같으면 같은 해시)
CONFIG_HASH=$(printf "%s|%s|%s" "${N_FRAMES}" "uniform" "448x448" \
  | sha256sum | cut -c1-6)

jq -n \
  --arg video_id "${VIDEO_ID}" \
  --argjson num_frames "${N_FRAMES}" \
  --arg sampling "uniform" \
  --arg frame_size "448x448" \
  --argjson source_duration "${DURATION_INT}" \
  --arg frames "${FRAME_KEYS}" \
  --arg hash "${CONFIG_HASH}" \
  '{
    video_id: $video_id,
    num_frames: $num_frames,
    sampling: $sampling,
    frame_size: $frame_size,
    source_duration: $source_duration,
    frames: ($frames | split("\n")),
    sampling_config_hash: $hash
  }' > "${WORK}/frames/frames.json"

# 3) 결과를 S3에 업로드 (frames.json 포함)
aws s3 cp "${WORK}/frames/" "s3://${BUCKET}/${PREFIX}/frames/" --recursive

# 4) 정리 (trap이 처리하므로 생략 가능)
