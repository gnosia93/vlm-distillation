"""MNIST raw 파일을 S3에 업로드하는 준비 스크립트.

훈련 Job(PyTorchJob)을 실행하기 전에 한 번만 로컬에서 돌리면 된다.
torchvision으로 MNIST를 내려받아, 훈련 코드가 기대하는 raw .gz 4개를
s3://<bucket>/<prefix>/ 로 올린다.

자격증명/엔드포인트는 표준 AWS_* 환경변수를 사용(실제 S3, MinIO 모두 동작).

예)
    python upload_mnist_to_s3.py --s3-bucket my-datasets --s3-prefix mnist/raw
"""

import argparse
import os

import boto3
from torchvision import datasets

# train.py 와 동일한 목록 — torchvision MNIST 가 기대하는 raw 파일.
MNIST_RAW_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--s3-bucket",
        default=os.environ.get("S3_BUCKET"),
        required=os.environ.get("S3_BUCKET") is None,
        help="업로드 대상 버킷",
    )
    parser.add_argument(
        "--s3-prefix",
        default=os.environ.get("S3_PREFIX", "mnist/raw"),
        help="raw .gz 파일을 올릴 prefix (train.py 의 S3_PREFIX 와 일치해야 함)",
    )
    parser.add_argument(
        "--data-dir",
        default="./data",
        help="MNIST 를 내려받을 로컬 임시 디렉터리",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="이미 S3 에 있어도 다시 업로드",
    )
    args = parser.parse_args()

    # 1) 로컬로 MNIST raw 다운로드 (이미 있으면 재사용).
    print(f"MNIST 다운로드 -> {args.data_dir}", flush=True)
    datasets.MNIST(args.data_dir, train=True, download=True)
    datasets.MNIST(args.data_dir, train=False, download=True)
    raw_dir = os.path.join(args.data_dir, "MNIST", "raw")

    # 2) S3 로 업로드.
    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
    prefix = args.s3_prefix.rstrip("/")

    for fname in MNIST_RAW_FILES:
        src = os.path.join(raw_dir, fname)
        if not os.path.exists(src):
            raise FileNotFoundError(f"raw 파일이 없음: {src}")

        key = f"{prefix}/{fname}"

        if not args.overwrite and _exists(s3, args.s3_bucket, key):
            print(f"건너뜀(이미 존재): s3://{args.s3_bucket}/{key}", flush=True)
            continue

        print(f"업로드: {src} -> s3://{args.s3_bucket}/{key}", flush=True)
        s3.upload_file(src, args.s3_bucket, key)

    print("완료.", flush=True)


def _exists(s3, bucket, key):
    """객체 존재 여부. 없으면 404 로 False."""
    try:
        s3.head_object(Bucket=bucket, Key=key)
        return True
    except s3.exceptions.ClientError:
        return False


if __name__ == "__main__":
    main()
