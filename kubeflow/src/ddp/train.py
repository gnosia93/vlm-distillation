"""MNIST DDP training for Kubeflow (멀티노드 / 단일노드 multi-GPU 공용).

Kubeflow Trainer(또는 V1 PyTorchJob) 가 표준 torch.distributed 환경변수
(MASTER_ADDR, MASTER_PORT, WORLD_SIZE, RANK, LOCAL_RANK) 를 각 프로세스에
주입하므로, dist.init_process_group() 만 호출하면 된다. 백엔드는 GPU 면 nccl,
CPU 면 gloo 로 자동 선택.

S3 다운로드를 LOCAL_RANK 0 기준으로 처리해, numNodes/numProcPerNode 숫자만
바꾸면 코드 수정 없이 두 스케일 구성 모두에서 동작한다.
"""

import argparse
import os
import time

import boto3
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from torchvision.datasets.utils import extract_archive

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = self.dropout2(x)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# torchvision MNIST expects the raw files under <root>/MNIST/raw/.
MNIST_RAW_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def is_distributed():
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def download_from_s3(bucket, prefix, data_dir):
    """Fetch the MNIST raw files from S3 into <data_dir>/MNIST/raw/.

    Credentials/endpoint come from the standard AWS_* env vars, so this works
    with real S3 or an S3-compatible store (MinIO) via AWS_ENDPOINT_URL.
    """
    raw_dir = os.path.join(data_dir, "MNIST", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
    for fname in MNIST_RAW_FILES:
        key = f"{prefix.rstrip('/')}/{fname}"
        dest = os.path.join(raw_dir, fname)
        print(f"S3 다운로드: s3://{bucket}/{key} -> {dest}", flush=True)
        s3.download_file(bucket, key, dest)
        extract_archive(dest, raw_dir)          # ★ .gz 압축 해제
        print(f"압축 해제 완료: {dest}", flush=True)


def setup(backend):
    if is_distributed():
        dist.init_process_group(backend=backend)


def cleanup():
    if is_distributed() and dist.is_initialized():
        dist.destroy_process_group()


def train_epoch(model, device, loader, optimizer, epoch, rank):
    model.train()
    for batch_idx, (data, target) in enumerate(loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 50 == 0:
            print(
                f"[rank {rank}] epoch {epoch} "
                f"[{batch_idx * len(data)}/{len(loader.dataset)}] "
                f"loss={loss.item():.4f}",
                flush=True,
            )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1.0)
    parser.add_argument("--gamma", type=float, default=0.7)
    parser.add_argument("--data-dir", default="/data")
    parser.add_argument("--save-path", default="/data/mnist_ddp.pt")
    parser.add_argument(
        "--s3-bucket", default=os.environ.get("S3_BUCKET"), help="MNIST 데이터 버킷"
    )
    parser.add_argument(
        "--s3-prefix",
        default=os.environ.get("S3_PREFIX", "mnist/raw"),
        help="raw .gz 파일들이 있는 prefix",
    )
    args = parser.parse_args()

    use_cuda = torch.cuda.is_available()
    backend = "nccl" if use_cuda else "gloo"

    setup(backend)

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if use_cuda:
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")

    print(
        f"start: rank={rank}/{world_size} local_rank={local_rank} "
        f"backend={backend} device={device}",
        flush=True,
    )

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
    )

    # 각 파드(노드)의 대표 프로세스(local_rank 0)만 S3 에서 받고, 같은 파드의
    # 나머지 프로세스는 barrier 로 대기한다. LOCAL_RANK 기준이라 아래 두 구성에서
    # 코드 변경 없이 그대로 동작한다:
    #   - 멀티노드      : 파드마다 local_rank 0 가 하나씩 -> 파드별로 각자 받음
    #   - 단일노드 multi-GPU: 파드 1개에 local_rank 0 하나 -> 한 번만 받음
    # download=False -> torchvision 은 절대 인터넷을 타지 않는다.
    if not args.s3_bucket:
        raise ValueError("--s3-bucket (또는 S3_BUCKET 환경변수) 가 필요합니다")
    if local_rank == 0:
        download_from_s3(args.s3_bucket, args.s3_prefix, args.data_dir)
    if is_distributed():
        dist.barrier()

    train_ds = datasets.MNIST(
        args.data_dir, train=True, download=False, transform=transform
    )

    sampler = (
        DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        if is_distributed()
        else None
    )
    loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        shuffle=(sampler is None),
        num_workers=2,
        pin_memory=use_cuda,
    )

    model = Net().to(device)
    if is_distributed():
        model = DDP(
            model,
            device_ids=[local_rank] if use_cuda else None,
        )

    optimizer = torch.optim.Adadelta(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=args.gamma)

    # 학습 구간 실행 시간 측정. GPU 는 커널이 비동기 실행되므로 측정 전후로
    # synchronize() 해서 실제 완료 시점을 기준으로 잰다.
    if use_cuda:
        torch.cuda.synchronize()
    train_start = time.perf_counter()

    for epoch in range(1, args.epochs + 1):
        if sampler is not None:
            sampler.set_epoch(epoch)
        epoch_start = time.perf_counter()
        train_epoch(model, device, loader, optimizer, epoch, rank)
        scheduler.step()
        if use_cuda:
            torch.cuda.synchronize()
        print(
            f"[rank {rank}] epoch {epoch} 소요={time.perf_counter() - epoch_start:.2f}s",
            flush=True,
        )

    if use_cuda:
        torch.cuda.synchronize()
    total_train_time = time.perf_counter() - train_start
    print(
        f"[rank {rank}] 총 학습 시간={total_train_time:.2f}s "
        f"(epochs={args.epochs}, world_size={world_size})",
        flush=True,
    )

    if rank == 0:
        state = model.module.state_dict() if is_distributed() else model.state_dict()
        torch.save(state, args.save_path)
        print(f"모델 저장 완료: {args.save_path}", flush=True)

        # Upload the checkpoint back to S3 under <prefix>/../checkpoints/.
        s3 = boto3.client("s3", endpoint_url=os.environ.get("AWS_ENDPOINT_URL"))
        ckpt_key = f"checkpoints/{os.path.basename(args.save_path)}"
        s3.upload_file(args.save_path, args.s3_bucket, ckpt_key)
        print(f"체크포인트 업로드: s3://{args.s3_bucket}/{ckpt_key}", flush=True)

    cleanup()


if __name__ == "__main__":
    main()
