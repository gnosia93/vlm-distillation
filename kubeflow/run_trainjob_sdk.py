"""Kubeflow Trainer(V2) Python SDK 로 MNIST DDP TrainJob 실행.

V2 의 1급(primary) 인터페이스는 이 SDK 다. YAML 없이 여기서 num_nodes /
num_proc_per_node 만 바꿔 멀티노드 <-> 단일노드 multi-GPU 를 전환한다.

⚠️ SDK 필드명은 설치된 kubeflow-trainer 버전에 따라 다를 수 있다.
   실행 전 `pip show kubeflow` 로 버전 확인, help(CustomTrainer) 로 시그니처 확인.

    pip install kubeflow
    python run_trainjob_sdk.py --num-nodes 2 --num-proc-per-node 1
"""

import argparse

from kubeflow.trainer import CustomTrainer, TrainerClient


def main():
    parser = argparse.ArgumentParser()
    # 스케일: 이 두 숫자만 바꾸면 멀티노드 <-> 단일노드 multi-GPU 전환.
    parser.add_argument("--num-nodes", type=int, default=2)
    parser.add_argument("--num-proc-per-node", type=int, default=1)
    parser.add_argument("--gpus-per-node", type=int, default=1)
    parser.add_argument("--image", default="YOUR_REGISTRY/mnist-ddp:latest")
    parser.add_argument("--runtime", default="torch-distributed")
    parser.add_argument("--s3-bucket", default="my-datasets")
    parser.add_argument("--s3-prefix", default="mnist/raw")
    args = parser.parse_args()

    trainer = CustomTrainer(
        # 학습 진입점. 이미지 안의 train.py 를 그대로 호출.
        command=[
            "python",
            "train.py",
            "--epochs=5",
            "--batch-size=64",
        ],
        num_nodes=args.num_nodes,
        num_proc_per_node=str(args.num_proc_per_node),
        resources_per_node={
            "cpu": "4",
            "memory": "8Gi",
            "nvidia.com/gpu": args.gpus_per_node,
        },
        env={
            "S3_BUCKET": args.s3_bucket,
            "S3_PREFIX": args.s3_prefix,
            # AWS 자격증명은 런타임/서비스어카운트(IRSA) 로 주입하는 것을 권장.
        },
        packages_to_install=None,  # 이미지에 이미 포함되어 있으므로 없음.
    )

    client = TrainerClient()
    job_name = client.train(runtime_ref=args.runtime, trainer=trainer)
    print(f"TrainJob 생성됨: {job_name}", flush=True)
    print(f"상태 확인:  kubectl get trainjob {job_name}", flush=True)


if __name__ == "__main__":
    main()
