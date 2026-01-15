#!/usr/bin/env python
"""
로컬 데이터셋을 Hugging Face Hub로 push하는 스크립트

사용법:
    python push_dataset.py --repo_id "사용자명/데이터셋이름" --root "./datasets"
    python push_dataset.py --repo_id "xhaka3456/my_dataset" --root "./datasets" --private
"""

import argparse
from pathlib import Path

from lerobot.datasets.lerobot_dataset import LeRobotDataset


def main():
    parser = argparse.ArgumentParser(description="Push local dataset to Hugging Face Hub")
    parser.add_argument("--repo_id", type=str, required=True, help="Hub repo ID (예: username/dataset_name)")
    parser.add_argument("--root", type=str, required=True, help="로컬 데이터셋 경로")
    parser.add_argument("--private", action="store_true", help="비공개 저장소로 업로드")
    parser.add_argument("--tags", type=str, nargs="*", default=None, help="태그 목록")
    parser.add_argument("--large", action="store_true", help="대용량 폴더 업로드 모드")
    args = parser.parse_args()

    root = Path(args.root)
    print(f"Loading dataset: {args.repo_id} from {root}")

    dataset = LeRobotDataset(args.repo_id, root=root)

    print(f"Pushing to hub (private={args.private}, large={args.large})...")
    dataset.push_to_hub(tags=args.tags, private=args.private, upload_large_folder=args.large)

    print("Done!")


if __name__ == "__main__":
    main()
