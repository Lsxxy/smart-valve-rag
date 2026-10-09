"""下载公开基础模型；微调意图模型需另外训练，模型权重不纳入 Git。"""

from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_REPOS = {
    "bge-m3": "BAAI/bge-m3",
    "bge-reranker-large": "BAAI/bge-reranker-large",
    "bert-base-chinese": "google-bert/bert-base-chinese",
}


def main() -> None:
    parser = argparse.ArgumentParser(description="准备智能阀知识助手的公开基础模型。")
    parser.add_argument("--model", action="append", choices=tuple(MODEL_REPOS),
                        help="指定模型，可重复；默认下载全部三个基础模型。")
    parser.add_argument("--model-root", type=Path, default=PROJECT_ROOT / "models")
    parser.add_argument("--revision", default="main", help="模型版本；单模型下载时可指定提交 SHA。")
    parser.add_argument("--list", action="store_true", help="仅列出来源与路径，不联网。")
    args = parser.parse_args()
    selected = args.model or list(MODEL_REPOS)
    if args.list:
        for name in selected:
            print(f"{MODEL_REPOS[name]} -> {args.model_root / name}")
        return

    from huggingface_hub import snapshot_download

    for name in selected:
        print(f"Downloading {MODEL_REPOS[name]} to {args.model_root / name}", flush=True)
        snapshot_download(repo_id=MODEL_REPOS[name], revision=args.revision,
                          local_dir=str(args.model_root / name),
                          ignore_patterns=["onnx/*", "openvino/*", "*.h5", "*.msgpack", "*.ot"])


if __name__ == "__main__":
    main()
