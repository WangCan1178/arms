#!/usr/bin/env python3
"""Download the four open-source VLM checkpoints into one Hugging Face cache."""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from typing import Iterable, Optional


DEFAULT_REPOS = [
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "HuggingFaceM4/idefics2-8b",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
]


def human_bytes(size: int) -> str:
    units = ["B", "KiB", "MiB", "GiB", "TiB"]
    value = float(size)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{size} B"


def directory_size(path: Path) -> int:
    total = 0
    if not path.exists():
        return total
    for item in path.rglob("*"):
        if item.is_symlink():
            try:
                total += item.lstat().st_size
            except OSError:
                pass
        elif item.is_file():
            try:
                total += item.stat().st_size
            except OSError:
                pass
    return total


def repo_cache_dir(cache_dir: Path, repo_id: str) -> Path:
    return cache_dir / "hub" / f"models--{repo_id.replace('/', '--')}"


def repo_stats(cache_dir: Path, repo_id: str) -> str:
    repo_dir = repo_cache_dir(cache_dir, repo_id)
    explicit_cache_dir = cache_dir / f"models--{repo_id.replace('/', '--')}"
    if not repo_dir.exists():
        extra = ""
        if explicit_cache_dir.exists():
            extra = (
                f" explicit_cache_dir_exists=True "
                f"explicit_size={human_bytes(directory_size(explicit_cache_dir))}"
            )
        return f"cache_dir_exists=False{extra}"
    files = [p for p in repo_dir.rglob("*") if p.is_file()]
    incomplete = [p for p in files if p.name.endswith(".incomplete")]
    snapshots = repo_dir / "snapshots"
    snapshot_count = len([p for p in snapshots.iterdir() if p.is_dir()]) if snapshots.exists() else 0
    extra = ""
    if explicit_cache_dir.exists():
        explicit_incomplete = len([p for p in explicit_cache_dir.rglob("*.incomplete") if p.is_file()])
        extra = (
            f" explicit_cache_dir_exists=True explicit_incomplete={explicit_incomplete} "
            f"explicit_size={human_bytes(directory_size(explicit_cache_dir))}"
        )
    return (
        f"cache_dir_exists=True files={len(files)} snapshots={snapshot_count} "
        f"incomplete={len(incomplete)} size={human_bytes(directory_size(repo_dir))}{extra}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(os.environ.get("HF_HOME", "hf_cache")).expanduser(),
        help="Hugging Face cache directory. Defaults to HF_HOME or ./hf_cache.",
    )
    parser.add_argument(
        "--repo",
        action="append",
        default=None,
        help="Repo id to download. May be repeated. Defaults to the four M2 Free VLMs.",
    )
    parser.add_argument(
        "--endpoint",
        default=os.environ.get("HF_ENDPOINT"),
        help="Optional Hugging Face endpoint, for example https://hf-mirror.com.",
    )
    parser.add_argument("--max-workers", type=int, default=4)
    parser.add_argument(
        "--attempt-gated-without-token",
        action="store_true",
        help="Attempt gated repos even when HF_TOKEN/HUGGING_FACE_HUB_TOKEN is absent.",
    )
    parser.add_argument(
        "--allow-gated-skip",
        action="store_true",
        help="Return success even if a gated default model is skipped.",
    )
    return parser.parse_args()


def iter_repos(args: argparse.Namespace) -> Iterable[str]:
    return args.repo if args.repo else DEFAULT_REPOS


def download_repo(repo_id: str, token: Optional[str], max_workers: int) -> tuple[str, str]:
    from huggingface_hub import snapshot_download

    start = time.time()
    path = snapshot_download(
        repo_id=repo_id,
        token=token,
        max_workers=max_workers,
    )
    elapsed = time.time() - start
    return path, f"{elapsed:.1f}s"


def main() -> int:
    args = parse_args()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cache_dir)
    if args.endpoint:
        os.environ["HF_ENDPOINT"] = args.endpoint

    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    print(f"HF_HOME={os.environ.get('HF_HOME')}", flush=True)
    print(f"HF_ENDPOINT={os.environ.get('HF_ENDPOINT', '')}", flush=True)
    print(f"HF_TOKEN_present={bool(token)}", flush=True)
    print(f"cache_size_before={human_bytes(directory_size(cache_dir))}", flush=True)

    failures = 0
    skipped = 0
    for repo_id in iter_repos(args):
        print(f"\n===== downloading {repo_id} =====", flush=True)
        print(f"before: {repo_stats(cache_dir, repo_id)}", flush=True)
        if repo_id.startswith("meta-llama/") and not token and not args.attempt_gated_without_token:
            skipped += 1
            print("SKIPPED_NEEDS_HF_TOKEN_AND_ACCEPTED_META_LICENSE", flush=True)
            continue
        try:
            path, elapsed = download_repo(repo_id, token, args.max_workers)
            print(f"DONE path={path} elapsed={elapsed}", flush=True)
        except Exception as exc:  # noqa: BLE001 - keep downloading other repos.
            failures += 1
            print(f"ERROR {type(exc).__name__}: {str(exc)[:1000]}", flush=True)
        print(f"after: {repo_stats(cache_dir, repo_id)}", flush=True)

    print(f"\ncache_size_after={human_bytes(directory_size(cache_dir))}", flush=True)
    print(f"summary failures={failures} skipped={skipped}", flush=True)
    return 1 if failures or (skipped and not args.allow_gated_skip) else 0


if __name__ == "__main__":
    raise SystemExit(main())
