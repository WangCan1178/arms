#!/usr/bin/env python3
"""Download official VHELM v2.0.1 instance files needed to restore OOD queries."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List

import requests
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VHELM_ROOT = PROJECT_ROOT / "data" / "raw" / "VHELM"
GCS_API = "https://storage.googleapis.com/storage/v1/b/crfm-helm-public/o"
GCS_OBJECT_BASE = "https://storage.googleapis.com/crfm-helm-public/"
DEFAULT_SUITE = "v2.0.1"
DEFAULT_MODEL_DIR = "HuggingFaceM4_idefics2-8b"
DEFAULT_FILES = ["instances.json", "run_spec.json", "scenario.json"]


def list_run_prefixes(suite: str, timeout: int) -> List[str]:
    prefix = f"vhelm/benchmark_output/runs/{suite}/"
    response = requests.get(GCS_API, params={"prefix": prefix, "delimiter": "/"}, timeout=timeout)
    response.raise_for_status()
    return list(response.json().get("prefixes", []))


def select_prefixes(prefixes: Iterable[str], model_dir: str) -> List[str]:
    selected = [prefix for prefix in prefixes if f"model={model_dir}" in prefix]
    return sorted(selected)


def local_path_for_object(vhelm_root: Path, object_name: str) -> Path:
    prefix = "vhelm/"
    if object_name.startswith(prefix):
        object_name = object_name[len(prefix) :]
    return vhelm_root / object_name


def download_object(object_name: str, output_path: Path, timeout: int, force: bool) -> Dict[str, Any]:
    if output_path.exists() and not force:
        return {"object": object_name, "path": str(output_path), "status": "exists", "size": output_path.stat().st_size}
    url = GCS_OBJECT_BASE + object_name
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_bytes(response.content)
    return {"object": object_name, "path": str(output_path), "status": "downloaded", "size": output_path.stat().st_size}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", default=DEFAULT_SUITE)
    parser.add_argument("--model-dir", default=DEFAULT_MODEL_DIR)
    parser.add_argument("--vhelm-root", type=Path, default=VHELM_ROOT)
    parser.add_argument("--file", action="append", dest="files", default=None)
    parser.add_argument("--timeout", type=int, default=60)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    files = args.files or DEFAULT_FILES
    prefixes = list_run_prefixes(args.suite, args.timeout)
    selected = select_prefixes(prefixes, args.model_dir)
    if not selected:
        raise SystemExit(f"No VHELM run directories matched model={args.model_dir!r} in suite {args.suite!r}")

    manifest = {
        "suite": args.suite,
        "model_dir": args.model_dir,
        "selected_run_dirs": [prefix.removeprefix(f"vhelm/benchmark_output/runs/{args.suite}/").strip("/") for prefix in selected],
        "files": [],
    }
    for run_prefix in tqdm(selected, desc="download VHELM run files", unit="run"):
        for filename in files:
            object_name = run_prefix + filename
            output_path = local_path_for_object(args.vhelm_root, object_name)
            try:
                manifest["files"].append(download_object(object_name, output_path, args.timeout, args.force))
            except Exception as exc:  # noqa: BLE001 - keep downloading independent run directories.
                manifest["files"].append(
                    {"object": object_name, "path": str(output_path), "status": "failed", "error": f"{type(exc).__name__}: {exc}"}
                )

    manifest_path = args.vhelm_root / "vhelm_v2_0_1_instances_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    failures = [item for item in manifest["files"] if item.get("status") == "failed"]
    print(f"selected_run_dirs={len(selected)} files={len(manifest['files'])} failures={len(failures)} manifest={manifest_path}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
