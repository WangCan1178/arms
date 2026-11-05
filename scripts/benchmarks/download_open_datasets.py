#!/usr/bin/env python3
"""Download open benchmark sources for the Free part of the M^2 dataset.

This script only downloads or prepares raw open benchmark resources. It does
not run VLM inference and does not call commercial model APIs.
"""

from __future__ import annotations

import argparse
import datetime as dt
import inspect
import json
import logging
import os
import shutil
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
HF_HOME = DATA_ROOT / ".hf_cache"

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import requests
from git import GitCommandError, Repo
from huggingface_hub import snapshot_download


RAW_ROOT = DATA_ROOT / "raw"
LOG_ROOT = DATA_ROOT / "logs"

HF_DATASETS: Dict[str, Dict[str, Path | str]] = {
    "ALM-Bench": {
        "repo_id": "MBZUAI/ALM-Bench",
        "target": RAW_ROOT / "ALM-Bench",
    },
    "MMT-Bench": {
        "repo_id": "OpenGVLab/MMT-Bench",
        "target": RAW_ROOT / "MMT-Bench",
    },
    "MME-RealWorld-Base64": {
        "repo_id": "yifanzhang114/MME-RealWorld-Base64",
        "target": RAW_ROOT / "MME-RealWorld-Base64",
    },
}

VHELM_ROOT = RAW_ROOT / "VHELM"
VHELM_HELM_REPO = VHELM_ROOT / "helm"
VHELM_OFFICIAL_PAGES = VHELM_ROOT / "official_pages"
VHELM_STATUS_README = VHELM_ROOT / "README_DOWNLOAD_STATUS.md"

VHELM_RESOURCES = {
    "helm_git": "https://github.com/stanford-crfm/helm.git",
    "helm_zip": "https://codeload.github.com/stanford-crfm/helm/zip/refs/heads/main",
    "schema_vhelm": (
        "https://raw.githubusercontent.com/stanford-crfm/helm/"
        "refs/heads/main/src/helm/benchmark/static/schema_vhelm.yaml"
    ),
    "docs_latest": "https://crfm-helm.readthedocs.io/en/latest/vhelm/",
    "docs_v0_5_4": "https://crfm-helm.readthedocs.io/en/v0.5.4/vhelm/",
    "release_v2_0_1": "https://crfm.stanford.edu/helm/vhelm/v2.0.1/",
}


def configure_logging() -> logging.Logger:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_ROOT / f"download_open_datasets_{stamp}.log"

    logger = logging.getLogger("download_open_datasets")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Writing log to %s", log_file)
    return logger


def create_directories(logger: logging.Logger) -> None:
    for path in [
        RAW_ROOT / "ALM-Bench",
        RAW_ROOT / "MMT-Bench",
        RAW_ROOT / "MME-RealWorld-Base64",
        VHELM_ROOT,
        LOG_ROOT,
    ]:
        path.mkdir(parents=True, exist_ok=True)
        logger.info("Ensured directory: %s", path)
    HF_HOME.mkdir(parents=True, exist_ok=True)
    logger.info("HF_HOME=%s", HF_HOME)
    logger.info("HF_HUB_ENABLE_HF_TRANSFER=%s", os.environ.get("HF_HUB_ENABLE_HF_TRANSFER"))
    logger.info("HF_HUB_DISABLE_XET=%s", os.environ.get("HF_HUB_DISABLE_XET"))
    logger.info("HF_ENDPOINT=%s", os.environ.get("HF_ENDPOINT", "https://huggingface.co (default)"))


def iter_files(root: Path) -> Iterable[Path]:
    if not root.exists():
        return []
    return (path for path in root.rglob("*") if path.is_file())


def summarize_directory(root: Path, limit: int = 20) -> Dict[str, Any]:
    files = sorted(iter_files(root))
    total_size = sum(path.stat().st_size for path in files)
    first_paths = [str(path.relative_to(root)) for path in files[:limit]]
    return {
        "path": str(root),
        "file_count": len(files),
        "total_size_bytes": total_size,
        "total_size_gib": round(total_size / (1024**3), 4),
        "first_files": first_paths,
    }


def log_summary(logger: logging.Logger, name: str, summary: Dict[str, Any]) -> None:
    logger.info("%s file count: %s", name, summary["file_count"])
    logger.info("%s total size: %s bytes (%.4f GiB)", name, summary["total_size_bytes"], summary["total_size_gib"])
    logger.info("%s first %d files:", name, len(summary["first_files"]))
    for rel_path in summary["first_files"]:
        logger.info("  %s", rel_path)


def token_from_environment() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def snapshot_download_kwargs(repo_id: str, target: Path) -> Dict[str, Any]:
    token = token_from_environment()
    kwargs: Dict[str, Any] = {
        "repo_id": repo_id,
        "repo_type": "dataset",
        "local_dir": str(target),
        "cache_dir": str(HF_HOME),
        "token": token,
        "max_workers": 8,
    }

    signature = inspect.signature(snapshot_download)
    if "resume_download" in signature.parameters:
        kwargs["resume_download"] = True
    if "local_dir_use_symlinks" in signature.parameters:
        kwargs["local_dir_use_symlinks"] = False
    return kwargs


def download_hf_dataset(name: str, repo_id: str, target: Path, logger: logging.Logger) -> Dict[str, Any]:
    logger.info("Starting Hugging Face snapshot_download for %s (%s)", name, repo_id)
    try:
        target.mkdir(parents=True, exist_ok=True)
        snapshot_path = snapshot_download(**snapshot_download_kwargs(repo_id, target))
        summary = summarize_directory(target)
        log_summary(logger, name, summary)
        return {
            "name": name,
            "repo_id": repo_id,
            "target": str(target),
            "snapshot_path": snapshot_path,
            "status": "ok",
            "summary": summary,
        }
    except Exception as exc:  # noqa: BLE001 - keep the batch moving and log exact failure.
        logger.exception("Failed to download %s (%s): %s", name, repo_id, exc)
        summary = summarize_directory(target)
        return {
            "name": name,
            "repo_id": repo_id,
            "target": str(target),
            "status": "failed",
            "error": repr(exc),
            "summary": summary,
        }


def clone_or_update_helm(logger: logging.Logger) -> Dict[str, Any]:
    repo_url = VHELM_RESOURCES["helm_git"]
    if os.environ.get("VHELM_HELM_SOURCE", "").lower() == "zip":
        if (VHELM_HELM_REPO / "src" / "helm" / "benchmark").exists():
            logger.info("VHELM_HELM_SOURCE=zip and existing HELM tree found at %s", VHELM_HELM_REPO)
            return {
                "status": "ok",
                "method": "zip-existing",
                "path": str(VHELM_HELM_REPO),
                "commit": "unavailable-from-zip",
                "url": VHELM_RESOURCES["helm_zip"],
            }
        logger.info("VHELM_HELM_SOURCE=zip; skipping git clone and using zip fallback.")
        return download_helm_zip_fallback(logger, git_error=RuntimeError("git skipped by VHELM_HELM_SOURCE=zip"))

    try:
        if (VHELM_HELM_REPO / ".git").exists():
            logger.info("HELM repo exists; fetching latest shallow metadata at %s", VHELM_HELM_REPO)
            repo = Repo(VHELM_HELM_REPO)
            try:
                repo.remotes.origin.fetch("--depth=1")
                active_branch = repo.active_branch.name
                repo.remotes.origin.pull(active_branch, "--ff-only")
            except TypeError:
                repo.remotes.origin.pull("--ff-only")
            except Exception as exc:  # noqa: BLE001
                logger.warning("HELM update failed; keeping existing clone: %s", exc)
        elif VHELM_HELM_REPO.exists() and any(VHELM_HELM_REPO.iterdir()):
            raise RuntimeError(f"{VHELM_HELM_REPO} exists but is not a git repository")
        else:
            logger.info("Cloning HELM repository from %s", repo_url)
            Repo.clone_from(repo_url, VHELM_HELM_REPO, depth=1)

        repo = Repo(VHELM_HELM_REPO)
        commit = repo.head.commit.hexsha
        logger.info("Prepared HELM repo at commit %s", commit)
        return {"status": "ok", "method": "git", "path": str(VHELM_HELM_REPO), "commit": commit, "url": repo_url}
    except (GitCommandError, RuntimeError, OSError) as exc:
        logger.exception("Failed to prepare HELM repository with git: %s", exc)
        return download_helm_zip_fallback(logger, git_error=exc)


def download_helm_zip_fallback(logger: logging.Logger, git_error: Exception) -> Dict[str, Any]:
    zip_url = VHELM_RESOURCES["helm_zip"]
    archive_path = VHELM_ROOT / "helm-main.zip"
    extract_root = VHELM_ROOT / "_helm_zip_extract"
    logger.info("Trying HELM zip fallback from %s", zip_url)
    try:
        VHELM_ROOT.mkdir(parents=True, exist_ok=True)
        with requests.get(
            zip_url,
            timeout=(30, 300),
            stream=True,
            headers={"User-Agent": "m2-free-dataset-downloader/1.0"},
        ) as response:
            response.raise_for_status()
            with archive_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        handle.write(chunk)

        if extract_root.exists():
            shutil.rmtree(extract_root)
        extract_root.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(archive_path) as archive:
            archive.extractall(extract_root)

        extracted_dirs = [path for path in extract_root.iterdir() if path.is_dir()]
        if not extracted_dirs:
            raise RuntimeError(f"No directory found inside {archive_path}")

        if VHELM_HELM_REPO.exists():
            shutil.rmtree(VHELM_HELM_REPO)
        shutil.move(str(extracted_dirs[0]), str(VHELM_HELM_REPO))
        shutil.rmtree(extract_root)

        logger.info("Prepared HELM repository from zip fallback at %s", VHELM_HELM_REPO)
        return {
            "status": "ok",
            "method": "zip",
            "path": str(VHELM_HELM_REPO),
            "commit": "unavailable-from-zip",
            "url": zip_url,
            "git_error": repr(git_error),
            "archive": str(archive_path),
        }
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed HELM zip fallback: %s", exc)
        return {
            "status": "failed",
            "method": "git+zip",
            "path": str(VHELM_HELM_REPO),
            "url": zip_url,
            "git_error": repr(git_error),
            "error": repr(exc),
        }


def download_url(url: str, target: Path, logger: logging.Logger) -> Dict[str, Any]:
    logger.info("Downloading VHELM resource: %s", url)
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        response = requests.get(url, timeout=60, headers={"User-Agent": "m2-free-dataset-downloader/1.0"})
        response.raise_for_status()
        target.write_bytes(response.content)
        logger.info("Saved %s (%d bytes)", target, target.stat().st_size)
        return {"status": "ok", "url": url, "path": str(target), "bytes": target.stat().st_size}
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to download %s: %s", url, exc)
        return {"status": "failed", "url": url, "path": str(target), "error": repr(exc)}


def find_vhelm_files() -> List[str]:
    if not VHELM_HELM_REPO.exists():
        return []
    patterns = [
        "src/helm/benchmark/static/schema_vhelm.yaml",
        "src/helm/benchmark/presentation/*vhelm*",
        "docs/**/*vhelm*",
        "src/helm/benchmark/scenarios/*vqa*",
        "src/helm/benchmark/scenarios/*mmmu*",
        "src/helm/benchmark/scenarios/*vizwiz*",
    ]
    matches: List[Path] = []
    for pattern in patterns:
        matches.extend(VHELM_HELM_REPO.glob(pattern))
    return sorted({str(path.relative_to(VHELM_HELM_REPO)) for path in matches if path.is_file()})[:80]


def write_vhelm_status(vhelm_status: Dict[str, Any], logger: logging.Logger) -> None:
    now = dt.datetime.now().isoformat(timespec="seconds")
    helm_repo = vhelm_status.get("helm_repo", {})
    downloaded = vhelm_status.get("downloads", {})
    candidate_files = find_vhelm_files()

    lines = [
        "# VHELM Download Status",
        "",
        f"Generated at: `{now}`",
        "",
        "This directory prepares official Stanford CRFM HELM/VHELM resources only.",
        "It does not run model inference and does not call OpenAI, Anthropic, Google, or other commercial model APIs.",
        "",
        "## Official Sources Checked",
        "",
        f"- HELM GitHub repository: {VHELM_RESOURCES['helm_git']}",
        f"- VHELM documentation: {VHELM_RESOURCES['docs_latest']}",
        f"- VHELM v2.0.1 release/leaderboard: {VHELM_RESOURCES['release_v2_0_1']}",
        f"- VHELM schema: {VHELM_RESOURCES['schema_vhelm']}",
        "",
        "## Prepared Locally",
        "",
        f"- HELM repository status: `{helm_repo.get('status', 'unknown')}`",
        f"- HELM preparation method: `{helm_repo.get('method', 'unknown')}`",
        f"- HELM repository path: `{helm_repo.get('path', VHELM_HELM_REPO)}`",
        f"- HELM commit: `{helm_repo.get('commit', 'unavailable')}`",
    ]

    for key, item in downloaded.items():
        lines.append(f"- {key}: `{item.get('status', 'unknown')}` -> `{item.get('path', '')}`")

    lines.extend(
        [
            "",
            "## Candidate HELM/VHELM Files",
            "",
        ]
    )
    if candidate_files:
        lines.extend(f"- `{path}`" for path in candidate_files)
    else:
        lines.append("- No candidate files found yet. Check HELM clone/download errors above.")

    lines.extend(
        [
            "",
            "## What Is Still Missing",
            "",
            "- VHELM is not treated here as a single Hugging Face dataset.",
            "- Scenario-specific raw datasets, query text, image references, and ground truth are restored through HELM scenario code and its documented VHELM run entries.",
            "- This script intentionally does not execute VLM inference or download commercial model outputs.",
            "",
            "## Next Restoration Step",
            "",
            "1. Install HELM with VLM extras in a separate environment if needed: `pip install \"crfm-helm[vlm]\"`.",
            "2. Inspect the VHELM schema and scenario implementations under the cloned HELM repository.",
            "3. Use HELM scenario code to materialize instances/query/image/ground-truth only, with no commercial model credentials and no model API calls.",
            "4. Save the restored VHELM raw queries/images/ground truth under this directory before later M^2 sampling.",
        ]
    )

    VHELM_STATUS_README.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote VHELM status README: %s", VHELM_STATUS_README)


def prepare_vhelm(logger: logging.Logger) -> Dict[str, Any]:
    VHELM_ROOT.mkdir(parents=True, exist_ok=True)
    VHELM_OFFICIAL_PAGES.mkdir(parents=True, exist_ok=True)

    helm_repo_status = clone_or_update_helm(logger)
    schema_status = download_url(VHELM_RESOURCES["schema_vhelm"], VHELM_ROOT / "schema_vhelm.yaml", logger)
    if schema_status.get("status") != "ok":
        local_schema = VHELM_HELM_REPO / "src" / "helm" / "benchmark" / "static" / "schema_vhelm.yaml"
        if local_schema.exists():
            target_schema = VHELM_ROOT / "schema_vhelm.yaml"
            shutil.copy2(local_schema, target_schema)
            schema_status = {
                "status": "ok",
                "method": "helm_repo_copy",
                "url": VHELM_RESOURCES["schema_vhelm"],
                "path": str(target_schema),
                "source": str(local_schema),
                "bytes": target_schema.stat().st_size,
            }
            logger.info("Copied VHELM schema from HELM repo fallback: %s", local_schema)

    downloads = {
        "schema_vhelm": schema_status,
        "docs_latest": download_url(VHELM_RESOURCES["docs_latest"], VHELM_OFFICIAL_PAGES / "vhelm_docs_latest.html", logger),
        "docs_v0_5_4": download_url(VHELM_RESOURCES["docs_v0_5_4"], VHELM_OFFICIAL_PAGES / "vhelm_docs_v0_5_4.html", logger),
        "release_v2_0_1": download_url(
            VHELM_RESOURCES["release_v2_0_1"], VHELM_OFFICIAL_PAGES / "vhelm_release_v2_0_1.html", logger
        ),
    }

    status = {"helm_repo": helm_repo_status, "downloads": downloads}
    write_vhelm_status(status, logger)
    log_summary(logger, "VHELM", summarize_directory(VHELM_ROOT))
    return status


def write_status_json(results: Dict[str, Any], logger: logging.Logger) -> Path:
    status_path = LOG_ROOT / "download_open_datasets_status.json"
    status_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote machine-readable status: %s", status_path)
    return status_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-hf", action="store_true", help="Skip Hugging Face dataset downloads.")
    parser.add_argument("--skip-vhelm", action="store_true", help="Skip HELM/VHELM resource preparation.")
    parser.add_argument(
        "--fail-on-error",
        action="store_true",
        help="Exit non-zero if any dataset/resource fails after all attempts finish.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logging()
    create_directories(logger)

    if token_from_environment():
        logger.info("HF_TOKEN/HUGGING_FACE_HUB_TOKEN detected in environment.")
    else:
        logger.info("No HF token detected. Public datasets will still download; gated datasets may fail.")

    results: Dict[str, Any] = {"hf_home": str(HF_HOME), "hf_datasets": {}, "vhelm": {}}

    if not args.skip_hf:
        for name, item in HF_DATASETS.items():
            results["hf_datasets"][name] = download_hf_dataset(
                name=name,
                repo_id=str(item["repo_id"]),
                target=Path(item["target"]),
                logger=logger,
            )
    else:
        logger.info("Skipping Hugging Face datasets by request.")

    if not args.skip_vhelm:
        results["vhelm"] = prepare_vhelm(logger)
    else:
        logger.info("Skipping VHELM preparation by request.")

    write_status_json(results, logger)

    failed = [
        name for name, item in results["hf_datasets"].items() if item.get("status") != "ok"
    ]
    vhelm_failed = [
        key
        for key, item in results.get("vhelm", {}).get("downloads", {}).items()
        if item.get("status") != "ok"
    ]
    if results.get("vhelm", {}).get("helm_repo", {}).get("status") == "failed":
        vhelm_failed.append("helm_repo")

    if failed or vhelm_failed:
        logger.warning("Completed with failures. HF=%s VHELM=%s", failed, vhelm_failed)
        return 1 if args.fail_on_error else 0

    logger.info("All requested open dataset resources finished without recorded failures.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
