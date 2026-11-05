#!/usr/bin/env python3
"""Inspect downloaded open benchmark sources for M^2 Free reproduction."""

from __future__ import annotations

import argparse
import base64
import binascii
import datetime as dt
import inspect
import io
import json
import logging
import os
import re
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
HF_HOME = DATA_ROOT / ".hf_cache"

os.environ.setdefault("HF_HOME", str(HF_HOME))
os.environ.setdefault("HF_HUB_ENABLE_HF_TRANSFER", "1")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")

import pandas as pd
import pyarrow.parquet as pq
from datasets import (
    Dataset,
    DatasetDict,
    IterableDataset,
    IterableDatasetDict,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
    load_dataset_builder,
)
from PIL import Image, UnidentifiedImageError


RAW_ROOT = DATA_ROOT / "raw"
LOG_ROOT = DATA_ROOT / "logs"
DEBUG_SAMPLE_ROOT = DATA_ROOT / "debug_samples"

DATASETS = {
    "ALM-Bench": {
        "repo_id": "MBZUAI/ALM-Bench",
        "path": RAW_ROOT / "ALM-Bench",
        "expected_fields": [
            "ID",
            "Language",
            "Category",
            "Question_Type",
            "English_Question",
            "English_Answer",
            "Translated_Question",
            "Translated_Answer",
            "Image_Url",
        ],
    },
    "MMT-Bench": {
        "repo_id": "OpenGVLab/MMT-Bench",
        "path": RAW_ROOT / "MMT-Bench",
    },
    "MME-RealWorld-Base64": {
        "repo_id": "yifanzhang114/MME-RealWorld-Base64",
        "path": RAW_ROOT / "MME-RealWorld-Base64",
    },
}

VHELM_ROOT = RAW_ROOT / "VHELM"
VHELM_STATUS_README = VHELM_ROOT / "README_DOWNLOAD_STATUS.md"


def configure_logging() -> logging.Logger:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = LOG_ROOT / f"inspect_downloads_{stamp}.log"

    logger = logging.getLogger("inspect_downloads")
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


def token_from_environment() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")


def supports_kwarg(func: Any, name: str) -> bool:
    signature = inspect.signature(func)
    return name in signature.parameters or any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD for parameter in signature.parameters.values()
    )


def dataset_common_kwargs(func: Any) -> Dict[str, Any]:
    kwargs: Dict[str, Any] = {}
    token = token_from_environment()
    if token:
        if supports_kwarg(func, "token"):
            kwargs["token"] = token
        elif supports_kwarg(func, "use_auth_token"):
            kwargs["use_auth_token"] = token
    if supports_kwarg(func, "cache_dir"):
        kwargs["cache_dir"] = str(HF_HOME)
    return kwargs


def is_nonempty_dir(path: Path) -> bool:
    return path.exists() and path.is_dir() and any(path.iterdir())


def dataset_source(repo_id: str, path: Path) -> str:
    return str(path) if is_nonempty_dir(path) else repo_id


def first_files(path: Path, limit: int = 10) -> List[str]:
    if not path.exists():
        return []
    return [str(file.relative_to(path)) for file in sorted(path.rglob("*")) if file.is_file()][:limit]


def get_configs(source: str) -> Tuple[List[str | None], str | None]:
    try:
        configs = get_dataset_config_names(source, **dataset_common_kwargs(get_dataset_config_names))
        return (configs or [None]), None
    except Exception as exc:  # noqa: BLE001 - keep inspection going.
        return [None], repr(exc)


def get_splits(source: str, config: str | None) -> Tuple[List[str], str | None]:
    kwargs = dataset_common_kwargs(get_dataset_split_names)
    if config:
        kwargs["name"] = config
    try:
        splits = get_dataset_split_names(source, **kwargs)
        return splits or ["train"], None
    except Exception as exc:  # noqa: BLE001
        return ["train", "test", "validation"], repr(exc)


def builder_split_counts(source: str, config: str | None) -> Dict[str, int | None]:
    kwargs = dataset_common_kwargs(load_dataset_builder)
    if config:
        kwargs["name"] = config
    try:
        builder = load_dataset_builder(source, **kwargs)
        if not builder.info.splits:
            return {}
        return {name: split.num_examples for name, split in builder.info.splits.items()}
    except Exception:
        return {}


def normalize_dataset_result(result: Any) -> Tuple[Any | None, List[str], int | None]:
    if isinstance(result, (DatasetDict, IterableDatasetDict)):
        if not result:
            return None, [], None
        first_split_name = next(iter(result.keys()))
        return normalize_dataset_result(result[first_split_name])
    if isinstance(result, Dataset):
        if len(result) == 0:
            return None, list(result.features.keys()), 0
        row = result[0]
        return row, list(result.features.keys()), len(result)
    if isinstance(result, IterableDataset):
        iterator = iter(result)
        row = next(iterator, None)
        fields = list(row.keys()) if isinstance(row, dict) else []
        return row, fields, None
    return None, [], None


def load_small_sample(
    source: str,
    config: str | None,
    splits: Sequence[str],
    *,
    allow_non_streaming: bool = True,
) -> Tuple[Any | None, List[str], str | None]:
    tried_errors: List[str] = []
    split_candidates = list(splits) if splits else ["train", "test", "validation"]
    for split in split_candidates:
        kwargs = dataset_common_kwargs(load_dataset)
        if config:
            kwargs["name"] = config
        kwargs["split"] = split
        kwargs["streaming"] = True
        try:
            result = load_dataset(source, **kwargs)
            row, fields, _ = normalize_dataset_result(result)
            if row is not None:
                return row, fields, None
        except Exception as exc:  # noqa: BLE001
            tried_errors.append(f"{split} streaming: {exc!r}")

        if not allow_non_streaming:
            continue

        kwargs = dataset_common_kwargs(load_dataset)
        if config:
            kwargs["name"] = config
        kwargs["split"] = f"{split}[:1]"
        try:
            result = load_dataset(source, **kwargs)
            row, fields, _ = normalize_dataset_result(result)
            if row is not None:
                return row, fields, None
        except Exception as exc:  # noqa: BLE001
            tried_errors.append(f"{split}[:1]: {exc!r}")

    return None, [], "; ".join(tried_errors[-4:])


def sample_from_json_file(file: Path, max_bytes: int = 256 * 1024 * 1024) -> Any | None:
    decoder = json.JSONDecoder()
    buffer = ""
    bytes_read = 0
    with file.open("r", encoding="utf-8") as handle:
        while bytes_read <= max_bytes:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            bytes_read += len(chunk.encode("utf-8"))
            buffer += chunk
            stripped = buffer.lstrip()
            if not stripped:
                continue

            if stripped[0] == "[":
                after_bracket = stripped[1:].lstrip()
                start = len(buffer) - len(stripped) + 1 + (len(stripped[1:]) - len(after_bracket))
            else:
                start = len(buffer) - len(stripped)

            try:
                obj, _ = decoder.raw_decode(buffer[start:])
                return obj
            except json.JSONDecodeError:
                continue
    return None


def sample_from_local_files(path: Path) -> Tuple[Any | None, List[str], str | None]:
    if not path.exists():
        return None, [], "local directory missing"

    parquet_files = sorted(path.rglob("*.parquet"))
    for file in parquet_files[:10]:
        try:
            parquet = pq.ParquetFile(file)
            for batch in parquet.iter_batches(batch_size=1):
                data = batch.to_pydict()
                row = {key: values[0] for key, values in data.items()}
                return row, list(row.keys()), str(file)
        except Exception:
            continue

    csv_files = sorted(path.rglob("*.csv"))
    for file in csv_files[:10]:
        try:
            frame = pd.read_csv(file, nrows=1)
            if not frame.empty:
                row = frame.iloc[0].to_dict()
                return row, list(frame.columns), str(file)
        except Exception:
            continue

    jsonl_files = sorted(path.rglob("*.jsonl"))
    for file in jsonl_files[:10]:
        try:
            with file.open("r", encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if line:
                        row = json.loads(line)
                        fields = list(row.keys()) if isinstance(row, dict) else []
                        return row, fields, str(file)
        except Exception:
            continue

    json_files = sorted(
        file for file in path.rglob("*.json") if ".cache" not in file.parts and file.name != "dataset_info.json"
    )
    for file in json_files[:10]:
        try:
            row = sample_from_json_file(file)
            if row is not None:
                fields = list(row.keys()) if isinstance(row, dict) else []
                return row, fields, str(file)
        except Exception:
            continue

    return None, [], "no readable parquet/csv/jsonl/json sample found"


def short_value(value: Any, max_len: int = 120) -> str:
    text = repr(value)
    if len(text) > max_len:
        return text[: max_len - 3] + "..."
    return text


def iter_nested_values(value: Any, prefix: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(value, dict):
        for key, nested in value.items():
            child_prefix = f"{prefix}.{key}" if prefix else str(key)
            yield from iter_nested_values(nested, child_prefix)
    elif isinstance(value, (list, tuple)):
        for index, nested in enumerate(value[:5]):
            child_prefix = f"{prefix}[{index}]"
            yield from iter_nested_values(nested, child_prefix)
    else:
        yield prefix, value


def decode_base64_image(value: Any) -> Image.Image | None:
    if isinstance(value, Image.Image):
        return value
    if isinstance(value, bytes):
        try:
            image = Image.open(io.BytesIO(value))
            image.load()
            return image
        except (UnidentifiedImageError, OSError):
            return None
    if not isinstance(value, str):
        return None

    text = value.strip()
    if len(text) < 100:
        return None
    if "," in text and text.lower().startswith("data:image"):
        text = text.split(",", 1)[1]
    text = re.sub(r"\s+", "", text)
    if len(text) % 4:
        text += "=" * (4 - len(text) % 4)
    if not re.fullmatch(r"[A-Za-z0-9+/=_-]+", text):
        return None

    try:
        raw = base64.b64decode(text, validate=False)
        image = Image.open(io.BytesIO(raw))
        image.load()
        return image
    except (binascii.Error, UnidentifiedImageError, OSError, ValueError):
        return None


def inspect_base64_image(row: Any, logger: logging.Logger) -> Dict[str, Any]:
    DEBUG_SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    if row is None:
        logger.warning("MME-RealWorld-Base64: no sample row available for base64 validation.")
        return {"status": "no_sample"}

    for field_path, value in iter_nested_values(row):
        image = decode_base64_image(value)
        if image is None:
            continue
        output_path = DEBUG_SAMPLE_ROOT / "mme_realworld_base64_sample.png"
        image.save(output_path)
        logger.info("MME-RealWorld-Base64: decoded image field `%s` -> %s", field_path, output_path)
        return {"status": "decoded", "field": field_path, "path": str(output_path), "size": image.size}

    logger.warning("MME-RealWorld-Base64: no decodable base64 image field found in sample fields.")
    return {"status": "not_found", "sample_type": type(row).__name__}


def inspect_hf_dataset(name: str, repo_id: str, path: Path, logger: logging.Logger) -> Dict[str, Any]:
    logger.info("========== Inspecting %s ==========", name)
    logger.info("Expected local path: %s", path)
    logger.info("Local directory exists and non-empty: %s", is_nonempty_dir(path))
    logger.info("First local files: %s", first_files(path))

    source = dataset_source(repo_id, path)
    logger.info("datasets.load_dataset source: %s", source)
    configs, config_error = get_configs(source)
    if config_error:
        logger.warning("%s config discovery failed: %s", name, config_error)
    logger.info("%s configs: %s", name, configs)

    first_config = configs[0] if configs else None
    splits, split_error = get_splits(source, first_config)
    if split_error:
        logger.warning("%s split discovery failed: %s", name, split_error)
    logger.info("%s splits for config %s: %s", name, first_config, splits)

    counts = builder_split_counts(source, first_config)
    if counts:
        logger.info("%s sample count estimates from DatasetInfo: %s", name, counts)
    else:
        logger.info("%s sample count estimates from DatasetInfo: unavailable", name)

    row, fields, load_error = load_small_sample(
        source,
        first_config,
        splits,
        allow_non_streaming=name != "MME-RealWorld-Base64",
    )
    fallback_source = None
    if row is None:
        logger.warning("%s datasets.load_dataset sample failed: %s", name, load_error)
        row, fields, fallback_source = sample_from_local_files(path)
        if row is not None:
            logger.info("%s local file fallback sample source: %s", name, fallback_source)
        else:
            logger.warning("%s local file fallback failed: %s", name, fallback_source)

    logger.info("%s fields: %s", name, fields)
    if isinstance(row, dict):
        preview = {key: short_value(value) for key, value in list(row.items())[:8]}
        logger.info("%s sample preview: %s", name, preview)

    result: Dict[str, Any] = {
        "name": name,
        "repo_id": repo_id,
        "path": str(path),
        "path_exists": is_nonempty_dir(path),
        "configs": configs,
        "splits": splits,
        "counts": counts,
        "fields": fields,
        "sample_loaded": row is not None,
        "load_error": load_error,
        "fallback_source": fallback_source,
    }

    if name == "ALM-Bench":
        expected_fields = DATASETS[name]["expected_fields"]
        logger.info("ALM-Bench expected reference fields: %s", expected_fields)
        missing_expected = [field for field in expected_fields if field not in fields]
        logger.info("ALM-Bench expected fields not found verbatim: %s", missing_expected)
        result["expected_fields"] = expected_fields
        result["missing_expected_fields"] = missing_expected

    if name == "MME-RealWorld-Base64":
        result["base64_image_check"] = inspect_base64_image(row, logger)

    return result


def write_vhelm_status(logger: logging.Logger) -> Dict[str, Any]:
    helm_repo = VHELM_ROOT / "helm"
    schema = VHELM_ROOT / "schema_vhelm.yaml"
    official_pages = VHELM_ROOT / "official_pages"
    candidate_files = []
    if helm_repo.exists():
        for pattern in [
            "src/helm/benchmark/static/schema_vhelm.yaml",
            "src/helm/benchmark/presentation/*vhelm*",
            "docs/**/*vhelm*",
            "src/helm/benchmark/scenarios/*vqa*",
            "src/helm/benchmark/scenarios/*mmmu*",
            "src/helm/benchmark/scenarios/*vizwiz*",
        ]:
            candidate_files.extend(helm_repo.glob(pattern))
    candidate_rel = sorted({str(path.relative_to(helm_repo)) for path in candidate_files if path.is_file()})[:80]

    status = {
        "vhelm_root": str(VHELM_ROOT),
        "helm_repo_exists": (helm_repo / ".git").exists(),
        "helm_tree_exists": (helm_repo / "src" / "helm" / "benchmark").exists(),
        "schema_exists": schema.exists(),
        "official_pages": first_files(official_pages, limit=20),
        "candidate_files": candidate_rel,
    }

    now = dt.datetime.now().isoformat(timespec="seconds")
    lines = [
        "# VHELM Download Status",
        "",
        f"Generated at: `{now}`",
        "",
        "This status file was generated by `scripts/benchmarks/inspect_downloads.py`.",
        "",
        "## Local Checks",
        "",
        f"- VHELM root: `{VHELM_ROOT}`",
        f"- HELM git repo cloned: `{status['helm_repo_exists']}`",
        f"- HELM source tree prepared: `{status['helm_tree_exists']}`",
        f"- VHELM schema downloaded: `{status['schema_exists']}`",
        f"- Saved official pages: `{status['official_pages']}`",
        "",
        "## Candidate Files for Query/Image/Ground-Truth Restoration",
        "",
    ]
    if candidate_rel:
        lines.extend(f"- `{path}`" for path in candidate_rel)
    else:
        lines.append("- No candidate files found. Re-run `scripts/benchmarks/download_open_datasets.py --skip-hf` and inspect logs.")

    lines.extend(
        [
            "",
            "## Current Limitation",
            "",
            "VHELM is prepared through the official Stanford CRFM HELM repository and documentation rather than a single Hugging Face dataset.",
            "The next step is to use HELM scenario code to materialize raw VHELM instances (query, image references/content, and ground truth) without running model inference or using commercial model credentials.",
        ]
    )
    VHELM_ROOT.mkdir(parents=True, exist_ok=True)
    VHELM_STATUS_README.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Wrote/updated VHELM status README: %s", VHELM_STATUS_README)
    logger.info("VHELM status: %s", status)
    return status


def write_status_json(results: Dict[str, Any], logger: logging.Logger) -> Path:
    LOG_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = LOG_ROOT / "inspect_downloads_status.json"
    status_path.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    logger.info("Wrote machine-readable inspection status: %s", status_path)
    return status_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-load-dataset", action="store_true", help="Only check files, skip datasets.load_dataset.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logger = configure_logging()
    DEBUG_SAMPLE_ROOT.mkdir(parents=True, exist_ok=True)
    logger.info("HF_HOME=%s", HF_HOME)

    results: Dict[str, Any] = {"hf_datasets": {}, "vhelm": {}}
    for name, item in DATASETS.items():
        if args.skip_load_dataset:
            path = Path(item["path"])
            logger.info("Skipping load_dataset for %s. Path exists: %s Files: %s", name, is_nonempty_dir(path), first_files(path))
            results["hf_datasets"][name] = {"path": str(path), "path_exists": is_nonempty_dir(path)}
            continue
        results["hf_datasets"][name] = inspect_hf_dataset(
            name=name,
            repo_id=str(item["repo_id"]),
            path=Path(item["path"]),
            logger=logger,
        )

    results["vhelm"] = write_vhelm_status(logger)
    write_status_json(results, logger)

    missing_paths = [
        name for name, result in results["hf_datasets"].items() if not result.get("path_exists", False)
    ]
    if missing_paths:
        logger.warning("Some Hugging Face dataset directories are missing or empty: %s", missing_paths)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
