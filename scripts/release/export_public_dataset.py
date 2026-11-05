#!/usr/bin/env python3
"""Create compact, portable routing-supervision files for the paper splits.

The input files are the full query-level JSONL files produced by
``merge_parse_label_filter.py``.  The output keeps only identifiers, routing
metadata, relative image keys, and binary candidate outcomes.  Benchmark text,
images, source URLs, raw model generations, machine-specific paths, and timing
or cost estimates are deliberately omitted.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import io
import json
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, TextIO


EXPECTED_MODELS = (
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "HuggingFaceM4/idefics2-8b",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
)

PAPER_OOD_TASKS = frozenset(
    {
        "a_okvqa_base",
        "a_okvqa_chinese",
        "a_okvqa_dialect",
        "a_okvqa_hindi",
        "a_okvqa_robustness",
        "a_okvqa_spanish",
    }
)

EXPECTED_COUNTS = {
    "original_pool_id_train": 29_626,
    "original_pool_id_test": 3_000,
    "original_pool_ood_test": 7_084,
}


@contextmanager
def open_text(path: Path, mode: str) -> Iterator[TextIO]:
    """Open plain or gzip-compressed UTF-8 text."""
    if "b" in mode:
        raise ValueError("open_text expects text mode")
    if path.suffix == ".gz":
        with gzip.open(path, mode, encoding="utf-8") as handle:
            yield handle
    else:
        with path.open(mode, encoding="utf-8") as handle:
            yield handle


@contextmanager
def deterministic_gzip_text(path: Path) -> Iterator[TextIO]:
    """Write gzip text with a fixed timestamp for byte-stable archives."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as zipped:
            with io.TextIOWrapper(zipped, encoding="utf-8", newline="\n") as text:
                yield text


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open_text(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                value = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: each row must be an object")
            yield value


def relative_image_key(value: Any) -> str:
    """Convert the experiment-server image path to a portable package key."""
    text = str(value or "").replace("\\", "/").strip()
    if not text:
        raise ValueError("empty image_path")
    marker = "/data/"
    if marker in text:
        return "data/" + text.split(marker, 1)[1].lstrip("/")
    path = Path(text)
    if path.is_absolute():
        raise ValueError(f"cannot safely relativize image_path: {text}")
    normalized = path.as_posix().lstrip("./")
    if normalized.startswith("../") or normalized == "..":
        raise ValueError(f"image_path escapes the package root: {text}")
    return normalized if normalized.startswith("data/") else f"data/{normalized}"


def compact_row(row: Dict[str, Any]) -> Dict[str, Any]:
    query_id = str(row.get("query_id") or "").strip()
    if not query_id:
        raise ValueError("missing query_id")

    outputs = row.get("model_outputs")
    if not isinstance(outputs, dict):
        raise ValueError(f"{query_id}: model_outputs must be an object")
    if set(outputs) != set(EXPECTED_MODELS):
        missing = sorted(set(EXPECTED_MODELS) - set(outputs))
        extra = sorted(set(outputs) - set(EXPECTED_MODELS))
        raise ValueError(f"{query_id}: candidate mismatch; missing={missing}, extra={extra}")

    labels: Dict[str, bool] = {}
    for model_name in EXPECTED_MODELS:
        output = outputs[model_name]
        if not isinstance(output, dict):
            raise ValueError(f"{query_id}: malformed output for {model_name}")
        error = str(output.get("error") or "").strip()
        if error:
            raise ValueError(f"{query_id}: inference error for {model_name}: {error[:120]}")
        label = output.get("label")
        if not isinstance(label, bool):
            raise ValueError(f"{query_id}: non-boolean label for {model_name}")
        labels[model_name] = label

    if not any(labels.values()):
        raise ValueError(f"{query_id}: answerability filter was not applied")

    return {
        "query_id": query_id,
        "benchmark": str(row.get("benchmark") or ""),
        "task": str(row.get("task") or ""),
        "language": str(row.get("language") or ""),
        "answer_type": str(row.get("answer_type") or ""),
        "image_key": relative_image_key(row.get("image_path")),
        "candidate_labels": labels,
    }


def write_split(
    input_path: Path,
    output_path: Path,
    split_name: str,
    allowed_tasks: frozenset[str] | None = None,
) -> Dict[str, Any]:
    seen: set[str] = set()
    task_counts: Counter[str] = Counter()
    benchmark_counts: Counter[str] = Counter()
    count = 0

    with deterministic_gzip_text(output_path) as handle:
        for source_row in iter_jsonl(input_path):
            task = str(source_row.get("task") or "")
            if allowed_tasks is not None and task not in allowed_tasks:
                continue
            row = compact_row(source_row)
            query_id = row["query_id"]
            if query_id in seen:
                raise ValueError(f"{input_path}: duplicate query_id: {query_id}")
            seen.add(query_id)
            task_counts[row["task"]] += 1
            benchmark_counts[row["benchmark"]] += 1
            handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            count += 1

    expected = EXPECTED_COUNTS[split_name]
    if count != expected:
        output_path.unlink(missing_ok=True)
        raise ValueError(f"{split_name}: expected {expected:,} rows, found {count:,}")

    return {
        "file": output_path.name,
        "queries": count,
        "benchmarks": dict(sorted(benchmark_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--id-train", type=Path, required=True)
    parser.add_argument("--id-test", type=Path, required=True)
    parser.add_argument("--ood-test", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    specifications = (
        ("original_pool_id_train", args.id_train, None),
        ("original_pool_id_test", args.id_test, None),
        ("original_pool_ood_test", args.ood_test, PAPER_OOD_TASKS),
    )

    split_manifest: Dict[str, Any] = {}
    for split_name, input_path, allowed_tasks in specifications:
        output_path = args.output_dir / f"{split_name}.jsonl.gz"
        split_manifest[split_name] = write_split(
            input_path=input_path,
            output_path=output_path,
            split_name=split_name,
            allowed_tasks=allowed_tasks,
        )

    for entry in split_manifest.values():
        entry["sha256"] = sha256(args.output_dir / entry["file"])

    manifest = {
        "schema_version": "1.0",
        "candidate_order": list(EXPECTED_MODELS),
        "record_fields": [
            "query_id",
            "benchmark",
            "task",
            "language",
            "answer_type",
            "image_key",
            "candidate_labels",
        ],
        "splits": split_manifest,
        "ood_selection": {
            "included_tasks": sorted(PAPER_OOD_TASKS),
            "excluded_tasks": ["a_okvqa_swahili", "crossmodal_3600", "bingo"],
        },
    }
    manifest_path = args.output_dir / "manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
