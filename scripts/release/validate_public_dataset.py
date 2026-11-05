#!/usr/bin/env python3
"""Validate the compact public ARMS routing-supervision release."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterator


EXPECTED_COUNTS = {
    "original_pool_id_train": 29_626,
    "original_pool_id_test": 3_000,
    "original_pool_ood_test": 7_084,
}
EXPECTED_MODELS = {
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "HuggingFaceM4/idefics2-8b",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
}
REQUIRED_FIELDS = {
    "query_id",
    "benchmark",
    "task",
    "language",
    "answer_type",
    "image_key",
    "candidate_labels",
}
EXPECTED_OOD_TASKS = {
    "a_okvqa_base",
    "a_okvqa_chinese",
    "a_okvqa_dialect",
    "a_okvqa_hindi",
    "a_okvqa_robustness",
    "a_okvqa_spanish",
}
MACHINE_PATH_MARKERS = ("/root/", "/Users/", "/home/")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_rows(path: Path) -> Iterator[Dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise AssertionError(f"{path}:{line_number}: invalid JSON: {exc}") from exc
            assert isinstance(row, dict), f"{path}:{line_number}: row is not an object"
            yield row


def validate_split(path: Path, expected_count: int) -> tuple[Dict[str, Any], set[str], set[str]]:
    seen: set[str] = set()
    tasks: set[str] = set()
    positive_pairs = 0
    count = 0

    for row in iter_rows(path):
        assert set(row) == REQUIRED_FIELDS, f"{path}: schema mismatch in {row.get('query_id')}"
        query_id = row["query_id"]
        assert isinstance(query_id, str) and query_id, f"{path}: empty query_id"
        assert query_id not in seen, f"{path}: duplicate query_id {query_id}"
        seen.add(query_id)

        image_key = row["image_key"]
        assert isinstance(image_key, str) and image_key.startswith("data/"), (
            f"{query_id}: image_key must be relative to the reconstruction root"
        )
        assert not Path(image_key).is_absolute() and ".." not in Path(image_key).parts, (
            f"{query_id}: unsafe image_key"
        )
        rendered = json.dumps(row, ensure_ascii=False)
        assert not any(marker in rendered for marker in MACHINE_PATH_MARKERS), (
            f"{query_id}: machine-specific path detected"
        )

        labels = row["candidate_labels"]
        assert isinstance(labels, dict) and set(labels) == EXPECTED_MODELS, (
            f"{query_id}: candidate set mismatch"
        )
        assert all(isinstance(value, bool) for value in labels.values()), (
            f"{query_id}: candidate labels must be boolean"
        )
        assert any(labels.values()), f"{query_id}: no candidate answered correctly"

        tasks.add(str(row["task"]))
        positive_pairs += sum(labels.values())
        count += 1

    assert count == expected_count, f"{path}: expected {expected_count}, found {count}"
    return {"queries": count, "positive_pairs": positive_pairs}, seen, tasks


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("data/original_pool"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = args.data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["schema_version"] == "1.0"
    assert set(manifest["candidate_order"]) == EXPECTED_MODELS
    assert set(manifest["record_fields"]) == REQUIRED_FIELDS

    report: Dict[str, Any] = {"splits": {}}
    query_ids: Dict[str, set[str]] = {}
    split_tasks: Dict[str, set[str]] = {}
    for split_name, expected_count in EXPECTED_COUNTS.items():
        entry = manifest["splits"][split_name]
        path = args.data_dir / entry["file"]
        assert path.is_file(), f"missing data file: {path}"
        assert entry["queries"] == expected_count
        assert sha256(path) == entry["sha256"], f"checksum mismatch: {path}"
        result, ids, tasks = validate_split(path, expected_count)
        report["splits"][split_name] = result
        query_ids[split_name] = ids
        split_tasks[split_name] = tasks

    split_names = list(query_ids)
    for index, left in enumerate(split_names):
        for right in split_names[index + 1 :]:
            assert query_ids[left].isdisjoint(query_ids[right]), (
                f"{left} and {right} contain overlapping query IDs"
            )
    assert split_tasks["original_pool_ood_test"] == EXPECTED_OOD_TASKS, (
        "OOD task selection does not match the paper split"
    )

    report["total_queries"] = sum(EXPECTED_COUNTS.values())
    report["total_query_model_labels"] = report["total_queries"] * len(EXPECTED_MODELS)
    report["status"] = "passed"
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
