#!/usr/bin/env python3
"""Patch MMT-Bench ground-truth answers into an existing standardized ID JSONL."""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_INPUT = DATA_ROOT / "processed" / "id_queries_standardized.jsonl"
DEFAULT_MMT_ANSWER = DATA_ROOT / "raw" / "MMT-Bench" / "MMT-Bench_ALL_MI_anno.tsv"


AnswerMap = Dict[str, Tuple[str, str]]


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def load_answer_map(paths: Iterable[Path]) -> AnswerMap:
    csv.field_size_limit(sys.maxsize)
    answers: AnswerMap = {}
    for path in paths:
        if not path.exists():
            raise FileNotFoundError(f"MMT answer file does not exist: {path}")
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if "index" not in (reader.fieldnames or []) or "answer" not in (reader.fieldnames or []):
                raise ValueError(f"MMT answer file must contain index and answer columns: {path}")
            for row in reader:
                raw_id = str(row.get("index") or "").strip()
                answer = str(row.get("answer") or "").strip().upper()
                if raw_id and answer:
                    answers[raw_id] = (answer, path.name)
    return answers


def mmt_index(row: Mapping[str, Any]) -> str:
    source = row.get("source")
    if isinstance(source, dict):
        raw_id = str(source.get("index") or "").strip()
        if raw_id:
            return raw_id
    query_id = str(row.get("query_id") or "")
    return query_id.rsplit(":", 1)[-1].strip()


def patch_row(row: Dict[str, Any], answers: AnswerMap) -> Tuple[Dict[str, Any], bool, str]:
    if row.get("benchmark") != "MMT-Bench":
        return row, False, "non_mmt"
    if str(row.get("ground_truth") or "").strip():
        return row, False, "already_had_ground_truth"

    raw_id = mmt_index(row)
    answer, source_name = answers.get(raw_id, ("", ""))
    if not answer:
        return row, False, "missing_annotation"

    patched = dict(row)
    patched["ground_truth"] = answer
    source = dict(patched.get("source") or {})
    source["answer_source"] = source_name
    patched["source"] = source
    return patched, True, "patched"


def patch_file(input_path: Path, output_path: Path, answers: AnswerMap) -> Counter:
    stats: Counter = Counter()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in iter_jsonl(input_path):
            stats["rows"] += 1
            if row.get("benchmark") == "MMT-Bench":
                stats["mmt_rows"] += 1
            patched, changed, status = patch_row(row, answers)
            stats[status] += 1
            if changed:
                stats["patched_rows"] += 1
            handle.write(json.dumps(patched, ensure_ascii=False) + "\n")
    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=None, help="Defaults to in-place patching of --input.")
    parser.add_argument("--answer-files", type=Path, nargs="+", default=[DEFAULT_MMT_ANSWER])
    parser.add_argument("--no-backup", action="store_true", help="Skip backup when patching in place.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    input_path = args.input
    output_path = args.output or input_path
    answers = load_answer_map(args.answer_files)
    if not answers:
        raise RuntimeError("No MMT answers loaded from annotation files.")

    temp_path = output_path.with_name(output_path.name + ".tmp")
    stats = patch_file(input_path, temp_path, answers)
    if output_path == input_path:
        if not args.no_backup:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = input_path.with_name(f"{input_path.stem}.before_mmt_answers_{timestamp}{input_path.suffix}")
            shutil.copy2(input_path, backup_path)
            print(f"backup={backup_path}")
        temp_path.replace(input_path)
    else:
        temp_path.replace(output_path)

    print(f"answers_loaded={len(answers)}")
    for key, value in sorted(stats.items()):
        print(f"{key}={value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
