#!/usr/bin/env python3
"""Merge VLM outputs, parse answers, label correctness, and write M2 Free splits."""

from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional

try:
    from tqdm import tqdm
except ImportError:  # pragma: no cover - exercised only in minimal environments.
    def tqdm(iterable, **_: Any):
        return iterable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_ID_TRAIN = DATA_ROOT / "processed" / "m2_id_free_train.jsonl"
DEFAULT_ID_TEST = DATA_ROOT / "processed" / "m2_id_free_test.jsonl"
DEFAULT_OOD_TEST = DATA_ROOT / "processed" / "m2_ood_free_test.jsonl"
DEFAULT_ID_CANONICAL = DATA_ROOT / "processed" / "id_queries_standardized.jsonl"
CANONICAL_FIELDS = {
    "benchmark",
    "task",
    "language",
    "image_path",
    "question",
    "prompt",
    "ground_truth",
    "choices",
    "answer_type",
    "source",
}


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_canonical_inputs(path: Optional[Path]) -> Dict[str, Dict[str, Any]]:
    if path is None or not path.exists():
        return {}
    rows = {}
    for row in iter_jsonl(path):
        query_id = row.get("query_id")
        if query_id:
            rows[str(query_id)] = row
    return rows


def apply_canonical_fields(row: Dict[str, Any], canonical_by_id: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    canonical = canonical_by_id.get(str(row.get("query_id")))
    if not canonical:
        return row
    merged = dict(row)
    for key in CANONICAL_FIELDS:
        if key in canonical:
            merged[key] = canonical[key]
    return merged


def normalize_text(value: Any) -> str:
    return re.sub(r"\s+", " ", str(value or "").strip().lower())


def parse_choice_answer(text: Any, choices: List[Dict[str, Any]], extra_letters: str = "") -> str:
    output = str(text or "").strip()
    choice_letters = "".join(str(item.get("letter", "")).upper() for item in choices)
    letters = "".join(dict.fromkeys(choice_letters + extra_letters.upper())) or "ABCDEFGHI"
    match = re.search(rf"\b([{re.escape(letters)}])\b", output.upper())
    if match:
        return match.group(1)
    match = re.search(rf"answer\s*(?:is|:)?\s*([{re.escape(letters)}])", output, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return output[:1].upper() if output[:1].upper() in letters else ""


def parsed_answer(row: Dict[str, Any]) -> str:
    if row.get("answer_type") == "multiple_choice":
        return parse_choice_answer(row.get("vlm_output"), row.get("choices") or [], str(row.get("ground_truth") or ""))
    return normalize_text(row.get("vlm_output"))


def is_correct(row: Dict[str, Any], parsed: str) -> bool:
    truth = str(row.get("ground_truth") or "").strip()
    if not truth:
        return False
    if row.get("answer_type") == "multiple_choice":
        return parsed.upper() == truth.upper()
    return parsed == normalize_text(truth)


def load_outputs(
    paths: Iterable[Path],
    limit: Optional[int],
    canonical_by_id: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, List[Dict[str, Any]]]:
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    canonical_by_id = canonical_by_id or {}
    count = 0
    for path in paths:
        for row in tqdm(iter_jsonl(path), desc=f"reading {path.name}", unit="row"):
            query_id = row.get("query_id")
            if not query_id:
                continue
            row = apply_canonical_fields(row, canonical_by_id)
            parsed = parsed_answer(row)
            row["parsed_answer"] = parsed
            row["label"] = is_correct(row, parsed)
            grouped[str(query_id)].append(row)
            count += 1
            if limit is not None and count >= limit:
                return grouped
    return grouped


def merged_rows(grouped: Dict[str, List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    merged = []
    for query_id, rows in grouped.items():
        base = dict(rows[0])
        model_outputs = {}
        any_correct = False
        for row in rows:
            model = row.get("vlm_name") or "unknown"
            model_outputs[model] = {
                "vlm_output": row.get("vlm_output", ""),
                "error": row.get("error", ""),
                "parsed_answer": row.get("parsed_answer", ""),
                "label": bool(row.get("label")),
            }
            any_correct = any_correct or bool(row.get("label"))
        for key in ["vlm_name", "vlm_output", "error", "parsed_answer", "label"]:
            base.pop(key, None)
        base["model_outputs"] = model_outputs
        base["label"] = any_correct
        if any_correct:
            merged.append(base)
    merged.sort(key=lambda row: row["query_id"])
    return merged


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} rows to {path}")
    return count


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["id", "ood"], required=True)
    parser.add_argument("--inputs", type=Path, nargs="+", required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--id-train-output", type=Path, default=DEFAULT_ID_TRAIN)
    parser.add_argument("--id-test-output", type=Path, default=DEFAULT_ID_TEST)
    parser.add_argument("--ood-test-output", type=Path, default=DEFAULT_OOD_TEST)
    parser.add_argument(
        "--canonical-input",
        type=Path,
        default=None,
        help="Optional standardized query JSONL used to override stale metadata in raw VLM outputs before labeling.",
    )
    parser.add_argument("--train-size", type=int, default=29626)
    parser.add_argument("--test-size", type=int, default=3000)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical_path = args.canonical_input
    if canonical_path is None and args.mode == "id" and DEFAULT_ID_CANONICAL.exists():
        canonical_path = DEFAULT_ID_CANONICAL
    canonical_by_id = load_canonical_inputs(canonical_path)
    if canonical_by_id:
        print(f"canonical_inputs={len(canonical_by_id)} source={canonical_path}")
    grouped = load_outputs(args.inputs, args.limit, canonical_by_id)
    rows = merged_rows(grouped)
    print(f"queries_with_outputs={len(grouped)} queries_after_filter={len(rows)}")

    if args.mode == "ood":
        write_jsonl(args.ood_test_output, rows)
        return 0

    if len(rows) >= args.train_size + args.test_size:
        train_rows = rows[: args.train_size]
        test_rows = rows[args.train_size : args.train_size + args.test_size]
    else:
        split = max(0, len(rows) - min(args.test_size, max(1, len(rows) // 5)))
        train_rows = rows[:split]
        test_rows = rows[split:]
        print(
            f"Not enough rows for requested {args.train_size}+{args.test_size}; "
            f"using train={len(train_rows)} test={len(test_rows)}"
        )
    write_jsonl(args.id_train_output, train_rows)
    write_jsonl(args.id_test_output, test_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
