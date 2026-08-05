#!/usr/bin/env python3
"""Join compact routing labels with locally reconstructed benchmark queries."""

from __future__ import annotations

import argparse
import gzip
import json
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, Iterator, TextIO


@contextmanager
def open_text(path: Path, mode: str) -> Iterator[TextIO]:
    if path.suffix == ".gz":
        with gzip.open(path, mode, encoding="utf-8") as handle:
            yield handle
    else:
        with path.open(mode, encoding="utf-8") as handle:
            yield handle


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with open_text(path, "rt") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def load_queries(paths: list[Path]) -> Dict[str, Dict[str, Any]]:
    records: Dict[str, Dict[str, Any]] = {}
    for path in paths:
        for row in iter_jsonl(path):
            query_id = str(row.get("query_id") or "").strip()
            if not query_id:
                raise ValueError(f"{path}: row without query_id")
            previous = records.get(query_id)
            if previous is not None and previous != row:
                raise ValueError(f"conflicting canonical rows for {query_id}")
            records[query_id] = row
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", type=Path, required=True, help="Compact *.jsonl.gz split.")
    parser.add_argument(
        "--queries",
        type=Path,
        nargs="+",
        required=True,
        help="One or more standardized public-benchmark query JSONL files.",
    )
    parser.add_argument("--project-root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-missing-images",
        action="store_true",
        help="Write records even when reconstructed image files are not present yet.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    canonical = load_queries(args.queries)
    project_root = args.project_root.resolve()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    missing_queries: list[str] = []
    missing_images: list[str] = []
    written = 0
    with args.output.open("w", encoding="utf-8") as handle:
        for compact in iter_jsonl(args.labels):
            query_id = str(compact.get("query_id") or "")
            query = canonical.get(query_id)
            if query is None:
                missing_queries.append(query_id)
                continue

            image_path = project_root / str(compact["image_key"])
            if not image_path.is_file():
                missing_images.append(str(compact["image_key"]))
                if not args.allow_missing_images:
                    continue

            materialized = {
                "query_id": query_id,
                "benchmark": compact["benchmark"],
                "task": compact["task"],
                "language": compact["language"],
                "question": query.get("question", ""),
                "prompt": query.get("prompt", query.get("question", "")),
                "image_path": image_path.as_posix(),
                "candidate_labels": compact["candidate_labels"],
            }
            if not str(materialized["question"]).strip():
                raise ValueError(f"{query_id}: reconstructed query has no question text")
            handle.write(json.dumps(materialized, ensure_ascii=False) + "\n")
            written += 1

    if missing_queries:
        args.output.unlink(missing_ok=True)
        sample = ", ".join(missing_queries[:5])
        raise ValueError(f"{len(missing_queries)} labels lack reconstructed queries; examples: {sample}")
    if missing_images and not args.allow_missing_images:
        args.output.unlink(missing_ok=True)
        sample = ", ".join(missing_images[:5])
        raise ValueError(f"{len(missing_images)} reconstructed images are missing; examples: {sample}")

    print(
        json.dumps(
            {
                "written": written,
                "missing_queries": len(missing_queries),
                "missing_images": len(missing_images),
                "output": str(args.output),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
