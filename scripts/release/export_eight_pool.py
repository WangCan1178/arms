#!/usr/bin/env python3
"""Export eight-model routing supervision without modifying source JSONL files.

Each query row keeps the question, local image path, ground-truth answer,
benchmark metadata, the original top-level label, and the eight candidate
labels. Pair rows follow the ``JudgedRecord`` fields used by ``train.py``,
with ``image_path`` instead of an embedded image. Call ``attach_image`` to
open that path into the ``image`` field.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, TextIO

from export_public_dataset import PAPER_OOD_TASKS


ORIGINAL_MODELS = (
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "HuggingFaceM4/idefics2-8b",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
)
NEW_MODELS = (
    "kimi-k3",
    "deepseek-v4.1-flash",
    "qwen3.8-max",
    "glm-5.3-flash",
)
ALL_MODELS = ORIGINAL_MODELS + NEW_MODELS

REMOTE_TO_LOCAL = (
    (
        "/root/autodl-tmp/m2-free-datasets/data/processed/images/",
        "dataset/result/images/",
    ),
    (
        "/root/autodl-tmp/m2-free-datasets/data/raw/VHELM/",
        "dataset/result/vhelm/",
    ),
)

QUERY_FIELDS = (
    "query_id",
    "benchmark",
    "task",
    "language",
    "answer_type",
    "question",
    "prompt",
    "ground_truth",
    "choices",
    "image_path",
    "original_pool_label",
    "candidate_labels",
    "parsed_answers",
)
JUDGED_FIELDS = (
    "record_id",
    "source_dataset",
    "source_id",
    "lang",
    "question",
    "answer",
    "image_path",
    "model_name",
    "model_answer",
    "correct",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
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


def local_image_path(raw_path: Any) -> str:
    text = str(raw_path or "").replace("\\", "/").strip()
    if not text:
        raise ValueError("empty image_path")
    for remote, local in REMOTE_TO_LOCAL:
        if text.startswith(remote):
            return local + text[len(remote) :]
    if text.startswith("dataset/result/images/") or text.startswith("dataset/result/vhelm/"):
        return text
    raise ValueError(f"unrecognized image_path: {text}")


def dump_line(handle: TextIO, row: Dict[str, Any]) -> None:
    handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_ids(path: Path, query_ids: Iterable[str]) -> None:
    path.write_text("".join(f"{query_id}\n" for query_id in query_ids), encoding="utf-8")


def export_split(
    source: Path,
    query_output: Path,
    judged_output: Path,
    project_root: Path,
    paper_tasks: bool,
) -> Dict[str, Any]:
    seen: set[str] = set()
    paper_ids: list[str] = []
    answerable_ids: list[str] = []
    paper_answerable_ids: list[str] = []
    task_counts: Counter[str] = Counter()
    benchmark_counts: Counter[str] = Counter()
    blank_by_model: Counter[str] = Counter()
    error_by_model: Counter[str] = Counter()
    original_answerable = 0
    label_mismatches = 0
    missing_image_count = 0
    missing_images: list[str] = []
    query_count = 0
    judged_count = 0

    with query_output.open("w", encoding="utf-8") as query_handle, gzip.open(
        judged_output, "wt", encoding="utf-8"
    ) as judged_handle:
        for source_row in iter_jsonl(source):
            query_id = str(source_row.get("query_id") or "").strip()
            if not query_id:
                raise ValueError(f"{source}: row without query_id")
            if query_id in seen:
                raise ValueError(f"{source}: duplicate query_id: {query_id}")
            seen.add(query_id)

            outputs = source_row.get("model_outputs")
            if not isinstance(outputs, dict):
                raise ValueError(f"{query_id}: model_outputs must be an object")
            missing = [name for name in ALL_MODELS if name not in outputs]
            if missing:
                raise ValueError(f"{query_id}: missing models: {missing}")

            labels: Dict[str, bool] = {}
            parsed: Dict[str, str] = {}
            for model_name in ALL_MODELS:
                output = outputs[model_name]
                if not isinstance(output, dict):
                    raise ValueError(f"{query_id}: malformed output for {model_name}")
                if str(output.get("vlm_output") or "").strip() == "":
                    blank_by_model[model_name] += 1
                if str(output.get("error") or "").strip():
                    error_by_model[model_name] += 1
                label = output.get("label")
                if not isinstance(label, bool):
                    raise ValueError(f"{query_id}: non-boolean label for {model_name}")
                labels[model_name] = label
                parsed[model_name] = str(output.get("parsed_answer") or "")

            original_label = source_row.get("label")
            if not isinstance(original_label, bool):
                raise ValueError(f"{query_id}: top-level label is not boolean")
            if original_label != any(labels[name] for name in ORIGINAL_MODELS):
                label_mismatches += 1
            if original_label:
                original_answerable += 1

            image_path = local_image_path(source_row.get("image_path"))
            if not (project_root / image_path).is_file():
                missing_image_count += 1
                if len(missing_images) < 20:
                    missing_images.append(image_path)

            task = str(source_row.get("task") or "")
            benchmark = str(source_row.get("benchmark") or "")
            task_counts[task] += 1
            benchmark_counts[benchmark] += 1
            answerable = any(labels.values())
            in_paper = task in PAPER_OOD_TASKS
            if paper_tasks and in_paper:
                paper_ids.append(query_id)
            if answerable:
                answerable_ids.append(query_id)
            if paper_tasks and in_paper and answerable:
                paper_answerable_ids.append(query_id)

            query_row = {
                "query_id": query_id,
                "benchmark": benchmark,
                "task": task,
                "language": str(source_row.get("language") or ""),
                "answer_type": str(source_row.get("answer_type") or ""),
                "question": source_row.get("question", ""),
                "prompt": source_row.get("prompt", ""),
                "ground_truth": source_row.get("ground_truth", ""),
                "choices": source_row.get("choices", []),
                "image_path": image_path,
                "original_pool_label": original_label,
                "candidate_labels": labels,
                "parsed_answers": parsed,
            }
            dump_line(query_handle, query_row)
            query_count += 1

            for model_name in ALL_MODELS:
                judged_row = {
                    "record_id": query_id,
                    "source_dataset": benchmark,
                    "source_id": query_id,
                    "lang": query_row["language"],
                    "question": query_row["question"],
                    "answer": query_row["ground_truth"],
                    "image_path": image_path,
                    "model_name": model_name,
                    "model_answer": parsed[model_name],
                    "correct": labels[model_name],
                }
                dump_line(judged_handle, judged_row)
                judged_count += 1

    return {
        "queries": query_count,
        "judged_rows": judged_count,
        "blank_vlm_outputs": dict(blank_by_model),
        "outputs_with_error": dict(error_by_model),
        "original_pool_answerable": original_answerable,
        "original_label_mismatches": label_mismatches,
        "missing_image_count": missing_image_count,
        "missing_images": missing_images,
        "paper_ids": paper_ids,
        "answerable_ids": answerable_ids,
        "paper_answerable_ids": paper_answerable_ids,
        "benchmarks": dict(sorted(benchmark_counts.items())),
        "tasks": dict(sorted(task_counts.items())),
    }


def attach_image(row: Dict[str, Any], project_root: Path):
    """Return a ``JudgedRecord`` by opening ``image_path``."""
    from PIL import Image

    image_path = project_root / str(row["image_path"])
    record = dict(row)
    record["image"] = Image.open(image_path)
    return record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--result-dir",
        type=Path,
        default=Path("dataset/result"),
        help="Directory containing m2_*_free_*.jsonl source files.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/supervision/eight_pool"),
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path("."),
        help="Root used to resolve dataset/result image paths.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result_dir = args.result_dir.resolve()
    output_dir = args.output_dir.resolve()
    project_root = args.project_root.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    sources = {
        "id_train": result_dir / "m2_id_free_train.jsonl",
        "id_test": result_dir / "m2_id_free_test.jsonl",
        "ood_test": result_dir / "m2_ood_free_test.jsonl",
    }
    for path in sources.values():
        if not path.is_file():
            raise ValueError(f"missing source file: {path}")
    source_hashes = {name: sha256(path) for name, path in sources.items()}

    summaries: Dict[str, Any] = {}
    for name, source in sources.items():
        summary = export_split(
            source=source,
            query_output=output_dir / f"{name}.jsonl",
            judged_output=output_dir / f"{name}_judged.jsonl.gz",
            project_root=project_root,
            paper_tasks=name == "ood_test",
        )
        paper_ids = summary.pop("paper_ids")
        answerable_ids = summary.pop("answerable_ids")
        paper_answerable_ids = summary.pop("paper_answerable_ids")
        missing_images = summary.pop("missing_images")
        missing_image_count = summary.pop("missing_image_count")
        write_ids(output_dir / f"{name}_eight_answerable_query_ids.txt", answerable_ids)
        summary["eight_answerable"] = len(answerable_ids)
        if name == "ood_test":
            write_ids(output_dir / "ood_paper_query_ids.txt", paper_ids)
            write_ids(
                output_dir / "ood_paper_eight_answerable_query_ids.txt",
                paper_answerable_ids,
            )
            summary["paper_ood"] = len(paper_ids)
            summary["paper_ood_eight_answerable"] = len(paper_answerable_ids)
        summary["missing_image_count"] = missing_image_count
        summary["missing_image_examples"] = missing_images
        summaries[name] = summary

    if sha256(sources["id_train"]) != source_hashes["id_train"]:
        raise ValueError("source id train changed during export")
    if sha256(sources["id_test"]) != source_hashes["id_test"]:
        raise ValueError("source id test changed during export")
    if sha256(sources["ood_test"]) != source_hashes["ood_test"]:
        raise ValueError("source ood test changed during export")

    problems: list[str] = []
    expected_queries = {"id_train": 29_626, "id_test": 3_000, "ood_test": 8_626}
    for name, expected in expected_queries.items():
        summary = summaries[name]
        if summary["queries"] != expected:
            problems.append(f"{name}: expected {expected} queries, found {summary['queries']}")
        if summary["judged_rows"] != expected * len(ALL_MODELS):
            problems.append(f"{name}: judged row count {summary['judged_rows']}")
        if summary["original_label_mismatches"] != 0:
            problems.append(
                f"{name}: top-level label mismatches {summary['original_label_mismatches']}"
            )
        if summary["missing_image_count"] != 0:
            problems.append(f"{name}: missing images {summary['missing_image_count']}")
    paper_count = summaries["ood_test"]["paper_ood"]
    if paper_count != 7_084:
        problems.append(f"paper OOD subset: expected 7084, found {paper_count}")

    manifest = {
        "schema_version": "1.0",
        "candidate_order": list(ALL_MODELS),
        "original_models": list(ORIGINAL_MODELS),
        "new_models": list(NEW_MODELS),
        "query_fields": list(QUERY_FIELDS),
        "judged_fields": list(JUDGED_FIELDS),
        "judged_image": "image_path is repo-relative; attach_image opens it as JudgedRecord.image",
        "original_pool_label": "copied from the source top-level label and not recomputed",
        "source_sha256": source_hashes,
        "splits": summaries,
    }
    (output_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({"problems": problems, "splits": summaries}, ensure_ascii=False, indent=2))
    if problems:
        raise SystemExit(1)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
