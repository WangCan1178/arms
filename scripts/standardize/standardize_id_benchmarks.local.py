#!/usr/bin/env python3
"""Standardize ALM-Bench, MMT-Bench, and MME-RealWorld-Base64 into M2 query JSONL."""

from __future__ import annotations

import argparse
import ast
import base64
import csv
import io
import json
import re
import sys
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

from PIL import Image
import pyarrow.parquet as pq
from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = PROJECT_ROOT / "data"
RAW_ROOT = DATA_ROOT / "raw"
PROCESSED_ROOT = DATA_ROOT / "processed"
DEFAULT_OUTPUT = PROCESSED_ROOT / "id_queries_standardized.jsonl"
DEFAULT_IMAGES_ROOT = PROCESSED_ROOT / "images"
DEFAULT_MMT_FILES = ("MMT-Bench_ALL.tsv",)
DEFAULT_MMT_ANSWER_FILES = ("MMT-Bench_ALL_MI_anno.tsv",)

MC_INSTRUCTION = "Answer the multiple choice question by just giving the letter of the correct answer."
LETTERS = "ABCDEFGHI"


def sanitize(value: Any) -> str:
    text = str(value or "unknown")
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", text).strip("_")
    return text[:160] or "unknown"


def ensure_rgb(image: Image.Image) -> Image.Image:
    if image.mode in {"RGB", "L"}:
        return image.convert("RGB")
    return image.convert("RGB")


def save_image(image: Image.Image, path: Path, overwrite: bool = False) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and not overwrite:
        return path
    ensure_rgb(image).save(path, format="JPEG", quality=95)
    return path


def decode_base64_image(value: str | bytes) -> Image.Image:
    if isinstance(value, bytes):
        raw = value
    else:
        text = value.strip()
        if text.startswith("data:image") and "," in text:
            text = text.split(",", 1)[1]
        text = re.sub(r"\s+", "", text)
        if len(text) % 4:
            text += "=" * (4 - len(text) % 4)
        raw = base64.b64decode(text, validate=False)
    image = Image.open(io.BytesIO(raw))
    image.load()
    return ensure_rgb(image)


def write_jsonl(path: Path, rows: Iterable[Dict[str, Any]]) -> Counter:
    path.parent.mkdir(parents=True, exist_ok=True)
    stats: Counter = Counter()
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            stats["rows"] += 1
            stats[row.get("benchmark", "unknown")] += 1
            stats[f"answer_type:{row.get('answer_type', 'unknown')}"] += 1
    return stats


def normalize_choices_from_text(answer: str) -> Tuple[str, List[Dict[str, str]], str]:
    match = re.match(r"^(.*?)\s*\(Options:\s*(.*?)\)\s*$", answer or "", flags=re.IGNORECASE | re.DOTALL)
    if not match:
        return answer or "", [], "free_form"

    answer_text = match.group(1).strip()
    raw_options = [item.strip() for item in match.group(2).split(",") if item.strip()]
    choices = [{"letter": LETTERS[i], "text": option} for i, option in enumerate(raw_options[: len(LETTERS)])]
    ground_truth = answer_text
    for item in choices:
        if item["text"].strip().lower() == answer_text.lower():
            ground_truth = item["letter"]
            break
    return ground_truth, choices, "multiple_choice"


def choices_from_mmt(row: Dict[str, Any]) -> List[Dict[str, str]]:
    choices = []
    for letter in LETTERS:
        value = row.get(letter)
        if value is None:
            continue
        value = str(value).strip()
        if value and value.lower() != "nan":
            choices.append({"letter": letter, "text": value})
    return choices


def load_mmt_answer_map(dataset_path: Path, answer_files: Sequence[str]) -> Dict[str, Tuple[str, str]]:
    answer_map: Dict[str, Tuple[str, str]] = {}
    csv.field_size_limit(sys.maxsize)
    for filename in answer_files:
        path = dataset_path / filename
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            if "index" not in (reader.fieldnames or []) or "answer" not in (reader.fieldnames or []):
                continue
            for row in reader:
                raw_id = str(row.get("index") or "").strip()
                answer = str(row.get("answer") or "").strip().upper()
                if raw_id and answer:
                    answer_map[raw_id] = (answer, filename)
    return answer_map


def choices_from_mme(value: Any) -> List[Dict[str, str]]:
    if isinstance(value, str):
        try:
            parsed = ast.literal_eval(value)
        except (SyntaxError, ValueError):
            parsed = [item.strip() for item in value.split("\n") if item.strip()]
    else:
        parsed = value
    if not isinstance(parsed, list):
        return []

    choices = []
    for idx, item in enumerate(parsed[: len(LETTERS)]):
        text = str(item).strip()
        match = re.match(r"^\(?([A-I])\)?[.)]?\s*(.*)$", text)
        if match:
            letter = match.group(1)
            body = match.group(2).strip() or text
        else:
            letter = LETTERS[idx]
            body = text
        choices.append({"letter": letter, "text": body})
    return choices


def format_prompt(question: str, choices: Sequence[Dict[str, str]], answer_type: str) -> str:
    question = (question or "").strip()
    if answer_type != "multiple_choice" or not choices:
        return question
    option_lines = [f"({item['letter']}) {item['text']}" for item in choices]
    return "\n".join([question, "", *option_lines, "", MC_INSTRUCTION])


def base_row(
    *,
    query_id: str,
    benchmark: str,
    task: str,
    language: str,
    image_path: Path,
    question: str,
    ground_truth: str,
    choices: Sequence[Dict[str, str]],
    answer_type: str,
    source: Dict[str, Any],
) -> Dict[str, Any]:
    prompt = format_prompt(question, choices, answer_type)
    return {
        "query_id": query_id,
        "benchmark": benchmark,
        "task": task,
        "language": language,
        "image_path": str(image_path),
        "question": question,
        "prompt": prompt,
        "ground_truth": ground_truth,
        "choices": list(choices),
        "answer_type": answer_type,
        "source": source,
    }


def iter_alm(raw_root: Path, images_root: Path, overwrite_images: bool) -> Iterator[Dict[str, Any]]:
    dataset_path = raw_root / "ALM-Bench"
    parquet_files = sorted((dataset_path / "data").glob("*.parquet"))
    for parquet_file in parquet_files:
        parquet = pq.ParquetFile(parquet_file)
        for batch in parquet.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                yield alm_row_to_standardized(row, images_root, overwrite_images)


def alm_row_to_standardized(row: Dict[str, Any], images_root: Path, overwrite_images: bool) -> Dict[str, Any]:
        raw_id = row.get("ID")
        language = str(row.get("Language") or "unknown")
        question = str(row.get("Translated_Question") or row.get("English_Question") or "")
        answer = str(row.get("Translated_Answer") or row.get("English_Answer") or "")
        ground_truth, choices, answer_type = normalize_choices_from_text(answer)
        image_struct = row.get("file_name") or {}
        image_bytes = image_struct.get("bytes") if isinstance(image_struct, dict) else None
        if not image_bytes:
            raise ValueError(f"ALM-Bench row {raw_id} has no image bytes in file_name")
        image = Image.open(io.BytesIO(image_bytes))
        image.load()
        image_path = save_image(image, images_root / "alm_bench" / f"{sanitize(raw_id)}.jpg", overwrite=overwrite_images)
        return base_row(
            query_id=f"alm:{raw_id}",
            benchmark="ALM-Bench",
            task=str(row.get("Category") or "unknown"),
            language=language,
            image_path=image_path,
            question=question,
            ground_truth=str(ground_truth),
            choices=choices,
            answer_type=answer_type,
            source={"id": raw_id, "question_type": row.get("Question_Type"), "image_url": row.get("Image_Url")},
        )


def iter_mmt(
    raw_root: Path,
    images_root: Path,
    overwrite_images: bool,
    mmt_files: Sequence[str] = DEFAULT_MMT_FILES,
    mmt_answer_files: Sequence[str] = DEFAULT_MMT_ANSWER_FILES,
) -> Iterator[Dict[str, Any]]:
    csv.field_size_limit(sys.maxsize)
    dataset_path = raw_root / "MMT-Bench"
    annotation_answers = load_mmt_answer_map(dataset_path, mmt_answer_files)
    for filename in mmt_files:
        tsv_path = dataset_path / filename
        if not tsv_path.exists():
            continue
        with tsv_path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                raw_id = row.get("index")
                choices = choices_from_mmt(row)
                answer = str(row.get("answer") or "").strip().upper()
                answer_source = filename if answer else ""
                if not answer:
                    answer, answer_source = annotation_answers.get(str(raw_id), ("", ""))
                answer_type = "multiple_choice" if choices else "free_form"
                image = decode_base64_image(row.get("image") or "")
                image_path = save_image(
                    image,
                    images_root / "mmt_bench" / sanitize(Path(filename).stem) / f"{sanitize(raw_id)}.jpg",
                    overwrite=overwrite_images,
                )
                yield base_row(
                    query_id=f"mmt:{Path(filename).stem}:{raw_id}",
                    benchmark="MMT-Bench",
                    task=str(row.get("l2-category") or row.get("category") or "unknown"),
                    language="unknown",
                    image_path=image_path,
                    question=str(row.get("question") or ""),
                    ground_truth=answer,
                    choices=choices,
                    answer_type=answer_type,
                    source={
                        "file": filename,
                        "index": raw_id,
                        "category": row.get("category"),
                        "split": row.get("split"),
                        "answer_source": answer_source,
                    },
                )


def iter_json_array(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[Any]:
    decoder = json.JSONDecoder()
    buffer = ""
    pos = 0
    in_array = False
    with path.open("r", encoding="utf-8") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk and pos >= len(buffer):
                return
            buffer += chunk
            while True:
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if not in_array:
                    if pos < len(buffer) and buffer[pos] == "[":
                        in_array = True
                        pos += 1
                    elif pos < len(buffer):
                        obj, _ = decoder.raw_decode(buffer[pos:])
                        yield obj
                        return
                    else:
                        break
                while pos < len(buffer) and buffer[pos].isspace():
                    pos += 1
                if pos < len(buffer) and buffer[pos] == ",":
                    pos += 1
                    continue
                if pos < len(buffer) and buffer[pos] == "]":
                    return
                try:
                    obj, rel_end = decoder.raw_decode(buffer[pos:])
                except json.JSONDecodeError:
                    if not chunk:
                        raise
                    break
                yield obj
                pos += rel_end
                if pos > 8 * chunk_size:
                    buffer = buffer[pos:]
                    pos = 0


def iter_mme(raw_root: Path, images_root: Path, overwrite_images: bool) -> Iterator[Dict[str, Any]]:
    dataset_path = raw_root / "MME-RealWorld-Base64"
    json_files = sorted(
        path
        for path in dataset_path.rglob("*.json")
        if ".cache" not in path.parts and path.name != "dataset_info.json"
    )
    for json_path in json_files:
        relative_parent = json_path.parent.relative_to(dataset_path)
        language = "Chinese" if "CN" in str(relative_parent) else "English"
        for row in iter_json_array(json_path):
            if not isinstance(row, dict):
                continue
            raw_id = row.get("index")
            choices = choices_from_mme(row.get("multi-choice options"))
            answer = str(row.get("answer") or "").strip().upper()
            image = decode_base64_image(row.get("image") or "")
            image_path = save_image(
                image,
                images_root
                / "mme_realworld"
                / sanitize(relative_parent)
                / sanitize(json_path.stem)
                / f"{sanitize(raw_id)}.jpg",
                overwrite=overwrite_images,
            )
            yield base_row(
                query_id=f"mme:{relative_parent}:{json_path.stem}:{raw_id}",
                benchmark="MME-RealWorld-Base64",
                task=str(row.get("l2-category") or row.get("category") or "unknown"),
                language=language,
                image_path=image_path,
                question=str(row.get("question") or ""),
                ground_truth=answer,
                choices=choices,
                answer_type="multiple_choice" if choices else "free_form",
                source={"file": str(json_path.relative_to(dataset_path)), "index": raw_id, "category": row.get("category")},
            )


def limited(iterable: Iterable[Dict[str, Any]], limit: Optional[int]) -> Iterator[Dict[str, Any]]:
    if limit is None:
        yield from iterable
    else:
        yield from islice(iterable, limit)


def iter_all(args: argparse.Namespace) -> Iterator[Dict[str, Any]]:
    raw_root = args.data_root / "raw"
    images_root = args.images_dir
    remaining = args.limit
    for name, producer in [
        ("ALM-Bench", iter_alm),
        ("MMT-Bench", iter_mmt),
        ("MME-RealWorld-Base64", iter_mme),
    ]:
        if args.benchmarks and name not in args.benchmarks:
            continue
        if name == "MMT-Bench":
            stream = producer(raw_root, images_root, args.overwrite_images, args.mmt_files, args.mmt_answer_files)
        else:
            stream = producer(raw_root, images_root, args.overwrite_images)
        for row in tqdm(stream, desc=f"standardizing {name}", unit="query"):
            if remaining is not None and remaining <= 0:
                return
            yield row
            if remaining is not None:
                remaining -= 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--images-dir", type=Path, default=DEFAULT_IMAGES_ROOT)
    parser.add_argument("--limit", type=int, default=None, help="Write only the first N standardized queries.")
    parser.add_argument("--benchmarks", nargs="*", choices=["ALM-Bench", "MMT-Bench", "MME-RealWorld-Base64"])
    parser.add_argument(
        "--mmt-files",
        nargs="+",
        default=list(DEFAULT_MMT_FILES),
        help=(
            "MMT-Bench TSV files to include. Defaults to MMT-Bench_ALL.tsv, the full combined-image split. "
            "Use *_MI only for VLM pipelines that support multi-image inputs."
        ),
    )
    parser.add_argument(
        "--mmt-answer-files",
        nargs="+",
        default=list(DEFAULT_MMT_ANSWER_FILES),
        help="Optional MMT-Bench annotation TSV files used to fill missing answer labels in full ALL splits.",
    )
    parser.add_argument("--overwrite-images", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    stats = write_jsonl(args.output, iter_all(args))
    print(f"Wrote {stats['rows']} standardized ID queries to {args.output}")
    for key, value in sorted(stats.items()):
        if key != "rows":
            print(f"  {key}: {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
