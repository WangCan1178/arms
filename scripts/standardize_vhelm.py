#!/usr/bin/env python3
"""Standardize materialized VHELM/HELM instances into M2 query JSONL.

The HELM source tree and VHELM run entries are not themselves the query set.
This script converts already-materialized HELM/VHELM instance files when they
exist, and otherwise reports the exact missing restoration step.
"""

from __future__ import annotations

import argparse
import json
import random
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import requests

from tqdm import tqdm


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
VHELM_ROOT = DATA_ROOT / "raw" / "VHELM"
DEFAULT_OUTPUT = DATA_ROOT / "processed" / "vhelm_10k_queries.jsonl"
MC_INSTRUCTION = "Answer the multiple choice question by just giving the letter of the correct answer."
DEFAULT_GCS_BASE_URL = "https://storage.googleapis.com/crfm-helm-public/vhelm/"


def read_json_records(path: Path) -> Iterator[Dict[str, Any]]:
    if path.suffix == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    obj = json.loads(line)
                    if isinstance(obj, dict):
                        yield obj
        return
    if path.suffix == ".json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(obj, list):
            for item in obj:
                if isinstance(item, dict):
                    yield item
        elif isinstance(obj, dict):
            request_states = obj.get("request_states")
            if isinstance(request_states, list):
                for item in request_states:
                    if isinstance(item, dict):
                        instance = item.get("instance")
                        if isinstance(instance, dict):
                            merged = dict(instance)
                            merged["_request_state"] = item
                            yield merged
                        else:
                            yield item
                return
            for key in ["instances", "requests", "examples", "data"]:
                value = obj.get(key)
                if isinstance(value, list):
                    for item in value:
                        if isinstance(item, dict):
                            yield item
                    return
            yield obj


def candidate_instance_files(vhelm_root: Path, extra_globs: List[str]) -> List[Path]:
    patterns = [
        "materialized/**/*.jsonl",
        "materialized/**/*.json",
        "benchmark_output/instances/**/input_instances.json",
        "benchmark_output/runs/**/scenario_state.json",
        "benchmark_output/**/*instances*.jsonl",
        "benchmark_output/**/*instances*.json",
        "benchmark_output/**/*requests*.jsonl",
        "benchmark_output/**/*requests*.json",
        "instances/**/*.jsonl",
        "instances/**/*.json",
        "queries/**/*.jsonl",
        "queries/**/*.json",
        *extra_globs,
    ]
    files: List[Path] = []
    for pattern in patterns:
        files.extend(path for path in vhelm_root.glob(pattern) if path.is_file())
    ignored = {"schema_vhelm.yaml", "package.json", "package-lock.json"}
    return sorted({path for path in files if path.name not in ignored and ".git" not in path.parts})


def first_present(record: Dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in record and record[key] not in (None, ""):
            return record[key]
    return None


def normalize_choices(value: Any) -> List[Dict[str, str]]:
    if not value:
        return []
    if isinstance(value, dict):
        return [{"letter": str(k), "text": str(v)} for k, v in value.items()]
    if isinstance(value, list):
        choices = []
        letters = "ABCDEFGHI"
        for idx, item in enumerate(value[: len(letters)]):
            if isinstance(item, dict):
                letter = str(item.get("letter") or item.get("label") or letters[idx])
                text = str(item.get("text") or item.get("value") or item)
            else:
                letter = letters[idx]
                text = str(item)
            choices.append({"letter": letter, "text": text})
        return choices
    return []


def parse_run_metadata(source_path: Path, root: Path) -> Dict[str, str]:
    try:
        rel = source_path.relative_to(root)
    except ValueError:
        rel = source_path
    run_name = ""
    parts = list(rel.parts)
    if "runs" in parts:
        run_idx = parts.index("runs")
        if len(parts) > run_idx + 2:
            run_name = parts[run_idx + 2]
    elif len(parts) > 1:
        run_name = parts[-2]

    metadata: Dict[str, str] = {}
    if run_name:
        metadata["run_name"] = run_name
        metadata["task"] = run_name.split(":", 1)[0]
        for item in run_name.replace(":", ",").split(","):
            if "=" in item:
                key, value = item.split("=", 1)
                metadata[key] = value
        augmentation = metadata.get("data_augmentation", "")
        if augmentation in {"chinese", "hindi", "spanish", "swahili"}:
            metadata.setdefault("language", augmentation)
        elif augmentation.startswith("dialect_"):
            metadata.setdefault("language", "english")
    return metadata


def resolve_media_path(image_path: str, source_path: Path, root: Path) -> str:
    if image_path.startswith(("http://", "https://")):
        return image_path
    candidate = Path(image_path)
    if candidate.is_absolute():
        return str(candidate)

    local_candidates = [
        (source_path.parent / candidate).resolve(),
        (root / candidate).resolve(),
    ]
    for local_candidate in local_candidates:
        if local_candidate.exists():
            return str(local_candidate)
    if image_path.startswith("benchmark_output/"):
        return str((root / candidate).resolve())
    return image_path


def media_from_helm_input(value: Any, source_path: Path, root: Path) -> Tuple[str, List[str]]:
    if not isinstance(value, dict):
        return "", []
    media_objects = (
        value.get("multimedia_content", {}).get("media_objects")
        if isinstance(value.get("multimedia_content"), dict)
        else None
    )
    if not isinstance(media_objects, list):
        media_objects = value.get("media_objects")
    if not isinstance(media_objects, list):
        return "", []

    text_parts: List[str] = []
    image_paths: List[str] = []
    for media_object in media_objects:
        if not isinstance(media_object, dict):
            continue
        content_type = str(media_object.get("content_type") or media_object.get("mime_type") or "")
        if content_type.startswith("text/") or "text" in media_object:
            text = media_object.get("text")
            if text is not None:
                text_parts.append(str(text))
            continue
        if content_type.startswith("image/") or "location" in media_object:
            location = media_object.get("location") or media_object.get("path") or media_object.get("url")
            if location:
                image_paths.append(resolve_media_path(str(location), source_path, root))
    return "\n".join(part.strip() for part in text_parts if part and part.strip()), image_paths


def references_from_helm(value: Any) -> Tuple[str, List[Dict[str, str]], str]:
    if not isinstance(value, list):
        return "", [], "free_form"

    choices: List[Dict[str, str]] = []
    correct: List[str] = []
    letters = "ABCDEFGHI"
    for idx, reference in enumerate(value[: len(letters)]):
        if not isinstance(reference, dict):
            continue
        output = reference.get("output")
        text = ""
        if isinstance(output, dict):
            text = str(output.get("text") or output.get("value") or "")
        elif output is not None:
            text = str(output)
        tags = reference.get("tags") or []
        tags = [str(tag).lower() for tag in tags] if isinstance(tags, list) else [str(tags).lower()]
        letter = letters[idx]
        if text:
            choices.append({"letter": letter, "text": text})
        if "correct" in tags:
            correct.append(letter if len(value) > 1 else text)

    if correct:
        answer_type = "multiple_choice" if len(choices) > 1 else "free_form"
        return correct[0], choices if answer_type == "multiple_choice" else [], answer_type
    if len(choices) == 1:
        return choices[0]["text"], [], "free_form"
    return "", choices, "multiple_choice" if choices else "free_form"


def normalize_helm_instance(record: Dict[str, Any], source_path: Path, root: Path, idx: int) -> Optional[Dict[str, Any]]:
    input_value = record.get("input")
    question, images = media_from_helm_input(input_value, source_path, root)
    answer, choices, answer_type = references_from_helm(record.get("references"))
    if not question or answer == "":
        return None

    prompt = question
    if answer_type == "multiple_choice" and choices:
        prompt = "\n".join([prompt, "", *[f"({c['letter']}) {c['text']}" for c in choices], "", MC_INSTRUCTION])

    rel = source_path.relative_to(root)
    request_state = record.get("_request_state") if isinstance(record.get("_request_state"), dict) else {}
    metadata = parse_run_metadata(source_path, root)
    return {
        "query_id": f"vhelm:{rel}:{record.get('id', request_state.get('request_mode', 'row'))}:{idx}",
        "benchmark": "VHELM",
        "task": str(first_present(record, ["scenario", "task", "category"]) or metadata.get("groups") or metadata.get("task") or first_present(request_state, ["scenario"]) or "unknown"),
        "language": str(first_present(record, ["language", "locale"]) or metadata.get("language") or metadata.get("location") or "unknown"),
        "image_path": images[0] if images else "",
        "question": question,
        "prompt": prompt,
        "ground_truth": str(answer),
        "choices": choices,
        "answer_type": answer_type,
        "source": {
            "file": str(rel),
            "raw_id": record.get("id"),
            "image_paths": images,
            "split": record.get("split"),
            "run_metadata": metadata,
        },
    }


def normalize_record(record: Dict[str, Any], source_path: Path, root: Path, idx: int) -> Optional[Dict[str, Any]]:
    if isinstance(record.get("input"), dict) and (
        "multimedia_content" in record["input"] or "media_objects" in record["input"]
    ):
        return normalize_helm_instance(record, source_path, root, idx)

    question = first_present(record, ["question", "query", "prompt", "input", "text"])
    answer = first_present(record, ["ground_truth", "answer", "gold", "target", "correct_answer", "reference"])
    image = first_present(record, ["image_path", "local_image_path", "image", "image_file", "image_url"])
    if not question or answer is None:
        return None

    image_path = ""
    if image:
        image_path = resolve_media_path(str(image), source_path, root)

    choices = normalize_choices(first_present(record, ["choices", "options", "multi-choice options"]))
    answer_type = "multiple_choice" if choices else "free_form"
    prompt = str(question)
    if answer_type == "multiple_choice":
        prompt = "\n".join([prompt, "", *[f"({c['letter']}) {c['text']}" for c in choices], "", MC_INSTRUCTION])

    rel = source_path.relative_to(root)
    metadata = parse_run_metadata(source_path, root)
    return {
        "query_id": f"vhelm:{rel}:{record.get('id', record.get('index', 'row'))}:{idx}",
        "benchmark": "VHELM",
        "task": str(first_present(record, ["scenario", "task", "category", "run_group"]) or metadata.get("groups") or metadata.get("task") or "unknown"),
        "language": str(first_present(record, ["language", "locale"]) or metadata.get("language") or metadata.get("location") or "unknown"),
        "image_path": image_path,
        "question": str(question),
        "prompt": prompt,
        "ground_truth": str(answer),
        "choices": choices,
        "answer_type": answer_type,
        "source": {"file": str(rel), "raw_id": record.get("id", record.get("index")), "run_metadata": metadata},
    }


def download_image_if_needed(image_path: str, root: Path, base_url: str, timeout: int) -> bool:
    if not image_path or image_path.startswith(("http://", "https://")):
        return False
    path = Path(image_path)
    if path.exists():
        return True
    try:
        rel = path.resolve().relative_to(root.resolve())
    except ValueError:
        return False
    url = base_url.rstrip("/") + "/" + str(rel).replace("\\", "/")
    path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    path.write_bytes(response.content)
    return True


def write_missing_status(vhelm_root: Path, output: Path, candidates: List[Path]) -> None:
    status_path = output.parent / "vhelm_standardization_status.md"
    run_entries = vhelm_root / "helm" / "src" / "helm" / "benchmark" / "presentation" / "run_entries_vhelm.conf"
    schema = vhelm_root / "schema_vhelm.yaml"
    docs = vhelm_root / "helm" / "docs" / "vhelm.md"
    lines = [
        "# VHELM Standardization Status",
        "",
        "No materialized VHELM query/instance files were found, so `vhelm_10k_queries.jsonl` was not generated.",
        "",
        "Prepared resources:",
        f"- Schema: `{schema}` exists={schema.exists()}",
        f"- Run entries: `{run_entries}` exists={run_entries.exists()}",
        f"- Docs: `{docs}` exists={docs.exists()}",
        "",
        "Expected next step:",
        "- Use HELM/VHELM scenario code to materialize query, image, and ground-truth instances.",
        "- Recommended official path after installing HELM dependencies:",
        "",
        "```bash",
        "cd data/raw/VHELM/helm",
        "helm-run \\",
        "  --conf-paths src/helm/benchmark/presentation/run_entries_vhelm.conf \\",
        "  --suite vhelm-materialize \\",
        "  --max-eval-instances 1 \\",
        "  --cache-instances \\",
        "  --cache-instances-only",
        "```",
        "",
        "- This script now scans `benchmark_output/instances/**/input_instances.json` and `benchmark_output/runs/**/scenario_state.json`.",
        "- You can also place materialized files under `data/raw/VHELM/materialized/` or pass `--source-glob`.",
        "- Do not use commercial model credentials or run model inference during materialization.",
        "",
        f"Candidate files scanned: {len(candidates)}",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    status_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote status: {status_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vhelm-root", type=Path, default=VHELM_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--sample-size", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source-glob", action="append", default=[], help="Additional glob under VHELM root containing materialized instances.")
    parser.add_argument("--download-images", action="store_true", help="Download sampled GCS images referenced by HELM instances.")
    parser.add_argument("--image-base-url", default=DEFAULT_GCS_BASE_URL)
    parser.add_argument("--image-timeout", type=int, default=60)
    parser.add_argument("--image-workers", type=int, default=16)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    candidates = candidate_instance_files(args.vhelm_root, args.source_glob)
    rows: List[Dict[str, Any]] = []
    for path in tqdm(candidates, desc="reading VHELM candidate files", unit="file"):
        for idx, record in enumerate(read_json_records(path)):
            row = normalize_record(record, path, args.vhelm_root, idx)
            if row:
                rows.append(row)
                if args.limit is not None and len(rows) >= args.limit:
                    break
        if args.limit is not None and len(rows) >= args.limit:
            break

    if not rows:
        write_missing_status(args.vhelm_root, args.output, candidates)
        return 2

    rng = random.Random(args.seed)
    if len(rows) > args.sample_size:
        rows = rng.sample(rows, args.sample_size)
    rows.sort(key=lambda row: row["query_id"])

    if args.download_images:
        image_paths = sorted({str(row.get("image_path") or "") for row in rows if row.get("image_path")})
        downloaded = 0
        missing = 0
        errors: Dict[str, str] = {}
        with ThreadPoolExecutor(max_workers=max(1, args.image_workers)) as executor:
            futures = {
                executor.submit(download_image_if_needed, image_path, args.vhelm_root, args.image_base_url, args.image_timeout): image_path
                for image_path in image_paths
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="download VHELM sampled images", unit="image"):
                image_path = futures[future]
                try:
                    if future.result():
                        downloaded += 1
                    else:
                        missing += 1
                except Exception as exc:  # noqa: BLE001 - preserve the query and report the bad image.
                    errors[image_path] = f"{type(exc).__name__}: {exc}"
                    missing += 1
        for row in rows:
            image_error = errors.get(str(row.get("image_path") or ""))
            if image_error:
                row.setdefault("source", {})["image_download_error"] = image_error
        print(f"sampled_images_available={downloaded} sampled_images_missing_or_remote={missing}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"Wrote {len(rows)} VHELM standardized queries to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
