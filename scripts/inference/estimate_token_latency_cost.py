#!/usr/bin/env python3
"""Estimate token usage and latency allocation for M2 Free outputs.

This script does not run VLM inference. It loads each model tokenizer/processor,
counts text tokens from existing `prompt` and `vlm_output` fields, and allocates
the run-level tqdm elapsed time proportionally to per-sample total token count.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Tuple


MODEL_FILES = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "idefics2": "HuggingFaceM4/idefics2-8b",
    "smolvlm2": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "llama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}

RAW_OUTPUTS = [
    ("id", "qwen", "outputs/id/id_qwen.jsonl", "logs/id_qwen.log"),
    ("id", "idefics2", "outputs/id/id_idefics2.jsonl", "logs/id_idefics2.log"),
    ("id", "smolvlm2", "outputs/id/id_smolvlm2.jsonl", "logs/id_smolvlm2.log"),
    ("id", "llama", "outputs/id/id_llama.jsonl", "logs/id_llama.log"),
    ("ood", "qwen", "outputs/ood/ood_qwen.jsonl", "logs/ood_qwen.log"),
    ("ood", "idefics2", "outputs/ood/ood_idefics2.jsonl", "logs/ood_idefics2.log"),
    ("ood", "smolvlm2", "outputs/ood/ood_smolvlm2.jsonl", "logs/ood_smolvlm2.log"),
    ("ood", "llama", "outputs/ood/ood_llama.jsonl", "logs/ood_llama.log"),
]

PROCESSED_FILES = [
    "data/processed/m2_id_free_train.jsonl",
    "data/processed/m2_id_free_test.jsonl",
    "data/processed/m2_ood_free_test.jsonl",
]


@dataclass(frozen=True)
class RunSpec:
    split_group: str
    model_key: str
    model_name: str
    output_path: Path
    log_path: Path


def iter_jsonl(path: Path) -> Iterator[Dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            count += 1
    return count


def parse_elapsed_seconds(text: str) -> Optional[float]:
    parts = text.strip().split(":")
    try:
        nums = [int(part) for part in parts]
    except ValueError:
        return None
    if len(nums) == 2:
        minutes, seconds = nums
        return float(minutes * 60 + seconds)
    if len(nums) == 3:
        hours, minutes, seconds = nums
        return float(hours * 3600 + minutes * 60 + seconds)
    return None


def parse_tqdm_elapsed(log_path: Path) -> Optional[float]:
    if not log_path.exists():
        return None
    text = log_path.read_text(errors="ignore").replace("\r", "\n")
    # Example: inference Qwen/...: 83614query [17:44:03,  1.31query/s]
    pattern = re.compile(r"inference .*?:\s*(\d+)query\s*\[((?:\d+:)?\d{1,2}:\d{2}),\s*([^\]]+)\]")
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    # Use the largest completed query count, then the latest occurrence for that count.
    parsed: List[Tuple[int, float]] = []
    for match in matches:
        elapsed = parse_elapsed_seconds(match.group(2))
        if elapsed is not None:
            parsed.append((int(match.group(1)), elapsed))
    if not parsed:
        return None
    max_queries = max(item[0] for item in parsed)
    candidates = [elapsed for queries, elapsed in parsed if queries == max_queries]
    return candidates[-1] if candidates else parsed[-1][1]


def resolve_snapshot_path(model_name: str, hf_home: Path) -> Optional[Path]:
    hub = hf_home / "hub"
    repo_dir = hub / ("models--" + model_name.replace("/", "--"))
    snapshots = repo_dir / "snapshots"
    if not snapshots.exists():
        return None
    candidates = [path for path in snapshots.iterdir() if path.is_dir()]
    if not candidates:
        return None
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0]


def load_tokenizer(model_name: str, hf_home: Path, local_files_only: bool):
    from transformers import AutoProcessor, AutoTokenizer

    kwargs = {
        "cache_dir": str(hf_home),
        "local_files_only": local_files_only,
        "trust_remote_code": True,
    }
    errors: List[str] = []
    locations: List[str] = [model_name]
    snapshot = resolve_snapshot_path(model_name, hf_home)
    if snapshot is not None:
        locations.append(str(snapshot))

    for location in locations:
        try:
            processor = AutoProcessor.from_pretrained(location, **kwargs)
            tokenizer = getattr(processor, "tokenizer", None)
            if tokenizer is not None:
                return tokenizer
        except Exception as exc:  # noqa: BLE001
            errors.append(f"AutoProcessor {location}: {exc}")
        try:
            return AutoTokenizer.from_pretrained(location, **kwargs)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"AutoTokenizer {location}: {exc}")
    raise RuntimeError("Could not load tokenizer for " + model_name + "\n" + "\n".join(errors[-6:]))


def count_tokens(tokenizer: Any, text: Any, cache: MutableMapping[str, int]) -> int:
    if text is None:
        text = ""
    if not isinstance(text, str):
        text = str(text)
    cached = cache.get(text)
    if cached is not None:
        return cached
    encoded = tokenizer(text, add_special_tokens=False)
    token_ids = encoded.get("input_ids", [])
    if token_ids and isinstance(token_ids[0], list):
        count = len(token_ids[0])
    else:
        count = len(token_ids)
    cache[text] = count
    return count


def build_run_specs(root: Path) -> List[RunSpec]:
    return [
        RunSpec(
            split_group=split_group,
            model_key=model_key,
            model_name=MODEL_FILES[model_key],
            output_path=root / output_rel,
            log_path=root / log_rel,
        )
        for split_group, model_key, output_rel, log_rel in RAW_OUTPUTS
    ]


def estimate_raw_run(
    spec: RunSpec,
    tokenizer: Any,
    output_dir: Path,
    max_rows: Optional[int] = None,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    prompt_cache: Dict[str, int] = {}
    output_cache: Dict[str, int] = {}
    total_tokens = 0
    input_tokens_total = 0
    output_tokens_total = 0
    errors = 0

    for index, row in enumerate(iter_jsonl(spec.output_path), start=1):
        if max_rows is not None and index > max_rows:
            break
        input_tokens = count_tokens(tokenizer, row.get("prompt", ""), prompt_cache)
        output_tokens = count_tokens(tokenizer, row.get("vlm_output", ""), output_cache)
        sample_total = input_tokens + output_tokens
        # Ensure every row can receive a non-NaN latency share even if both sides are empty.
        allocation_weight = max(sample_total, 1)
        total_tokens += allocation_weight
        input_tokens_total += input_tokens
        output_tokens_total += output_tokens
        if row.get("error"):
            errors += 1
        rows.append(
            {
                "query_id": row.get("query_id"),
                "benchmark": row.get("benchmark"),
                "split_group": spec.split_group,
                "model_name": spec.model_name,
                "raw_output_file": str(spec.output_path),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": sample_total,
                "latency_allocation_weight": allocation_weight,
                "error": row.get("error", ""),
            }
        )

    elapsed = parse_tqdm_elapsed(spec.log_path)
    if elapsed is None:
        elapsed = 0.0
    for row in rows:
        if total_tokens > 0 and elapsed > 0:
            row["latency_s_estimated"] = row["latency_allocation_weight"] / total_tokens * elapsed
        else:
            row["latency_s_estimated"] = 0.0
        row["latency_allocation_method"] = (
            "run_elapsed_s * max(input_tokens + output_tokens, 1) / "
            "sum(max(input_tokens + output_tokens, 1)) within each model output file"
        )
        row.pop("latency_allocation_weight", None)

    metrics_path = output_dir / "per_output" / f"{spec.split_group}_{spec.model_key}_token_latency.jsonl"
    write_jsonl(metrics_path, rows)

    lookup: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for row in rows:
        query_id = row.get("query_id")
        if not query_id:
            continue
        lookup.setdefault(str(query_id), {})[spec.model_name] = {
            "input_tokens": row["input_tokens"],
            "output_tokens": row["output_tokens"],
            "total_tokens": row["total_tokens"],
            "latency_s_estimated": row["latency_s_estimated"],
            "latency_allocation_method": row["latency_allocation_method"],
        }

    summary = {
        "split_group": spec.split_group,
        "model_key": spec.model_key,
        "model_name": spec.model_name,
        "raw_output_file": str(spec.output_path),
        "log_file": str(spec.log_path),
        "metrics_file": str(metrics_path),
        "rows": len(rows),
        "rows_with_error": errors,
        "run_elapsed_s_from_log": elapsed,
        "input_tokens": input_tokens_total,
        "output_tokens": output_tokens_total,
        "total_tokens": input_tokens_total + output_tokens_total,
        "allocation_weight_total": total_tokens,
        "avg_input_tokens": input_tokens_total / len(rows) if rows else 0,
        "avg_output_tokens": output_tokens_total / len(rows) if rows else 0,
        "avg_total_tokens": (input_tokens_total + output_tokens_total) / len(rows) if rows else 0,
        "avg_latency_s_estimated": elapsed / len(rows) if rows and elapsed else 0,
    }
    return lookup, summary


def merge_lookups(
    target: Dict[str, Dict[str, Dict[str, Any]]],
    source: Dict[str, Dict[str, Dict[str, Any]]],
) -> None:
    for query_id, models in source.items():
        target.setdefault(query_id, {}).update(models)


def enrich_processed_file(
    src: Path,
    dst: Path,
    metrics_by_query: Mapping[str, Mapping[str, Mapping[str, Any]]],
) -> Dict[str, Any]:
    missing_metrics = 0
    rows = 0
    model_values = 0

    def generate() -> Iterator[Dict[str, Any]]:
        nonlocal missing_metrics, rows, model_values
        for row in iter_jsonl(src):
            rows += 1
            query_id = str(row.get("query_id"))
            query_metrics = metrics_by_query.get(query_id, {})
            model_outputs = row.get("model_outputs", {})
            if isinstance(model_outputs, dict):
                for model_name, output in model_outputs.items():
                    if not isinstance(output, dict):
                        continue
                    model_values += 1
                    metrics = query_metrics.get(model_name)
                    if metrics is None:
                        missing_metrics += 1
                        continue
                    output["token_usage"] = {
                        "input_tokens": metrics["input_tokens"],
                        "output_tokens": metrics["output_tokens"],
                        "total_tokens": metrics["total_tokens"],
                    }
                    output["cost_tokens"] = metrics["total_tokens"]
                    output["latency_s_estimated"] = metrics["latency_s_estimated"]
                    output["latency_allocation_method"] = metrics["latency_allocation_method"]
            yield row

    written = write_jsonl(dst, generate())
    return {
        "source": str(src),
        "output": str(dst),
        "rows": rows,
        "written": written,
        "model_output_values": model_values,
        "missing_model_metrics": missing_metrics,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--hf-home", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--processed-output-dir", type=Path, default=None)
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument("--max-rows", type=int, default=None)
    args = parser.parse_args()

    root = args.root.resolve()
    hf_home = (args.hf_home or root / "hf_cache").resolve()
    output_dir = (args.output_dir or root / "data/metrics/token_latency").resolve()
    processed_output_dir = (
        args.processed_output_dir or root / "data/processed_with_token_latency"
    ).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_query: Dict[str, Dict[str, Dict[str, Any]]] = {}
    summaries: List[Dict[str, Any]] = []
    tokenizer_cache: Dict[str, Any] = {}

    for spec in build_run_specs(root):
        if not spec.output_path.exists():
            raise FileNotFoundError(spec.output_path)
        print(f"Loading tokenizer for {spec.model_name}", flush=True)
        tokenizer = tokenizer_cache.get(spec.model_name)
        if tokenizer is None:
            tokenizer = load_tokenizer(spec.model_name, hf_home, args.local_files_only)
            tokenizer_cache[spec.model_name] = tokenizer
        print(f"Estimating {spec.split_group}/{spec.model_key}: {spec.output_path}", flush=True)
        lookup, summary = estimate_raw_run(spec, tokenizer, output_dir, args.max_rows)
        merge_lookups(metrics_by_query, lookup)
        summaries.append(summary)
        print(json.dumps(summary, ensure_ascii=False), flush=True)

    processed_summaries: List[Dict[str, Any]] = []
    for rel in PROCESSED_FILES:
        src = root / rel
        if not src.exists():
            continue
        dst = processed_output_dir / src.name
        processed_summaries.append(enrich_processed_file(src, dst, metrics_by_query))

    summary = {
        "root": str(root),
        "hf_home": str(hf_home),
        "output_dir": str(output_dir),
        "processed_output_dir": str(processed_output_dir),
        "max_rows": args.max_rows,
        "runs": summaries,
        "processed_files": processed_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary: {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
