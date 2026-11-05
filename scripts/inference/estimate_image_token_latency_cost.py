#!/usr/bin/env python3
"""Add estimated image tokens to existing M2 Free token/latency metrics.

This local fallback does not require transformers, torch, or Pillow. It uses
image headers for dimensions and model-specific documented rules/approximations.
For processor-exact values, run a processor-based version on the server when SSH
and the HF cache are available.
"""

from __future__ import annotations

import argparse
import json
import math
import struct
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, MutableMapping, Optional, Tuple


MODEL_NAMES = {
    "qwen": "Qwen/Qwen2.5-VL-7B-Instruct",
    "idefics2": "HuggingFaceM4/idefics2-8b",
    "smolvlm2": "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "llama": "meta-llama/Llama-3.2-11B-Vision-Instruct",
}

RAW_OUTPUTS = [
    ("id", "qwen", "outputs/id/id_qwen.jsonl"),
    ("id", "idefics2", "outputs/id/id_idefics2.jsonl"),
    ("id", "smolvlm2", "outputs/id/id_smolvlm2.jsonl"),
    ("id", "llama", "outputs/id/id_llama.jsonl"),
    ("ood", "qwen", "outputs/ood/ood_qwen.jsonl"),
    ("ood", "idefics2", "outputs/ood/ood_idefics2.jsonl"),
    ("ood", "smolvlm2", "outputs/ood/ood_smolvlm2.jsonl"),
    ("ood", "llama", "outputs/ood/ood_llama.jsonl"),
]

PROCESSED_FILES = [
    "m2_id_free_train.jsonl",
    "m2_id_free_test.jsonl",
    "m2_ood_free_test.jsonl",
]

SERVER_ROOT = "/root/autodl-tmp/m2-free-datasets/"


@dataclass(frozen=True)
class ImageTokenEstimate:
    image_tokens: int
    width: int
    height: int
    method: str
    detail: Dict[str, Any]


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


def read_png_size(handle: Any) -> Optional[Tuple[int, int]]:
    handle.seek(16)
    raw = handle.read(8)
    if len(raw) != 8:
        return None
    width, height = struct.unpack(">II", raw)
    return width, height


def read_gif_size(handle: Any) -> Optional[Tuple[int, int]]:
    handle.seek(6)
    raw = handle.read(4)
    if len(raw) != 4:
        return None
    width, height = struct.unpack("<HH", raw)
    return width, height


def read_jpeg_size(handle: Any) -> Optional[Tuple[int, int]]:
    handle.seek(0)
    if handle.read(2) != b"\xff\xd8":
        return None
    while True:
        byte = handle.read(1)
        if not byte:
            return None
        if byte != b"\xff":
            continue
        marker = handle.read(1)
        while marker == b"\xff":
            marker = handle.read(1)
        if not marker:
            return None
        marker_code = marker[0]
        if marker_code in {0xD8, 0xD9, 0x01} or 0xD0 <= marker_code <= 0xD7:
            continue
        raw_len = handle.read(2)
        if len(raw_len) != 2:
            return None
        segment_len = struct.unpack(">H", raw_len)[0]
        if segment_len < 2:
            return None
        # SOF markers that carry dimensions.
        if marker_code in {
            0xC0,
            0xC1,
            0xC2,
            0xC3,
            0xC5,
            0xC6,
            0xC7,
            0xC9,
            0xCA,
            0xCB,
            0xCD,
            0xCE,
            0xCF,
        }:
            data = handle.read(5)
            if len(data) != 5:
                return None
            height, width = struct.unpack(">HH", data[1:5])
            return width, height
        handle.seek(segment_len - 2, 1)


def read_webp_size(handle: Any) -> Optional[Tuple[int, int]]:
    handle.seek(0)
    header = handle.read(30)
    if len(header) < 16 or header[:4] != b"RIFF" or header[8:12] != b"WEBP":
        return None
    chunk = header[12:16]
    if chunk == b"VP8X" and len(header) >= 30:
        width = 1 + int.from_bytes(header[24:27], "little")
        height = 1 + int.from_bytes(header[27:30], "little")
        return width, height
    if chunk == b"VP8 " and len(header) >= 30:
        # Lossy bitstream stores dimensions little-endian at bytes 26..30.
        width = struct.unpack("<H", header[26:28])[0] & 0x3FFF
        height = struct.unpack("<H", header[28:30])[0] & 0x3FFF
        return width, height
    if chunk == b"VP8L" and len(header) >= 25:
        bits = int.from_bytes(header[21:25], "little")
        width = (bits & 0x3FFF) + 1
        height = ((bits >> 14) & 0x3FFF) + 1
        return width, height
    return None


def read_image_size(path: Path) -> Tuple[int, int]:
    with path.open("rb") as handle:
        signature = handle.read(12)
        handle.seek(0)
        if signature.startswith(b"\x89PNG\r\n\x1a\n"):
            size = read_png_size(handle)
        elif signature[:6] in {b"GIF87a", b"GIF89a"}:
            size = read_gif_size(handle)
        elif signature.startswith(b"\xff\xd8"):
            size = read_jpeg_size(handle)
        elif signature[:4] == b"RIFF" and signature[8:12] == b"WEBP":
            size = read_webp_size(handle)
        else:
            size = None
    if size:
        return size
    # macOS fallback for rare formats.
    try:
        result = subprocess.run(
            ["sips", "-g", "pixelWidth", "-g", "pixelHeight", str(path)],
            check=True,
            text=True,
            capture_output=True,
        )
        width = height = None
        for line in result.stdout.splitlines():
            if "pixelWidth:" in line:
                width = int(line.rsplit(":", 1)[1].strip())
            if "pixelHeight:" in line:
                height = int(line.rsplit(":", 1)[1].strip())
        if width and height:
            return width, height
    except Exception:  # noqa: BLE001
        pass
    raise ValueError(f"Could not read image dimensions: {path}")


def localize_image_path(root: Path, image_path: str) -> Path:
    if image_path.startswith(SERVER_ROOT):
        return root / image_path[len(SERVER_ROOT) :]
    path = Path(image_path)
    if path.is_absolute():
        return path
    return root / image_path


def floor_by_factor(value: float, factor: int) -> int:
    return max(factor, math.floor(value / factor) * factor)


def ceil_by_factor(value: float, factor: int) -> int:
    return max(factor, math.ceil(value / factor) * factor)


def round_by_factor(value: float, factor: int) -> int:
    return max(factor, round(value / factor) * factor)


def qwen_image_tokens(width: int, height: int, min_tokens: int, max_tokens: int) -> ImageTokenEstimate:
    factor = 28
    min_pixels = min_tokens * factor * factor
    max_pixels = max_tokens * factor * factor
    h_bar = round_by_factor(height, factor)
    w_bar = round_by_factor(width, factor)
    original_pixels = width * height
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt(original_pixels / max_pixels)
        h_bar = floor_by_factor(height / beta, factor)
        w_bar = floor_by_factor(width / beta, factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / original_pixels)
        h_bar = ceil_by_factor(height * beta, factor)
        w_bar = ceil_by_factor(width * beta, factor)
    tokens = max(1, (h_bar // factor) * (w_bar // factor))
    return ImageTokenEstimate(
        image_tokens=tokens,
        width=width,
        height=height,
        method=(
            "qwen2.5-vl visual token estimate: resize to multiples of 28 within "
            f"{min_tokens}-{max_tokens} visual tokens; image_tokens=resized_h*resized_w/(28*28)"
        ),
        detail={"resized_width": w_bar, "resized_height": h_bar, "factor": factor},
    )


def fixed_image_tokens(width: int, height: int, tokens: int, method: str) -> ImageTokenEstimate:
    return ImageTokenEstimate(
        image_tokens=tokens,
        width=width,
        height=height,
        method=method,
        detail={},
    )


def llama_tiles(width: int, height: int, tile_size: int = 560, max_tiles: int = 4) -> Tuple[int, Tuple[int, int]]:
    area_tiles = max(1, min(max_tiles, math.ceil((width * height) / float(tile_size * tile_size))))
    candidates = [(rows, cols) for rows in range(1, max_tiles + 1) for cols in range(1, max_tiles + 1) if rows * cols == area_tiles]
    if not candidates:
        candidates = [(1, area_tiles)]
    aspect = width / max(height, 1)
    rows, cols = min(candidates, key=lambda rc: abs(math.log((rc[1] / rc[0]) / aspect)))
    return rows * cols, (rows, cols)


def llama_image_tokens(width: int, height: int) -> ImageTokenEstimate:
    tile_size = 560
    patch_size = 14
    tiles, grid = llama_tiles(width, height, tile_size=tile_size, max_tiles=4)
    tokens_per_tile = (tile_size // patch_size) ** 2
    tokens = tiles * tokens_per_tile
    return ImageTokenEstimate(
        image_tokens=tokens,
        width=width,
        height=height,
        method=(
            "llama-3.2-vision tile estimate: estimate 1-4 tiles from image area/aspect; "
            "image_tokens=tiles*(560/14)^2"
        ),
        detail={"tiles": tiles, "tile_grid": grid, "tile_size": tile_size, "patch_size": patch_size},
    )


def estimate_image_tokens(
    model_key: str,
    width: int,
    height: int,
    qwen_min_tokens: int,
    qwen_max_tokens: int,
    idefics2_tokens: int,
    smolvlm2_tokens: int,
) -> ImageTokenEstimate:
    if model_key == "qwen":
        return qwen_image_tokens(width, height, qwen_min_tokens, qwen_max_tokens)
    if model_key == "idefics2":
        return fixed_image_tokens(
            width,
            height,
            idefics2_tokens,
            f"idefics2 fixed estimate: {idefics2_tokens} image tokens per image; assumes no image splitting",
        )
    if model_key == "smolvlm2":
        return fixed_image_tokens(
            width,
            height,
            smolvlm2_tokens,
            f"smolvlm2 fixed estimate: image_seq_len={smolvlm2_tokens} tokens per image",
        )
    if model_key == "llama":
        return llama_image_tokens(width, height)
    raise KeyError(model_key)


def load_run_elapsed(summary_path: Path) -> Dict[Tuple[str, str], float]:
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    return {
        (run["split_group"], run["model_key"]): float(run["run_elapsed_s_from_log"])
        for run in summary["runs"]
    }


def estimate_per_output(
    root: Path,
    output_dir: Path,
    qwen_min_tokens: int,
    qwen_max_tokens: int,
    idefics2_tokens: int,
    smolvlm2_tokens: int,
) -> Tuple[Dict[str, Dict[str, Dict[str, Any]]], List[Dict[str, Any]]]:
    elapsed_by_run = load_run_elapsed(root / "data/metrics/token_latency/summary.json")
    image_size_cache: Dict[str, Tuple[int, int]] = {}
    metrics_by_query: Dict[str, Dict[str, Dict[str, Any]]] = {}
    summaries: List[Dict[str, Any]] = []

    for split_group, model_key, raw_rel in RAW_OUTPUTS:
        model_name = MODEL_NAMES[model_key]
        raw_path = root / raw_rel
        text_metrics_path = root / "data/metrics/token_latency/per_output" / f"{split_group}_{model_key}_token_latency.jsonl"
        rows: List[Dict[str, Any]] = []
        missing_images = 0
        observed_tokens_by_benchmark: Dict[str, List[int]] = defaultdict(list)

        raw_iter = iter_jsonl(raw_path)
        text_iter = iter_jsonl(text_metrics_path)
        for raw_row, text_row in zip(raw_iter, text_iter):
            image_path = str(raw_row.get("image_path", ""))
            local_image = localize_image_path(root, image_path)
            cache_key = str(local_image)
            try:
                if cache_key not in image_size_cache:
                    image_size_cache[cache_key] = read_image_size(local_image)
                width, height = image_size_cache[cache_key]
                estimate = estimate_image_tokens(
                    model_key,
                    width,
                    height,
                    qwen_min_tokens,
                    qwen_max_tokens,
                    idefics2_tokens,
                    smolvlm2_tokens,
                )
            except Exception as exc:  # noqa: BLE001
                missing_images += 1
                estimate = ImageTokenEstimate(
                    image_tokens=-1,
                    width=0,
                    height=0,
                    method=f"image token estimate failed: {type(exc).__name__}: {exc}",
                    detail={},
                )
            benchmark = str(raw_row.get("benchmark", ""))
            input_text_tokens = int(text_row["input_tokens"])
            output_tokens = int(text_row["output_tokens"])
            text_total_tokens = int(text_row["total_tokens"])
            image_tokens = int(estimate.image_tokens)
            if image_tokens >= 0:
                observed_tokens_by_benchmark[benchmark].append(image_tokens)
            rows.append(
                {
                    "query_id": raw_row.get("query_id"),
                    "benchmark": benchmark,
                    "split_group": split_group,
                    "model_name": model_name,
                    "raw_output_file": raw_rel,
                    "image_path": image_path,
                    "image_width": estimate.width,
                    "image_height": estimate.height,
                    "input_text_tokens": input_text_tokens,
                    "image_tokens_estimated": image_tokens,
                    "output_tokens": output_tokens,
                    "text_total_tokens": text_total_tokens,
                    "total_tokens_with_image": None,
                    "cost_tokens_with_image": None,
                    "image_token_estimation_method": estimate.method,
                    "image_token_estimation_detail": estimate.detail,
                    "latency_allocation_weight": None,
                    "image_tokens_imputed": image_tokens < 0,
                    "error": raw_row.get("error", ""),
                }
            )

        benchmark_avgs = {
            benchmark: sum(values) / len(values)
            for benchmark, values in observed_tokens_by_benchmark.items()
            if values
        }
        all_observed = [value for values in observed_tokens_by_benchmark.values() for value in values]
        global_avg = sum(all_observed) / len(all_observed) if all_observed else 0.0
        imputed_images = 0
        allocation_total = 0
        image_total = text_input_total = output_total = total_with_image = 0
        for row in rows:
            image_tokens = int(row["image_tokens_estimated"])
            if image_tokens < 0:
                benchmark = str(row.get("benchmark", ""))
                average = benchmark_avgs.get(benchmark, global_avg)
                image_tokens = max(0, int(round(average)))
                imputed_images += 1
                row["image_tokens_estimated"] = image_tokens
                row["image_token_estimation_detail"] = {
                    "previous_failure": row["image_token_estimation_method"],
                    "imputed_from": "same split_group/model/benchmark observed images"
                    if benchmark in benchmark_avgs
                    else "same split_group/model observed images",
                    "observed_rows": len(observed_tokens_by_benchmark.get(benchmark, []))
                    if benchmark in benchmark_avgs
                    else len(all_observed),
                    "observed_average_image_tokens": average,
                }
                row["image_token_estimation_method"] = (
                    "local image dimension missing; imputed image tokens from observed average "
                    f"for split={split_group}, model={model_key}, benchmark={benchmark or '<unknown>'}"
                )
            sample_total = int(row["input_text_tokens"]) + image_tokens + int(row["output_tokens"])
            allocation_weight = max(sample_total, 1)
            row["total_tokens_with_image"] = sample_total
            row["cost_tokens_with_image"] = sample_total
            row["latency_allocation_weight"] = allocation_weight
            allocation_total += allocation_weight
            image_total += image_tokens
            text_input_total += int(row["input_text_tokens"])
            output_total += int(row["output_tokens"])
            total_with_image += sample_total

        elapsed = elapsed_by_run[(split_group, model_key)]
        for row in rows:
            row["latency_s_estimated_with_image"] = (
                row["latency_allocation_weight"] / allocation_total * elapsed if allocation_total > 0 else 0.0
            )
            row["latency_allocation_method_with_image"] = (
                "run_elapsed_s * max(input_text_tokens + image_tokens_estimated + output_tokens, 1) / "
                "sum(max(input_text_tokens + image_tokens_estimated + output_tokens, 1)) within each model output file"
            )
            row.pop("latency_allocation_weight", None)

        metrics_path = output_dir / "per_output" / f"{split_group}_{model_key}_image_token_latency.jsonl"
        write_jsonl(metrics_path, rows)
        for row in rows:
            qid = row.get("query_id")
            if not qid:
                continue
            metrics_by_query.setdefault(str(qid), {})[model_name] = {
                "input_text_tokens": row["input_text_tokens"],
                "image_tokens_estimated": row["image_tokens_estimated"],
                "output_tokens": row["output_tokens"],
                "text_total_tokens": row["text_total_tokens"],
                "total_tokens_with_image": row["total_tokens_with_image"],
                "cost_tokens_with_image": row["cost_tokens_with_image"],
                "latency_s_estimated_with_image": row["latency_s_estimated_with_image"],
                "latency_allocation_method_with_image": row["latency_allocation_method_with_image"],
                "image_token_estimation_method": row["image_token_estimation_method"],
                "image_width": row["image_width"],
                "image_height": row["image_height"],
            }
        summaries.append(
            {
                "split_group": split_group,
                "model_key": model_key,
                "model_name": model_name,
                "raw_output_file": raw_rel,
                "metrics_file": str(metrics_path.relative_to(root)),
                "rows": len(rows),
                "missing_image_dimensions": missing_images,
                "imputed_image_token_rows": imputed_images,
                "run_elapsed_s_from_text_summary": elapsed,
                "input_text_tokens": text_input_total,
                "image_tokens_estimated": image_total,
                "output_tokens": output_total,
                "text_total_tokens": text_input_total + output_total,
                "total_tokens_with_image": total_with_image,
                "avg_image_tokens": image_total / len(rows) if rows else 0,
                "avg_total_tokens_with_image": total_with_image / len(rows) if rows else 0,
                "avg_latency_s_estimated_with_image": elapsed / len(rows) if rows else 0,
            }
        )
    return metrics_by_query, summaries


def enrich_processed(root: Path, output_dir: Path, metrics_by_query: Mapping[str, Mapping[str, Mapping[str, Any]]]) -> List[Dict[str, Any]]:
    summaries: List[Dict[str, Any]] = []
    source_dir = root / "data/processed_with_token_latency"
    if not source_dir.exists():
        source_dir = root / "data/processed"
    for name in PROCESSED_FILES:
        src = source_dir / name
        dst = output_dir / name
        rows = missing = values = 0

        def generate() -> Iterator[Dict[str, Any]]:
            nonlocal rows, missing, values
            for row in iter_jsonl(src):
                rows += 1
                query_id = str(row.get("query_id"))
                query_metrics = metrics_by_query.get(query_id, {})
                model_outputs = row.get("model_outputs", {})
                if isinstance(model_outputs, dict):
                    for model_name, output in model_outputs.items():
                        if not isinstance(output, dict):
                            continue
                        values += 1
                        metrics = query_metrics.get(model_name)
                        if metrics is None:
                            missing += 1
                            continue
                        previous_token_usage = output.get("token_usage")
                        previous_cost_tokens = output.get("cost_tokens")
                        previous_latency = output.get("latency_s_estimated")
                        if previous_token_usage is not None:
                            output["text_token_usage"] = previous_token_usage
                        if previous_cost_tokens is not None:
                            output["text_cost_tokens"] = previous_cost_tokens
                        if previous_latency is not None:
                            output["latency_s_estimated_text_only"] = previous_latency
                        output["token_usage"] = {
                            "input_text_tokens": metrics["input_text_tokens"],
                            "image_tokens_estimated": metrics["image_tokens_estimated"],
                            "output_tokens": metrics["output_tokens"],
                            "total_tokens": metrics["total_tokens_with_image"],
                        }
                        output["cost_tokens"] = metrics["cost_tokens_with_image"]
                        output["latency_s_estimated"] = metrics["latency_s_estimated_with_image"]
                        output["latency_allocation_method"] = metrics["latency_allocation_method_with_image"]
                        output["image_token_estimation_method"] = metrics["image_token_estimation_method"]
                        output["image_size"] = {
                            "width": metrics["image_width"],
                            "height": metrics["image_height"],
                        }
                yield row

        written = write_jsonl(dst, generate())
        summaries.append(
            {
                "source": str(src.relative_to(root)),
                "output": str(dst.relative_to(root)),
                "rows": rows,
                "written": written,
                "model_output_values": values,
                "missing_metric_values": missing,
            }
        )
    return summaries


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("data/metrics/image_token_latency"))
    parser.add_argument(
        "--processed-output-dir",
        type=Path,
        default=Path("data/processed_with_image_token_latency"),
    )
    parser.add_argument("--qwen-min-image-tokens", type=int, default=4)
    parser.add_argument("--qwen-max-image-tokens", type=int, default=16384)
    parser.add_argument("--idefics2-image-tokens", type=int, default=64)
    parser.add_argument("--smolvlm2-image-tokens", type=int, default=169)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = (root / args.output_dir).resolve() if not args.output_dir.is_absolute() else args.output_dir
    processed_output_dir = (
        (root / args.processed_output_dir).resolve()
        if not args.processed_output_dir.is_absolute()
        else args.processed_output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    processed_output_dir.mkdir(parents=True, exist_ok=True)

    metrics_by_query, run_summaries = estimate_per_output(
        root,
        output_dir,
        args.qwen_min_image_tokens,
        args.qwen_max_image_tokens,
        args.idefics2_image_tokens,
        args.smolvlm2_image_tokens,
    )
    processed_summaries = enrich_processed(root, processed_output_dir, metrics_by_query)
    summary = {
        "root": str(root),
        "output_dir": str(output_dir),
        "processed_output_dir": str(processed_output_dir),
        "assumptions": {
            "qwen": "default visual token range 4-16384; token corresponds to 28x28 resized pixels",
            "idefics2": f"{args.idefics2_image_tokens} image tokens per image; assumes no image splitting",
            "smolvlm2": f"{args.smolvlm2_image_tokens} image tokens per image from image_seq_len",
            "llama": "1-4 tile estimate; each 560x560 tile with 14x14 patches = 1600 tokens",
        },
        "runs": run_summaries,
        "processed_files": processed_summaries,
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
