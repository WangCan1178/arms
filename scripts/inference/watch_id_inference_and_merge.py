#!/usr/bin/env python3
"""Watch M2 ID VLM inference outputs and merge when all four are complete."""

from __future__ import annotations

import argparse
import json
import shlex
import subprocess
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Tuple


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUTS = {
    "qwen": PROJECT_ROOT / "outputs" / "id" / "id_qwen.jsonl",
    "idefics2": PROJECT_ROOT / "outputs" / "id" / "id_idefics2.jsonl",
    "smolvlm2": PROJECT_ROOT / "outputs" / "id" / "id_smolvlm2.jsonl",
    "llama": PROJECT_ROOT / "outputs" / "id" / "id_llama.jsonl",
}
DEFAULT_LOG = PROJECT_ROOT / "logs" / "id_postprocess_watcher.log"
DEFAULT_INPUT = PROJECT_ROOT / "data" / "processed" / "id_queries_standardized.jsonl"
DEFAULT_TRAIN = PROJECT_ROOT / "data" / "processed" / "m2_id_free_train.jsonl"
DEFAULT_TEST = PROJECT_ROOT / "data" / "processed" / "m2_id_free_test.jsonl"


def timestamp() -> str:
    return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def append_log(path: Path, message: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(f"{timestamp()} {message}\n")


def count_jsonl(path: Path) -> Tuple[int, int, int, int]:
    rows = 0
    bad_json = 0
    errors = 0
    empty = 0
    if not path.exists():
        return rows, bad_json, errors, empty
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                bad_json += 1
                continue
            rows += 1
            if row.get("error"):
                errors += 1
            if not str(row.get("vlm_output", "")).strip():
                empty += 1
    return rows, bad_json, errors, empty


def is_vlm_inference_process_args(args: str) -> bool:
    try:
        tokens = shlex.split(args)
    except ValueError:
        tokens = args.split()
    if not tokens:
        return False

    if not any(Path(token).name == "run_vlm_inference.py" for token in tokens):
        return False
    if "--model" not in tokens:
        return False

    for index, token in enumerate(tokens):
        output_path = ""
        if token == "--output" and index + 1 < len(tokens):
            output_path = tokens[index + 1]
        elif token.startswith("--output="):
            output_path = token.split("=", 1)[1]
        if not output_path:
            continue
        normalized = output_path.replace("\\", "/")
        if normalized.startswith("outputs/id/id_") or "/outputs/id/id_" in normalized:
            return True
    return False


def count_running_inference_from_ps(ps_output: str) -> int:
    return sum(1 for line in ps_output.splitlines() if is_vlm_inference_process_args(line))


def running_inference_count() -> int:
    result = subprocess.run(["ps", "-eo", "args"], check=True, text=True, capture_output=True)
    return count_running_inference_from_ps(result.stdout)


def input_line_count(path: Path) -> int:
    with path.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def products_exist(paths: Iterable[Path]) -> bool:
    return all(path.exists() and path.stat().st_size > 0 for path in paths)


def run_merge(output_paths: Dict[str, Path], log_path: Path) -> int:
    command = [
        str(Path("/root/miniconda3/envs/m2-free-download/bin/python")),
        "scripts/label/merge_parse_label_filter.py",
        "--mode",
        "id",
        "--inputs",
        *(str(path) for path in output_paths.values()),
        "--canonical-input",
        str(DEFAULT_INPUT),
    ]
    append_log(log_path, "running_merge command=" + " ".join(command))
    with log_path.open("a", encoding="utf-8") as log_handle:
        result = subprocess.run(command, cwd=PROJECT_ROOT, text=True, stdout=log_handle, stderr=subprocess.STDOUT)
    append_log(log_path, f"merge_exit={result.returncode}")
    return result.returncode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expected-lines", type=int, default=None)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--interval", type=int, default=600)
    parser.add_argument("--log", type=Path, default=DEFAULT_LOG)
    parser.add_argument("--once", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    expected = args.expected_lines or input_line_count(args.input)
    append_log(args.log, f"watcher_start expected={expected}")

    while True:
        stats = {name: count_jsonl(path) for name, path in DEFAULT_OUTPUTS.items()}
        running = running_inference_count()
        status = " ".join(
            f"{name}={values[0]}/{expected},bad={values[1]},err={values[2]},empty={values[3]}"
            for name, values in stats.items()
        )
        append_log(args.log, f"{status} running={running}")

        if products_exist([DEFAULT_TRAIN, DEFAULT_TEST]):
            append_log(args.log, "merge_products_exist exiting")
            return 0

        all_complete = all(values[0] == expected and values[1] == 0 for values in stats.values())
        if all_complete and running == 0:
            return run_merge(DEFAULT_OUTPUTS, args.log)

        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
