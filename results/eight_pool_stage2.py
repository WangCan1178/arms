#!/usr/bin/env python3
"""Stage-2 statistics for the eight-model pool. Reads labels only."""

from __future__ import annotations

import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SUPERVISION = ROOT / "dataset" / "supervision" / "eight_pool"
OUT = ROOT / "results" / "eight_pool_stage2.json"

ORIGINAL = (
    "Qwen/Qwen2.5-VL-7B-Instruct",
    "HuggingFaceM4/idefics2-8b",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct",
    "meta-llama/Llama-3.2-11B-Vision-Instruct",
)
NEW = (
    "kimi-k3",
    "deepseek-v4.1-flash",
    "qwen3.8-max",
    "glm-5.3-flash",
)
MODELS = ORIGINAL + NEW
SHORT = {
    "Qwen/Qwen2.5-VL-7B-Instruct": "qwen2.5-vl-7b",
    "HuggingFaceM4/idefics2-8b": "idefics2-8b",
    "HuggingFaceTB/SmolVLM2-2.2B-Instruct": "smolvlm2-2.2b",
    "meta-llama/Llama-3.2-11B-Vision-Instruct": "llama-3.2-11b",
    "kimi-k3": "kimi-k3",
    "deepseek-v4.1-flash": "deepseek-v4.1-flash",
    "qwen3.8-max": "qwen3.8-max",
    "glm-5.3-flash": "glm-5.3-flash",
}


def load_labels(path: Path, keep: set[str] | None = None) -> list[dict[str, bool]]:
    rows: list[dict[str, bool]] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if keep is not None and row["query_id"] not in keep:
                continue
            labels = row["candidate_labels"]
            rows.append({name: bool(labels[name]) for name in MODELS})
    return rows


def load_ids(path: Path) -> set[str]:
    return {line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()}


def single_model_table(rows: list[dict[str, bool]]) -> list[dict[str, float | int | str]]:
    total = len(rows)
    table = []
    for name in MODELS:
        correct = sum(row[name] for row in rows)
        table.append(
            {
                "model": name,
                "short": SHORT[name],
                "correct": correct,
                "queries": total,
                "em": correct / total,
            }
        )
    return table


def coverage(rows: list[dict[str, bool]]) -> dict[str, float | int]:
    missed = [row for row in rows if not any(row[name] for name in ORIGINAL)]
    recovered = [row for row in missed if any(row[name] for name in NEW)]
    total = len(rows)
    return {
        "queries": total,
        "original_all_wrong": len(missed),
        "recovered_by_new_models": len(recovered),
        "recovered_rate_of_all": (len(recovered) / total) if total else 0.0,
        "recovered_rate_of_missed": (len(recovered) / len(missed)) if missed else 0.0,
    }


def complementarity(rows: list[dict[str, bool]]) -> list[list[float]]:
    matrix: list[list[float]] = []
    for source in MODELS:
        missed = [row for row in rows if not row[source]]
        row_values = []
        for target in MODELS:
            if source == target or not missed:
                row_values.append(0.0)
            else:
                row_values.append(sum(row[target] for row in missed) / len(missed))
        matrix.append(row_values)
    return matrix


def main() -> None:
    paper_ids = load_ids(SUPERVISION / "ood_paper_query_ids.txt")
    id_rows = load_labels(SUPERVISION / "id_test.jsonl")
    ood_rows = load_labels(SUPERVISION / "ood_test.jsonl", paper_ids)
    if len(ood_rows) != 7084:
        raise SystemExit(f"paper OOD subset has {len(ood_rows)} rows, expected 7084")

    payload = {
        "id_test": {
            "queries": len(id_rows),
            "oracle_em": 1.0 if all(any(row.values()) for row in id_rows) else None,
            "single_model_em": single_model_table(id_rows),
            "coverage": coverage(id_rows),
        },
        "ood_paper": {
            "queries": len(ood_rows),
            "oracle_em": 1.0 if all(any(row.values()) for row in ood_rows) else None,
            "single_model_em": single_model_table(ood_rows),
            "coverage": coverage(ood_rows),
            "heatmap_models": [SHORT[name] for name in MODELS],
            "heatmap_given_row_wrong_col_correct": complementarity(ood_rows),
        },
    }
    for split_name in ("id_test", "ood_paper"):
        ranked = sorted(payload[split_name]["single_model_em"], key=lambda item: item["em"], reverse=True)
        best = ranked[0]
        payload[split_name]["best_single"] = {
            "model": best["model"],
            "em": best["em"],
            "gap_to_oracle": 1.0 - best["em"],
        }
    OUT.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
