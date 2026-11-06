#!/usr/bin/env python3
"""Draw the eight-model VHELM complementarity heatmap from stage-2 statistics."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


ROOT = Path(__file__).resolve().parents[1]
STATS = Path(__file__).with_name("eight_pool_stage2.json")
OUTPUTS = (
    Path(__file__).with_name("heatmap.pdf"),
    ROOT / "paper" / "fig_outlined" / "heatmap.pdf",
)


def main() -> None:
    payload = json.loads(STATS.read_text(encoding="utf-8"))
    block = payload["ood_paper"]
    names = block["heatmap_models"]
    frame = pd.DataFrame(
        block["heatmap_given_row_wrong_col_correct"],
        index=names,
        columns=names,
    )
    plt.rcParams.update({"font.size": 14})
    try:
        plt.rcParams.update({"font.family": "Times New Roman"})
    except Exception:
        plt.rcParams.update({"font.family": "serif"})
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(
        frame,
        annot=True,
        fmt=".2%",
        cmap="Blues",
        vmin=0,
        vmax=1,
        cbar=True,
        linewidths=0.5,
        linecolor="gray",
        square=True,
        ax=ax,
    )
    ax.set_xticklabels(ax.get_xticklabels(), fontsize=12, rotation=45, ha="right")
    ax.set_yticklabels(ax.get_yticklabels(), fontsize=12, rotation=0)
    fig.tight_layout(pad=0)
    for output in OUTPUTS:
        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, format="pdf", bbox_inches="tight", pad_inches=0)
        print(output)


if __name__ == "__main__":
    main()
