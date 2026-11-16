"""Figure for the report: does a higher WER band mean a lower human score

    python src/plot_wer_vs_human.py

Input is results/analysis/wer_bands.csv from analyse_ratings.py
One panel per language, one bar per WER band (standard normalisation), the bar is the mean human score of the rated items
The number of items in each band is printed under the band label
Writes results/analysis/wer_vs_human.pdf and wer_vs_human.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.figure import Figure

ROOT = Path(__file__).resolve().parents[1]
ANALYSIS_DIR = ROOT / "results" / "analysis"

LANGUAGES = [("en", "English"), ("it", "Italian"), ("de", "German"), ("el", "Greek"), ("gsw", "Swiss German")]
BANDS = ["0", "0–5%", "5–15%", "15–30%", ">30%"]
BAND_TICKS = ["0", "≤5", "≤15", "≤30", ">30"]

PINK = "#D81B7A"      # bars, 4.9:1 against white
PINK_PANEL = "#FEF1F7"
PINK_GRID = "#F6CCE0"
INK = "#3A1A2A"
INK_MUTED = "#7A5568"


def panel_values(bands: pd.DataFrame, language: str) -> tuple[list, list[int]]:
    """Mean human score and item count for each band, None and 0 where the language has no item in that band"""
    rows = bands[bands["language"] == language].set_index("wer_band")
    values = [float(rows.loc[b, "mean_human"]) if b in rows.index else None for b in BANDS]
    counts = [int(rows.loc[b, "n"]) if b in rows.index else 0 for b in BANDS]
    return values, counts


def make_figure(bands: pd.DataFrame) -> Figure:
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, axes = plt.subplots(1, len(LANGUAGES), figsize=(7.2, 2.35), sharey=True)
    x = range(len(BANDS))
    for ax, (code, name) in zip(axes, LANGUAGES):
        values, counts = panel_values(bands, code)
        ax.set_facecolor(PINK_PANEL)
        ax.set_axisbelow(True)
        ax.yaxis.grid(True, color=PINK_GRID, linewidth=0.8)
        heights = [v if v is not None else 0 for v in values]
        ax.bar(x, heights, width=0.68, color=PINK, linewidth=0)
        for i, v in enumerate(values):
            if v is not None:
                ax.text(i, v + 0.05, f"{v:.1f}", ha="center", va="bottom", fontsize=6.8, color=INK)
        ax.set_title(name, fontsize=9, fontweight="bold", color=INK, pad=5)
        ax.set_xticks(list(x))
        ax.set_xticklabels([f"{t}\n({n})" for t, n in zip(BAND_TICKS, counts)], fontsize=6.8, color=INK_MUTED)
        ax.set_ylim(0, 2.35)
        ax.set_yticks([0, 1, 2])
        ax.tick_params(axis="both", length=0, pad=3)
        for side in ("top", "right", "left", "bottom"):
            ax.spines[side].set_visible(False)
    axes[0].set_yticklabels(["0 lost", "1 partly", "2 fully"], fontsize=7, color=INK)
    fig.supxlabel("WER band, upper limit in % (standard normalisation); number of rated items in brackets", fontsize=7.5, color=INK, y=0.0)
    fig.supylabel("Mean human score", fontsize=7.5, color=INK, x=0.005)
    fig.tight_layout(rect=(0.01, 0.04, 1, 1), w_pad=0.6)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bands", type=Path, default=ANALYSIS_DIR / "wer_bands.csv")
    parser.add_argument("--out-dir", type=Path, default=ANALYSIS_DIR)
    args = parser.parse_args()

    bands = pd.read_csv(args.bands)
    fig = make_figure(bands)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        out = args.out_dir / f"wer_vs_human.{suffix}"
        fig.savefig(out, dpi=300, facecolor="white", bbox_inches="tight", pad_inches=0.04)
        print(f"-> {out.relative_to(ROOT) if out.is_relative_to(ROOT) else out}")


if __name__ == "__main__":
    main()
