"""Explore all manifests: size, durations, text properties, audio health

    python src/explore_data.py
    python src/explore_data.py --no-audio

Writes reports/data_exploration/
"""

import argparse
import re
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
OUT_DIR = ROOT / "reports" / "data_exploration"

# One fixed colour per language in all figures
LANG_COLORS = {"en": "#2a78d6", "el": "#eb6834", "it": "#1baf7a", "de": "#eda100", "gsw": "#e87ba4"}
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"

SEED = 42
N_UTTERANCES = 500
QUIET_RMS_DB = -50.0  # low gain but speech present
NO_SPEECH_RANGE_DB = 10.0  # frame spread below this means no speech
CLIP_PEAK = 0.999
FRAME = 400  # 25 ms at 16 kHz


def load_manifests() -> pd.DataFrame:
    paths = sorted(MANIFEST_DIR.glob("*.csv"))
    if not paths:
        raise SystemExit(f"No manifests in {MANIFEST_DIR}. Run src/prepare_fleurs.py first.")
    df = pd.concat([pd.read_csv(p, dtype={"sentence_id": str}, keep_default_na=False) for p in paths], ignore_index=True)
    df["duration"] = df["duration"].astype(float)
    df["n_words"] = df["reference"].str.split().str.len()
    df["n_chars"] = df["reference"].str.replace(" ", "").str.len()
    df["words_per_sec"] = df["n_words"] / df["duration"]
    return df


def script_of(token: str) -> str:
    names = {unicodedata.name(c, "").split()[0] for c in token if c.isalpha()}
    if not names:
        return "none"
    return names.pop().lower() if len(names) == 1 else "mixed"


def text_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Per utterance flags for content where WER depends on normalisation"""
    raw, norm = df["reference_raw"], df["reference"]
    flags = pd.DataFrame(index=df.index)
    flags["has_digit"] = raw.str.contains(r"\d")
    flags["has_symbol"] = raw.str.contains(r"[%€$£°&+=/]")
    flags["has_hyphen_or_apostrophe"] = raw.str.contains(r"\w[-'’]\w")
    flags["has_abbreviation"] = raw.str.contains(r"\b(?:[A-ZΑ-Ω]\.){2,}|\b[A-ZΑ-Ω]{2,}\b")
    flags["has_quote_or_bracket"] = raw.str.contains(r"[\"“”«»()\[\]]")
    expected = {"en": "latin", "it": "latin", "de": "latin", "gsw": "latin", "el": "greek"}
    flags["has_foreign_script"] = [
        any(script_of(t) not in (expected.get(lang, "latin"), "none") for t in re.findall(r"\w+", text))
        for lang, text in zip(df["language"], norm)
    ]
    return flags


def overview(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby(["language", "split"], sort=False)
    out = g.agg(
        utterances=("id", "size"),
        unique_sentences=("sentence_id", "nunique"),
        hours=("duration", lambda s: s.sum() / 3600),
        words=("n_words", "sum"),
        dur_mean=("duration", "mean"),
        dur_median=("duration", "median"),
        dur_p05=("duration", lambda s: s.quantile(0.05)),
        dur_p95=("duration", lambda s: s.quantile(0.95)),
        dur_max=("duration", "max"),
        words_mean=("n_words", "mean"),
        words_per_sec=("words_per_sec", "median"),
        over_30s=("duration", lambda s: int((s > 30).sum())),
    )
    out["recordings_per_sentence"] = out["utterances"] / out["unique_sentences"]
    return out.reset_index()


def gender_table(df: pd.DataFrame) -> pd.DataFrame:
    return df.groupby(["language", "split", "gender"]).size().unstack(fill_value=0).reset_index()


def split_leakage(df: pd.DataFrame) -> pd.DataFrame:
    """Sentences shared between splits"""
    rows = []
    for lang, d in df.groupby("language", sort=False):
        sets = {s: set(x["sentence_id"]) for s, x in d.groupby("split")}
        for a, b in [("train", "dev"), ("train", "test"), ("dev", "test")]:
            if a in sets and b in sets:
                rows.append({"language": lang, "splits": f"{a} ∩ {b}", "shared_sentences": len(sets[a] & sets[b])})
    return pd.DataFrame(rows)


def parallel_coverage(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Sentences shared across languages"""
    sets = {lang: set(d["sentence_id"]) for lang, d in df[(df["split"] == split) & (df["dataset"] == "fleurs")].groupby("language", sort=False)}
    langs = list(sets)
    rows = [{"languages": lang, "sentences": len(sets[lang])} for lang in langs]
    if len(langs) > 1:
        rows.append({"languages": " ∩ ".join(langs), "sentences": len(set.intersection(*sets.values()))})
    return pd.DataFrame(rows)


def frame_dynamic_range(mono: np.ndarray) -> float:
    """dB spread between loud (p95) and quiet (p10) frames"""
    frames = mono[: len(mono) // FRAME * FRAME].reshape(-1, FRAME)
    if not len(frames):
        return 0.0
    db = 20 * np.log10(np.maximum(np.sqrt((frames**2).mean(axis=1)), 1e-10))
    return float(np.percentile(db, 95) - np.percentile(db, 10))


def audio_health(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in df[df["audio_path"] != ""].itertuples():
        audio, sr = sf.read(ROOT / row.audio_path, always_2d=True)
        mono = audio.mean(axis=1)
        rms = float(np.sqrt(np.mean(mono**2))) if len(mono) else 0.0
        rows.append({
            "id": row.id, "language": row.language, "split": row.split,
            "sample_rate": sr, "channels": audio.shape[1],
            "file_duration": len(mono) / sr,
            "duration_mismatch": abs(len(mono) / sr - row.duration) > 0.05,
            "rms_db": 20 * np.log10(max(rms, 1e-10)),
            "dynamic_range_db": frame_dynamic_range(mono),
            "peak": float(np.abs(mono).max()) if len(mono) else 0.0,
        })
    out = pd.DataFrame(rows)
    out["quiet"] = out["rms_db"] < QUIET_RMS_DB
    out["no_speech"] = out["dynamic_range_db"] < NO_SPEECH_RANGE_DB
    out["clipped"] = out["peak"] >= CLIP_PEAK
    return out


def budget_comparison(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Utterances against audio time per language with one recording per sentence"""
    test = df[df["split"] == split]
    one_per_sentence = test.sample(frac=1, random_state=SEED).drop_duplicates(["language", "sentence_id"])
    rows = []
    min_seconds = None
    samples = {}
    for lang, d in one_per_sentence.groupby("language", sort=False):
        s = d.head(N_UTTERANCES)
        samples[lang] = d
        min_seconds = s["duration"].sum() if min_seconds is None else min(min_seconds, s["duration"].sum())
        rows.append({"language": lang, "budget": f"{N_UTTERANCES} utterances (or all unique)",
                     "utterances": len(s), "minutes": s["duration"].sum() / 60, "ref_words": int(s["n_words"].sum())})
    for lang, d in samples.items():
        s = d[d["duration"].cumsum() <= min_seconds]
        rows.append({"language": lang, "budget": f"{min_seconds / 60:.0f} min of audio",
                     "utterances": len(s), "minutes": s["duration"].sum() / 60, "ref_words": int(s["n_words"].sum())})
    return pd.DataFrame(rows)


def style_axes(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, length=0)


def plot_distributions(df: pd.DataFrame, split: str, path: Path) -> None:
    """One row per language with a shared x axis per measure"""
    test = df[df["split"] == split]
    langs = list(test["language"].unique())
    measures = [("duration", "Utterance duration (s)"), ("n_words", "Reference words per utterance"),
                ("words_per_sec", "Speaking rate (words / s)")]
    fig, axes = plt.subplots(len(langs), len(measures), figsize=(12, 2.6 * len(langs)), sharex="col", squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for col, (measure, label) in enumerate(measures):
        bins = np.linspace(0, test[measure].quantile(0.995), 36)
        for row, lang in enumerate(langs):
            ax, values = axes[row][col], test.loc[test["language"] == lang, measure]
            ax.hist(values, bins=bins, color=LANG_COLORS.get(lang, MUTED), edgecolor=SURFACE, linewidth=1)
            ax.axvline(values.median(), color=INK, linewidth=1, linestyle=(0, (3, 3)))
            ax.text(0.98, 0.9, f"{lang.upper()}  median {values.median():.1f}", transform=ax.transAxes,
                    ha="right", va="top", color=INK, fontsize=9)
            style_axes(ax)
            if col == 0:
                ax.set_ylabel("utterances", color=MUTED, fontsize=9)
            if row == len(langs) - 1:
                ax.set_xlabel(label, color=MUTED, fontsize=10)
    fig.suptitle(f"FLEURS {split} set: duration, length and speaking rate per language", color=INK, fontsize=12, x=0.01, ha="left")
    fig.tight_layout()
    fig.savefig(path, dpi=160, facecolor=SURFACE)
    plt.close(fig)


def md_table(df: pd.DataFrame, digits: int = 2) -> str:
    return df.to_markdown(index=False, floatfmt=f".{digits}f")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", default="test")
    parser.add_argument("--no-audio", action="store_true")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_manifests()
    flags = text_flags(df)
    test_mask = df["split"] == args.split

    ov = overview(df)
    flag_rates = (flags[test_mask].groupby(df.loc[test_mask, "language"], sort=False).mean() * 100).reset_index()
    budgets = budget_comparison(df, args.split)

    sections = [
        "# Data exploration\n\nGenerated by `src/explore_data.py`. Do not edit by hand.",
        "## Overview per language and split\n\n" + md_table(ov),
        "## Speaker gender\n\n" + md_table(gender_table(df)),
        "## Sentence overlap between splits\n\n" + md_table(split_leakage(df)),
        f"## Parallel sentences across languages ({args.split})\n\n" + md_table(parallel_coverage(df, args.split)),
        f"## Normalisation-sensitive content, % of {args.split} utterances\n\n"
        "Computed on the raw (cased, punctuated) reference.\n\n" + md_table(flag_rates, 1),
        f"## Utterances vs. time ({args.split}, one recording per sentence, seed {SEED})\n\n" + md_table(budgets, 1),
    ]

    if not args.no_audio:
        health = audio_health(df)
        health.to_csv(OUT_DIR / "audio_health.csv", index=False)
        summary = health.groupby(["language", "split"], sort=False).agg(
            files=("id", "size"), sample_rates=("sample_rate", lambda s: ",".join(map(str, sorted(s.unique())))),
            channels=("channels", "max"), duration_mismatch=("duration_mismatch", "sum"),
            quiet=("quiet", "sum"), no_speech=("no_speech", "sum"), clipped=("clipped", "sum"),
            rms_db_median=("rms_db", "median"), peak_median=("peak", "median"),
        ).reset_index()
        sections.append(
            f"## Audio health\n\nQuiet = RMS below {QUIET_RMS_DB:.0f} dBFS (low recording gain, speech still present). "
            f"No speech = under {NO_SPEECH_RANGE_DB:.0f} dB between loud and quiet frames. Clipped = peak at full scale. "
            "Per-file values are in `audio_health.csv`.\n\n" + md_table(summary, 3)
        )
        bad = health[health["no_speech"]]
        if len(bad):
            sections.append("### Files without speech (exclude these before sampling)\n\n"
                            + md_table(bad[["id", "rms_db", "dynamic_range_db", "file_duration"]], 1))

    plot_distributions(df, args.split, OUT_DIR / f"distributions_{args.split}.png")
    sections.append(f"## Distributions\n\n![distributions](distributions_{args.split}.png)")

    ov.to_csv(OUT_DIR / "overview.csv", index=False)
    (OUT_DIR / "summary.md").write_text("\n\n".join(sections) + "\n")
    print("\n\n".join(sections))


if __name__ == "__main__":
    main()
