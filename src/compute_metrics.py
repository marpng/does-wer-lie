"""WER and CER for every transcription file in a results directory.

    python src/compute_metrics.py
    python src/compute_metrics.py --results-dir results/finetuned --name finetuned

Normalisations (applied to reference_raw and hypothesis alike):
    raw          whitespace only
    standard     Whisper BasicTextNormalizer
    standard_en  Whisper EnglishTextNormalizer, English only

Writes results/metrics/{name}_per_utterance.csv, {name}_table.csv and {name}_table.md.
"""

import argparse
import unicodedata
from pathlib import Path

import jiwer
import pandas as pd
from whisper_normalizer.basic import BasicTextNormalizer
from whisper_normalizer.english import EnglishTextNormalizer

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
OUT_DIR = ROOT / "results" / "metrics"

MODEL_ORDER = ["small", "medium", "large-v3"]
LANG_ORDER = ["en", "it", "de", "el", "gsw"]
WER_BINS = [-0.001, 0.0, 0.05, 0.15, 0.30, float("inf")]
WER_BIN_LABELS = ["0", "0–5%", "5–15%", "15–30%", ">30%"]

_basic = BasicTextNormalizer()
_english = EnglishTextNormalizer()


def clean(text: str) -> str:
    """Same Unicode form everywhere, no invisible format characters (zero-width space etc.)."""
    text = unicodedata.normalize("NFC", text)
    return "".join(c for c in text if unicodedata.category(c) != "Cf")


def normalisers(language: str) -> dict:
    out = {"raw": lambda text: " ".join(text.split()), "standard": _basic}
    if language == "en":
        out["standard_en"] = _english
    return out


def error_counts(reference: str, hypothesis: str) -> dict:
    words = jiwer.process_words(reference, hypothesis)
    chars = jiwer.process_characters(reference, hypothesis)
    return {
        "ref_words": words.hits + words.substitutions + words.deletions,
        "word_errors": words.substitutions + words.deletions + words.insertions,
        "ref_chars": chars.hits + chars.substitutions + chars.deletions,
        "char_errors": chars.substitutions + chars.deletions + chars.insertions,
    }


def load_references() -> pd.DataFrame:
    frames = [pd.read_csv(p, keep_default_na=False, usecols=["id", "reference_raw"]) for p in sorted(MANIFEST_DIR.glob("*.csv"))]
    return pd.concat(frames, ignore_index=True)


def per_utterance(results_dir: Path) -> pd.DataFrame:
    hyps = pd.concat([pd.read_csv(p, keep_default_na=False) for p in sorted(results_dir.glob("*.csv"))], ignore_index=True)
    df = hyps.merge(load_references(), on="id", how="left", validate="many_to_one")
    if df["reference_raw"].isna().any():
        raise SystemExit("Some transcribed IDs are missing from data/manifests.")

    rows = []
    for row in df.itertuples():
        for name, normalise in normalisers(row.language).items():
            reference = normalise(clean(row.reference_raw)).strip()
            hypothesis = normalise(clean(row.hypothesis)).strip()
            if not reference:
                continue
            counts = error_counts(reference, hypothesis)
            rows.append({
                "id": row.id, "language": row.language, "model": row.model, "normalisation": name,
                **counts,
                "wer": counts["word_errors"] / counts["ref_words"],
                "cer": counts["char_errors"] / counts["ref_chars"],
            })
    return pd.DataFrame(rows)


def ordered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for column, order in [("language", LANG_ORDER), ("model", MODEL_ORDER)]:
        present = set(df[column])
        known = [v for v in order if v in present]
        df[column] = pd.Categorical(df[column], known + sorted(present - set(known)))
    return df


def corpus_table(utt: pd.DataFrame) -> pd.DataFrame:
    g = ordered(utt).groupby(["language", "model", "normalisation"], observed=True)
    sums = g[["ref_words", "word_errors", "ref_chars", "char_errors"]].sum()
    table = pd.DataFrame({
        "utterances": g.size(),
        "wer": 100 * sums["word_errors"] / sums["ref_words"],
        "cer": 100 * sums["char_errors"] / sums["ref_chars"],
        "median_utt_wer": 100 * g["wer"].median(),
        "perfect_utts_%": 100 * g["wer"].apply(lambda s: (s == 0).mean()),
    })
    return table.reset_index()


def wer_distribution(utt: pd.DataFrame, normalisation: str) -> pd.DataFrame:
    """Utterances per WER band."""
    sub = ordered(utt[utt["normalisation"] == normalisation])
    bands = pd.cut(sub["wer"], WER_BINS, labels=WER_BIN_LABELS)
    counts = sub.groupby(["language", "model"], observed=True)[[]].size().index
    table = pd.crosstab([sub["language"], sub["model"]], bands, dropna=False).reindex(counts, fill_value=0)
    return table.reindex(columns=WER_BIN_LABELS, fill_value=0).reset_index()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results" / "baseline")
    parser.add_argument("--name", default="baseline")
    args = parser.parse_args()

    utt = per_utterance(args.results_dir)
    table = corpus_table(utt)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    utt.to_csv(OUT_DIR / f"{args.name}_per_utterance.csv", index=False)
    table.to_csv(OUT_DIR / f"{args.name}_table.csv", index=False)

    wide = table.pivot(index=["language", "model"], columns="normalisation", values=["wer", "cer"])
    wide.columns = [f"{metric.upper()} {norm}" for metric, norm in wide.columns]
    report = "\n\n".join([
        f"# {args.name.capitalize()} results\n\nGenerated by `src/compute_metrics.py`. Corpus-level, in %.",
        "## WER and CER by model, language and normalisation\n\n" + wide.reset_index().to_markdown(index=False, floatfmt=".2f"),
        "## Details\n\n" + table.to_markdown(index=False, floatfmt=".2f"),
        "## Utterances per WER band (standard normalisation)\n\n" + wer_distribution(utt, "standard").to_markdown(index=False),
    ])
    (OUT_DIR / f"{args.name}_table.md").write_text(report + "\n")
    print(report)


if __name__ == "__main__":
    main()
