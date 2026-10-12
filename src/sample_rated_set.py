"""Sample the utterances that humans will rate, stratified by WER.

A rated item is one sentence plus the output of ONE model for it. Outputs are drawn from all
baseline models, because the best model alone produces too few bad transcripts to stratify on.

Procedure (seeded, reproducible):
  1. Sentence difficulty = WER of the worst model output, averaged over the languages in a group.
  2. Sentences are drawn evenly from difficulty bands; a short band is topped up from the others.
  3. Per language, each chosen sentence gets the model whose output lands in the WER band that
     is least filled so far, which flattens the WER distribution of the rated items.

Languages given to --parallel share the SAME sentences (FLEURS sentence IDs are parallel).
All other languages are sampled independently.

Usage:
    python src/sample_rated_set.py --parallel en el
    python src/sample_rated_set.py --parallel en it de el --overlap-language en --overlap-size 30
    python src/sample_rated_set.py --parallel en el --out ratings/pilot/rated_set_ids.csv --per-language 10

Output (IDs only, no WER, safe to commit and to show to raters):
    ratings/rated_set_ids.csv   columns: item_id, language, id, sentence_id, model, overlap
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval_sets"
METRICS = ROOT / "results" / "metrics" / "baseline_per_utterance.csv"
SEED = 42
BAND_EDGES = [-0.001, 0.0, 0.05, 0.15, 0.30, float("inf")]
BAND_LABELS = ["0", "0–5%", "5–15%", "15–30%", ">30%"]


def band(wer: pd.Series) -> pd.Series:
    return pd.cut(wer, BAND_EDGES, labels=BAND_LABELS)


def pick_sentences(difficulty: pd.Series, n: int, rng: np.random.Generator) -> list[str]:
    """Equal numbers per difficulty band; bands that run out are topped up from the rest."""
    bands = band(difficulty)
    pools = {b: list(rng.permutation(difficulty.index[bands == b])) for b in BAND_LABELS}
    chosen = []
    while len(chosen) < n and any(pools.values()):
        for b in reversed(BAND_LABELS):  # hardest first, so rare bands are never crowded out
            if pools[b] and len(chosen) < n:
                chosen.append(pools[b].pop())
    return chosen


def assign_models(wer: pd.DataFrame, sentences: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """wer: sentence_id x model table for one language. One model per sentence, flattening the bands."""
    filled = dict.fromkeys(BAND_LABELS, 0)
    rows = []
    # Sentences with the fewest distinct bands on offer are placed first: they leave the least choice.
    order = sorted(rng.permutation(sentences), key=lambda s: band(wer.loc[s]).nunique())
    for sentence in order:
        options = band(wer.loc[sentence])
        model = min(rng.permutation(options.index), key=lambda m: filled[options[m]])
        filled[options[model]] += 1
        rows.append({"sentence_id": sentence, "model": model})
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--parallel", nargs="*", default=[], help="languages that share the same sentences")
    parser.add_argument("--per-language", type=int, default=80)
    parser.add_argument("--normalisation", default="standard", help="WER variant used for stratifying")
    parser.add_argument("--overlap-language", help="language whose items are partly rated by both raters")
    parser.add_argument("--overlap-size", type=int, default=30)
    parser.add_argument("--out", type=Path, default=ROOT / "ratings" / "rated_set_ids.csv")
    parser.add_argument("--force", action="store_true", help="overwrite an existing (frozen) file")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} already exists. The rated set is frozen; use --force only before rating starts.")

    metrics = pd.read_csv(METRICS)
    metrics = metrics[metrics["normalisation"] == args.normalisation]
    sentence_of = pd.concat([pd.read_csv(p, dtype=str) for p in EVAL_DIR.glob("*.csv")]).set_index("id")["sentence_id"]
    metrics["sentence_id"] = metrics["id"].map(sentence_of)
    languages = sorted(metrics["language"].unique())

    groups = ([args.parallel] if args.parallel else []) + [[l] for l in languages if l not in args.parallel]
    rng = np.random.default_rng(SEED)
    items = []
    for group in groups:
        tables = {l: metrics[metrics["language"] == l].pivot(index="sentence_id", columns="model", values="wer").dropna() for l in group}
        shared = sorted(set.intersection(*(set(t.index) for t in tables.values())))
        difficulty = pd.concat([tables[l].loc[shared].max(axis=1) for l in group], axis=1).mean(axis=1)
        sentences = pick_sentences(difficulty, args.per_language, rng)
        for language in group:
            chosen = assign_models(tables[language], sentences, rng)
            chosen.insert(0, "language", language)
            items.append(chosen)

    rated = pd.concat(items, ignore_index=True)
    lookup = metrics.set_index(["language", "sentence_id", "model"])
    keys = list(zip(rated["language"], rated["sentence_id"], rated["model"]))
    rated["id"] = lookup.loc[keys, "id"].values
    wer = lookup.loc[keys, "wer"].values  # used for reporting and the overlap draw only, never saved

    rated["overlap"] = False
    if args.overlap_language:
        pool = rated[rated["language"] == args.overlap_language].assign(band=band(pd.Series(wer, index=rated.index)))
        # Spread the overlap items over the WER bands as well.
        pool = pool.sample(frac=1, random_state=SEED)
        order = pool.assign(rank=pool.groupby("band", observed=True).cumcount()).sort_values("rank", kind="stable")
        rated.loc[order.index[: args.overlap_size], "overlap"] = True

    # Shuffled presentation order, so the item number tells raters nothing about difficulty.
    rated = rated.sample(frac=1, random_state=SEED).sort_values("language", kind="stable").reset_index(drop=True)
    rated.insert(0, "item_id", rated["language"] + "_" + (rated.groupby("language").cumcount() + 1).astype(str).str.zfill(3))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    rated[["item_id", "language", "id", "sentence_id", "model", "overlap"]].to_csv(args.out, index=False)

    report = rated.assign(band=band(lookup.loc[list(zip(rated["language"], rated["sentence_id"], rated["model"])), "wer"].reset_index(drop=True)))
    print(f"wrote {len(rated)} items -> {args.out}\n")
    print("Items per WER band:\n" + report.groupby(["language", "band"], observed=False).size().unstack().to_string())
    print("\nItems per model:\n" + report.groupby(["language", "model"]).size().unstack(fill_value=0).to_string())
    if args.overlap_language:
        print(f"\nOverlap items ({args.overlap_language}) per WER band:\n" + report[report["overlap"]].groupby("band", observed=False).size().to_string())


if __name__ == "__main__":
    main()
