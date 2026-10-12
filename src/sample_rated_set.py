"""Sample the items humans will rate, stratified by WER. One item = one sentence + one model's output.

    python src/sample_rated_set.py --parallel en it de el --overlap-language en --overlap-size 30

1. Sentence difficulty = WER of the worst model, averaged over the languages in a group.
2. Sentences are drawn evenly from the difficulty bands.
3. Per language, each sentence gets the model whose output falls in the least-filled WER band.

--parallel languages share the same sentences; the others are sampled independently.
Writes ratings/rated_set_ids.csv (item_id, language, id, sentence_id, model, overlap), no WER.
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
    """Round-robin over the bands, hardest first, until n sentences are chosen."""
    bands = band(difficulty)
    pools = {b: list(rng.permutation(difficulty.index[bands == b])) for b in BAND_LABELS}
    chosen = []
    while len(chosen) < n and any(pools.values()):
        for b in reversed(BAND_LABELS):
            if pools[b] and len(chosen) < n:
                chosen.append(pools[b].pop())
    return chosen


def assign_models(wer: pd.DataFrame, sentences: list[str], rng: np.random.Generator) -> pd.DataFrame:
    """wer: sentence_id x model table of one language. Returns one model per sentence."""
    filled = dict.fromkeys(BAND_LABELS, 0)
    rows = []
    # Sentences whose models all fall in one band go first: they leave no choice.
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
    parser.add_argument("--force", action="store_true", help="replace an existing file")
    args = parser.parse_args()

    if args.out.exists() and not args.force:
        raise SystemExit(f"{args.out} already exists; use --force only before rating has started.")

    metrics = pd.read_csv(METRICS)
    metrics = metrics[metrics["normalisation"] == args.normalisation]
    sentence_of = pd.concat([pd.read_csv(p, dtype=str) for p in EVAL_DIR.glob("*.csv")]).set_index("id")["sentence_id"]
    metrics["sentence_id"] = metrics["id"].map(sentence_of)
    languages = sorted(metrics["language"].unique())
    if unknown := set(args.parallel + [args.overlap_language or languages[0]]) - set(languages):
        raise SystemExit(f"No baseline metrics for: {sorted(unknown)}")

    groups = ([args.parallel] if args.parallel else []) + [[l] for l in languages if l not in args.parallel]
    rng = np.random.default_rng(SEED)
    items = []
    for group in groups:
        tables = {l: metrics[metrics["language"] == l].pivot(index="sentence_id", columns="model", values="wer").dropna() for l in group}
        shared = sorted(set.intersection(*(set(t.index) for t in tables.values())))
        difficulty = pd.concat([tables[l].loc[shared].max(axis=1) for l in group], axis=1).mean(axis=1)
        sentences = pick_sentences(difficulty, args.per_language, rng)
        if len(sentences) < args.per_language:
            print(f"WARNING {group}: only {len(sentences)} sentences available")
        for language in group:
            chosen = assign_models(tables[language], sentences, rng)
            chosen.insert(0, "language", language)
            items.append(chosen)

    rated = pd.concat(items, ignore_index=True)
    lookup = metrics.set_index(["language", "sentence_id", "model"])
    keys = list(zip(rated["language"], rated["sentence_id"], rated["model"]))
    rated["id"] = lookup.loc[keys, "id"].values
    wer = lookup.loc[keys, "wer"].values  # for the overlap draw and the printed summary only

    rated["overlap"] = False
    if args.overlap_language:
        pool = rated[rated["language"] == args.overlap_language].assign(band=band(pd.Series(wer, index=rated.index)))
        # Round-robin over the WER bands.
        pool = pool.sample(frac=1, random_state=SEED)
        order = pool.assign(rank=pool.groupby("band", observed=True).cumcount()).sort_values("rank", kind="stable")
        rated.loc[order.index[: args.overlap_size], "overlap"] = True

    # Shuffle so the item number says nothing about difficulty.
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
