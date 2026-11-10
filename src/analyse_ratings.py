"""Human ratings against automatic metrics (Q1 and Q2)

    python src/analyse_ratings.py
    python src/analyse_ratings.py --scores-dir ratings/scores --out-dir results/analysis

Input is ratings/scores/*.csv or *.numbers, ratings/rated_set_ids.csv and results/metrics/baseline_per_utterance.csv
Human score of an item is the mean of its raters (0 lost, 1 partly, 2 fully preserved)
Correlations use human score and minus metric so a positive rho means the metric agrees with humans

Writes to the output directory
  agreement.csv              rater agreement per language
  rater_disagreements.csv    items where raters differ
  correlations.csv           Spearman and Kendall with bootstrap CIs per language and normalisation
  correlations_by_model.csv  the same within one model
  normalisation_gain.csv     does one normalisation correlate better than another
  language_differences.csv   difference in rho between languages
  wer_bands.csv              mean human score per WER band
  items.csv                  one row per item
  error_candidates_{lang}.csv  items where WER and humans disagree most, fill in category by hand
  heatmap.png scatter.png error_categories.png
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
IDS = ROOT / "ratings" / "rated_set_ids.csv"
METRICS = ROOT / "results" / "metrics" / "baseline_per_utterance.csv"
MANIFEST_DIR = ROOT / "data" / "manifests"
BASELINE_DIR = ROOT / "results" / "baseline"

BAND_EDGES = [-0.001, 0.0, 0.05, 0.15, 0.30, float("inf")]  # same as sample_rated_set.py
BAND_LABELS = ["0", "0–5%", "5–15%", "15–30%", ">30%"]

LANG_ORDER = ["en", "it", "de", "el", "gsw"]
NORM_ORDER = ["raw", "standard", "standard_en", "content"]
SCORES = (0, 1, 2)
N_BOOT = 2000
SEED = 42
MAX_CANDIDATES = 10  # per direction
Q2_REFERENCE = "en"
MIN_MODEL_ITEMS = 15


def read_rating_file(path: Path) -> pd.DataFrame:
    """A rater download as a table from a CSV or the first table of a Numbers file"""
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, dtype={"item_id": str})
    try:
        from numbers_parser import Document
    except ImportError:
        raise SystemExit(f"{path.name} is a Numbers file: pip install numbers-parser, or export it as CSV.")
    rows = Document(str(path)).sheets[0].tables[0].rows(values_only=True)
    table = pd.DataFrame(rows[1:], columns=rows[0])
    table["item_id"] = table["item_id"].astype(str)
    return table


def load_ratings(scores_dir: Path, ids: pd.DataFrame) -> pd.DataFrame:
    files = sorted(p for p in scores_dir.iterdir() if p.suffix.lower() in {".csv", ".numbers"})
    csv_stems = {p.stem for p in files if p.suffix.lower() == ".csv"}
    skipped = [p.name for p in files if p.suffix.lower() == ".numbers" and p.stem in csv_stems]
    if skipped:
        print(f"Using the CSV instead of {skipped}.")
    files = [p for p in files if p.name not in skipped]
    if not files:
        raise SystemExit(f"No rating CSV or Numbers files in {scores_dir}.")
    ratings = pd.concat([read_rating_file(p).assign(file=p.name) for p in files], ignore_index=True)
    ratings["rater"] = ratings["rater"].fillna("").astype(str).str.strip().str.upper()
    if (ratings["rater"] == "").any():
        raise SystemExit("Some rows have no rater initials: " + ", ".join(sorted(ratings.loc[ratings["rater"] == "", "file"].unique())))

    unknown = set(ratings["item_id"]) - set(ids["item_id"])
    if unknown:
        raise SystemExit(f"{len(unknown)} item_ids are not in {IDS.name}, e.g. {sorted(unknown)[:3]}")

    blank = ratings["score"].isna()
    if blank.any():
        print(f"Dropping {blank.sum()} unrated rows: " + ratings[blank].groupby("rater")["item_id"].count().to_dict().__str__())
    ratings = ratings[~blank].astype({"score": int})
    ratings["flag"] = ratings["flag"].fillna(0).astype(int)
    if not ratings["score"].isin(SCORES).all():
        raise SystemExit("Scores outside 0/1/2 found.")

    dupes = ratings.duplicated(["item_id", "rater"], keep="last")
    if dupes.any():
        print(f"{dupes.sum()} duplicate (item, rater) rows, keeping the last one in file order.")
        ratings = ratings[~dupes]

    ratings = ratings.merge(ids[["item_id", "language"]], on="item_id")
    return ratings


def report_coverage(ratings: pd.DataFrame, ids: pd.DataFrame) -> None:
    print("\nCoverage (items rated / items in the frozen set):")
    for language, group in ids.groupby("language"):
        done = ratings[ratings["language"] == language]
        if done.empty:
            continue
        per_rater = done.groupby("rater")["item_id"].nunique().to_dict()
        both = int((done.groupby("item_id")["rater"].nunique() >= 2).sum())
        print(f"  {language}: {len(group)} items, raters {per_rater}, rated by 2+ raters: {both}")


def weighted_kappa(a: np.ndarray, b: np.ndarray, power: int) -> float:
    """Cohen kappa with linear (power 1) or quadratic (power 2) weights on the 0 1 2 scale"""
    k = len(SCORES)
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[x, y] += 1
    observed /= observed.sum()
    expected = np.outer(observed.sum(axis=1), observed.sum(axis=0))
    weights = np.abs(np.subtract.outer(range(k), range(k))) ** power / (k - 1) ** power
    denominator = (weights * expected).sum()
    if denominator == 0:  # kappa is undefined when a rater always gives the same score
        return float("nan")
    return 1 - (weights * observed).sum() / denominator


def gwet_ac2(a: np.ndarray, b: np.ndarray) -> float:
    """Gwet AC2 with linear weights, steadier than kappa when most items get one score"""
    k = len(SCORES)
    observed = np.zeros((k, k))
    for x, y in zip(a, b):
        observed[x, y] += 1
    observed /= observed.sum()
    agree = 1 - np.abs(np.subtract.outer(range(k), range(k))) / (k - 1)
    pa = (agree * observed).sum()
    pi = (observed.sum(axis=1) + observed.sum(axis=0)) / 2
    pe = agree.sum() / (k * (k - 1)) * (pi * (1 - pi)).sum()
    return (pa - pe) / (1 - pe)


def agreement_table(ratings: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    rows = []
    for language in LANG_ORDER:
        wide = ratings[ratings["language"] == language].pivot(index="item_id", columns="rater", values="score").dropna()
        if wide.shape[1] < 2:
            continue
        if wide.shape[1] > 2:
            print(f"{language}: {wide.shape[1]} raters, agreement computed for the first two ({list(wide.columns[:2])}).")
        a, b = wide.iloc[:, 0].to_numpy(), wide.iloc[:, 1].to_numpy()
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(a), len(a))
            boots.append(weighted_kappa(a[idx], b[idx], 1))
        low, high = np.nanpercentile(boots, [2.5, 97.5])
        rows.append({
            "language": language, "raters": "+".join(wide.columns[:2]), "items": len(a),
            "exact_agreement": (a == b).mean(),
            "mean_score_rater1": a.mean(), "mean_score_rater2": b.mean(),
            "kappa_linear": weighted_kappa(a, b, 1), "kappa_linear_ci_low": low, "kappa_linear_ci_high": high,
            "kappa_quadratic": weighted_kappa(a, b, 2),
            "gwet_ac2": gwet_ac2(a, b),
            "share_score_2_rater1": (a == 2).mean(), "share_score_2_rater2": (b == 2).mean(),
        })
    return pd.DataFrame(rows)


def build_items(ratings: pd.DataFrame, ids: pd.DataFrame, metrics: pd.DataFrame) -> pd.DataFrame:
    wide = ratings.pivot(index="item_id", columns="rater", values="score")
    wide.columns = [f"score_{c}" for c in wide.columns]
    flags = ratings.groupby("item_id")["flag"].max().rename("flagged")
    items = ids.set_index("item_id").join(wide, how="inner").join(flags)
    items["human"] = ratings.groupby("item_id")["score"].mean()
    items["n_raters"] = ratings.groupby("item_id")["rater"].nunique()
    comments = ratings[ratings["comment"].fillna("").str.strip() != ""].assign(comment=lambda d: d["rater"] + ": " + d["comment"].str.strip())
    items["comments"] = comments.groupby("item_id")["comment"].agg(" | ".join)
    items["comments"] = items["comments"].fillna("")

    # only items scored by every rater of the language
    expected = ratings.groupby("language")["rater"].nunique()
    incomplete = items["n_raters"] < items["language"].map(expected)
    if incomplete.any():
        print(f"Dropping {incomplete.sum()} items not scored by all raters of their language.")
        items = items[~incomplete]

    long = metrics.melt(id_vars=["id", "language", "model", "normalisation"], value_vars=["wer", "cer"], var_name="metric")
    long["metric"] = long["metric"] + "_" + long["normalisation"]
    wide_metrics = long.pivot(index=["id", "model"], columns="metric", values="value").reset_index()
    items = items.reset_index().merge(wide_metrics, on=["id", "model"], how="left", validate="one_to_one")
    missing = items["wer_raw"].isna()
    if missing.any():
        raise SystemExit(f"{missing.sum()} rated items have no metrics in {METRICS.name}.")
    return items


def bootstrap_corr(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> dict:
    """Spearman and Kendall of x with y with percentile bootstrap CIs over items"""
    def corr(idx):
        if len(set(x[idx])) < 2 or len(set(y[idx])) < 2:
            return np.nan, np.nan
        return stats.spearmanr(x[idx], y[idx])[0], stats.kendalltau(x[idx], y[idx])[0]

    rho, tau = corr(np.arange(len(x)))
    boots = np.array([corr(rng.integers(0, len(x), len(x))) for _ in range(N_BOOT)])
    (rho_low, tau_low), (rho_high, tau_high) = np.nanpercentile(boots, 2.5, axis=0), np.nanpercentile(boots, 97.5, axis=0)
    return {"spearman": rho, "spearman_ci_low": rho_low, "spearman_ci_high": rho_high,
            "kendall": tau, "kendall_ci_low": tau_low, "kendall_ci_high": tau_high}


def metric_columns(items: pd.DataFrame) -> list[str]:
    return [c for c in items.columns if c.startswith(("wer_", "cer_"))]


def correlation_table(items: pd.DataFrame, rng: np.random.Generator, by_model: bool = False) -> pd.DataFrame:
    """Metric against human score per language or per language and model"""
    rows = []
    for language in LANG_ORDER:
        language_items = items[items["language"] == language]
        groups = [("all", language_items)]
        if by_model:
            groups = [(m, g) for m, g in language_items.groupby("model") if len(g) >= MIN_MODEL_ITEMS]
        for model, subset in groups:
            for column in metric_columns(items):
                if subset[column].isna().all():
                    continue
                metric, normalisation = column.split("_", 1)
                result = bootstrap_corr(subset["human"].to_numpy(), -subset[column].to_numpy(), rng)
                rows.append({"language": language, "model": model, "metric": metric, "normalisation": normalisation, "n": len(subset), **result})
    table = pd.DataFrame(rows)
    if table.empty:
        return table
    for column, order in (("language", LANG_ORDER), ("normalisation", NORM_ORDER)):
        table[column] = pd.Categorical(table[column], order, ordered=True)
    table = table.sort_values(["language", "model", "metric", "normalisation"])
    return table.astype({"language": str, "normalisation": str})


def boot_rho(x: np.ndarray, columns: list[np.ndarray], rng: np.random.Generator) -> np.ndarray:
    """Spearman of x with each column over the same bootstrap resamples with row 0 as observed"""
    out = np.full((N_BOOT + 1, len(columns)), np.nan)
    for i in range(N_BOOT + 1):
        idx = np.arange(len(x)) if i == 0 else rng.integers(0, len(x), len(x))
        if len(set(x[idx])) < 2:
            continue
        out[i] = [stats.spearmanr(x[idx], c[idx])[0] for c in columns]
    return out


def gain_row(label: dict, diff: np.ndarray) -> dict:
    low, high = np.nanpercentile(diff[1:], [2.5, 97.5])
    return {**label, "delta_rho": diff[0], "ci_low": low, "ci_high": high, "excludes_zero": bool(low > 0 or high < 0)}


def normalisation_gain(items: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Paired bootstrap on the same items for the rho difference between normalisations"""
    pairs = [("raw", "standard"), ("standard", "content"), ("raw", "content"), ("standard", "standard_en")]
    rows = []
    for language in LANG_ORDER:
        subset = items[items["language"] == language]
        if subset.empty:
            continue
        for metric in ("wer", "cer"):
            for first, second in pairs:
                a, b = f"{metric}_{first}", f"{metric}_{second}"
                if a not in subset or b not in subset or subset[[a, b]].isna().any().any():
                    continue
                rho = boot_rho(subset["human"].to_numpy(), [-subset[a].to_numpy(), -subset[b].to_numpy()], rng)
                rows.append(gain_row({"language": language, "metric": metric, "from": first, "to": second, "n": len(subset)}, rho[:, 1] - rho[:, 0]))
    return pd.DataFrame(rows)


def language_differences(items: pd.DataFrame, rng: np.random.Generator) -> pd.DataFrame:
    """Q2 rho difference between a language and the reference language with independent bootstraps"""
    reference = items[items["language"] == Q2_REFERENCE]
    if reference.empty:
        return pd.DataFrame()
    rows = []
    for column in metric_columns(items):
        if reference[column].isna().any():
            continue
        base = boot_rho(reference["human"].to_numpy(), [-reference[column].to_numpy()], rng)[:, 0]
        for language in LANG_ORDER:
            subset = items[items["language"] == language]
            if language == Q2_REFERENCE or subset.empty or subset[column].isna().any():
                continue
            other = boot_rho(subset["human"].to_numpy(), [-subset[column].to_numpy()], rng)[:, 0]
            metric, normalisation = column.split("_", 1)
            rows.append(gain_row({"language": language, "reference": Q2_REFERENCE, "metric": metric, "normalisation": normalisation}, other - base))
    return pd.DataFrame(rows)


def rater_disagreements(items: pd.DataFrame, out: Path) -> None:
    """Items where the raters differ with texts and comments"""
    reference = pd.concat([pd.read_csv(p, keep_default_na=False) for p in MANIFEST_DIR.glob("*.csv")])[["id", "reference_raw"]]
    hypotheses = pd.concat([pd.read_csv(p, keep_default_na=False) for p in BASELINE_DIR.glob("*.csv")])[["id", "model", "hypothesis"]]
    score_columns = [c for c in items.columns if c.startswith("score_")]
    differ = items[items[score_columns].nunique(axis=1, dropna=True) > 1]
    differ = differ.merge(reference, on="id", how="left").merge(hypotheses, on=["id", "model"], how="left")
    columns = ["item_id", "language", "model"] + score_columns + ["wer_standard", "reference_raw", "hypothesis", "comments"]
    differ.dropna(axis=1, how="all")[[c for c in columns if c in differ and not differ[c].isna().all()]].to_csv(out, index=False)
    print(f"\n{len(differ)} items with differing rater scores -> {out.name}")


def wer_bands(items: pd.DataFrame) -> pd.DataFrame:
    bands = pd.cut(items["wer_standard"], BAND_EDGES, labels=BAND_LABELS)
    table = items.groupby(["language", bands], observed=True)["human"].agg(n="size", mean_human="mean").reset_index()
    return table.rename(columns={"wer_standard": "wer_band"})


def error_candidates(items: pd.DataFrame, out_dir: Path) -> None:
    """The items where WER and humans disagree most, found by rank, with texts and rater comments for manual categorisation.

    harsh   = high WER percentile and high human percentile (WER punishes something humans accept)
    lenient = low WER percentile and low human percentile (WER misses something humans penalise)
    """
    reference = pd.concat([pd.read_csv(p, keep_default_na=False) for p in MANIFEST_DIR.glob("*.csv")])[["id", "reference_raw"]]
    hypotheses = pd.concat([pd.read_csv(p, keep_default_na=False) for p in BASELINE_DIR.glob("*.csv")])[["id", "model", "hypothesis"]]
    items = items.merge(reference, on="id", how="left").merge(hypotheses, on=["id", "model"], how="left")

    for language, group in items.groupby("language"):
        group = group.assign(wer_pct=group["wer_standard"].rank(pct=True), human_pct=group["human"].rank(pct=True))
        group["harsh"] = group["wer_pct"] + group["human_pct"]
        group["lenient"] = (1 - group["wer_pct"]) + (1 - group["human_pct"])
        harsh = group[group["human"] >= 1.5].nlargest(MAX_CANDIDATES, "harsh").assign(mismatch="wer_harsh")
        lenient = group[group["human"] < 2].drop(harsh.index, errors="ignore").nlargest(MAX_CANDIDATES, "lenient").assign(mismatch="wer_lenient")
        picked = pd.concat([harsh, lenient])
        columns = ["item_id", "mismatch", "model", "wer_raw", "wer_standard", "wer_content", "human", "flagged", "reference_raw", "hypothesis", "comments"]
        path = out_dir / f"error_candidates_{language}.csv"
        picked["category"] = ""
        if path.exists() and "category" in pd.read_csv(path).columns:
            old = pd.read_csv(path, keep_default_na=False).drop_duplicates("item_id").set_index("item_id")["category"]
            picked["category"] = picked["item_id"].map(old).fillna("")
        picked[[c for c in columns if c in picked] + ["category"]].to_csv(path, index=False)
        print(f"  {language}: {len(harsh)} wer_harsh, {len(lenient)} wer_lenient -> {path.name}")


def plot_heatmap(table: pd.DataFrame, out: Path) -> None:
    table = table.assign(column=table["metric"].str.upper() + " " + table["normalisation"])
    rows = [f"{m} {n}" for m in ("WER", "CER") for n in NORM_ORDER if f"{m} {n}" in set(table["column"])]
    grid = table.pivot(index="column", columns="language", values="spearman").reindex(index=rows, columns=[l for l in LANG_ORDER if l in set(table["language"])])
    fig, ax = plt.subplots(figsize=(1.4 * grid.shape[1] + 2, 0.5 * grid.shape[0] + 1.5))
    image = ax.imshow(grid.to_numpy(), cmap="viridis", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(grid.shape[1]), grid.columns)
    ax.set_yticks(range(grid.shape[0]), grid.index)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            value = grid.iat[i, j]
            if pd.notna(value):
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", color="white" if value < 0.6 else "black")
    fig.colorbar(image, label="Spearman ρ (human score, −error)")
    ax.set_title("Agreement between metric and human judgement")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_scatter(items: pd.DataFrame, out: Path, normalisations: tuple = ("standard", "content")) -> None:
    languages = [l for l in LANG_ORDER if (items["language"] == l).any()]
    fig, axes = plt.subplots(len(normalisations), len(languages), figsize=(3.6 * len(languages), 3.2 * len(normalisations)), sharey=True, squeeze=False)
    rng = np.random.default_rng(SEED)
    for row, normalisation in zip(axes, normalisations):
        for ax, language in zip(row, languages):
            subset = items[items["language"] == language]
            ax.scatter(subset[f"wer_{normalisation}"].clip(upper=1) * 100, subset["human"] + rng.uniform(-0.04, 0.04, len(subset)), alpha=0.6, s=18)
            ax.set_title(f"{language}, {normalisation}")
            ax.set_xlabel("WER, % (capped at 100)")
            ax.set_yticks([0, 0.5, 1, 1.5, 2])
        row[0].set_ylabel("Mean human score")
    fig.tight_layout()
    fig.savefig(out, dpi=200)
    plt.close(fig)


def plot_error_categories(out_dir: Path) -> None:
    frames = []
    for path in sorted(out_dir.glob("error_candidates_*.csv")):
        data = pd.read_csv(path, keep_default_na=False)
        data = data[data["category"].str.strip() != ""]
        if not data.empty:
            frames.append(data.assign(language=path.stem.removeprefix("error_candidates_")))
    if not frames:
        print("No categorised error candidates yet (fill the `category` column), skipping error_categories.png.")
        return
    counts = pd.concat(frames).groupby(["category", "language"]).size().unstack(fill_value=0)
    counts = counts[[l for l in LANG_ORDER if l in counts.columns]]
    ax = counts.loc[counts.sum(axis=1).sort_values().index].plot.barh(stacked=True, figsize=(7, 0.45 * len(counts) + 1.5))
    ax.set_xlabel("Number of cases")
    ax.set_ylabel("")
    ax.set_title("Mismatches between WER and human judgement, by cause")
    ax.figure.tight_layout()
    ax.figure.savefig(out_dir / "error_categories.png", dpi=200)
    plt.close(ax.figure)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--scores-dir", type=Path, default=ROOT / "ratings" / "scores")
    parser.add_argument("--ids", type=Path, default=IDS)
    parser.add_argument("--metrics", type=Path, default=METRICS)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "analysis")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    ids = pd.read_csv(args.ids, dtype={"sentence_id": str})
    ratings = load_ratings(args.scores_dir, ids)
    report_coverage(ratings, ids)

    agreement = agreement_table(ratings, rng)
    agreement.to_csv(args.out_dir / "agreement.csv", index=False)
    print("\nInter-rater agreement:\n" + agreement.round(3).to_string(index=False))

    items = build_items(ratings, ids, pd.read_csv(args.metrics))
    items.to_csv(args.out_dir / "items.csv", index=False)
    flagged = items.groupby("language")["flagged"].sum().to_dict()
    print(f"\nFlagged items per language: {flagged}")

    table = correlation_table(items, rng)
    table.to_csv(args.out_dir / "correlations.csv", index=False)
    print("\nCorrelation of metric with human score (Spearman, 95% bootstrap CI):")
    shown = table.assign(ci=table.apply(lambda r: f"[{r.spearman_ci_low:.2f}, {r.spearman_ci_high:.2f}]", axis=1))
    print(shown[["language", "metric", "normalisation", "n", "spearman", "ci", "kendall"]].round(3).to_string(index=False))

    by_model = correlation_table(items, rng, by_model=True)
    by_model.to_csv(args.out_dir / "correlations_by_model.csv", index=False)
    print(f"\nWithin-model correlations, WER standard (models with >= {MIN_MODEL_ITEMS} items):")
    print(by_model[(by_model["metric"] == "wer") & (by_model["normalisation"] == "standard")][["language", "model", "n", "spearman", "spearman_ci_low", "spearman_ci_high"]].round(3).to_string(index=False))

    gain = normalisation_gain(items, rng)
    gain.to_csv(args.out_dir / "normalisation_gain.csv", index=False)
    print("\nDoes a normalisation correlate better with humans? (paired bootstrap, change in Spearman rho):")
    print(gain.round(3).to_string(index=False))

    differences = language_differences(items, rng)
    differences.to_csv(args.out_dir / "language_differences.csv", index=False)
    if differences.empty:
        print(f"\nNo language differences (needs ratings for {Q2_REFERENCE}).")
    else:
        print("\nQ2, correlation relative to " + Q2_REFERENCE + ":")
        print(differences.round(3).to_string(index=False))

    rater_disagreements(items, args.out_dir / "rater_disagreements.csv")

    wer_bands(items).to_csv(args.out_dir / "wer_bands.csv", index=False)
    print("\nMean human score by WER band (standard):")
    print(wer_bands(items).round(2).to_string(index=False))

    print("\nError candidates:")
    error_candidates(items, args.out_dir)
    plot_heatmap(table, args.out_dir / "heatmap.png")
    plot_scatter(items, args.out_dir / "scatter.png")
    plot_error_categories(args.out_dir)
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
