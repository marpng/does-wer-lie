"""Q3 automatic part: how much did fine-tuning lower WER and CER

    python src/analyse_finetuning.py --language el --tuned small-lora-el

Input is results/metrics/baseline_per_utterance.csv and finetuned_per_utterance.csv from compute_metrics.py
Gain is the relative error reduction 1 - error(tuned) / error(baseline) under each normalisation
If the reduction is larger under raw than under content normalisation part of the gain is cosmetic
Intervals are 95% paired bootstrap over sentences

Writes gains.csv and comparisons.csv to results/analysis_q3
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from analyse_ratings import N_BOOT, SEED

ROOT = Path(__file__).resolve().parents[1]
METRICS_DIR = ROOT / "results" / "metrics"
NORM_ORDER = ["raw", "standard", "standard_en", "content"]


def load_metrics(path: Path, model: str, language: str) -> pd.DataFrame:
    utt = pd.read_csv(path)
    utt = utt[(utt["model"] == model) & (utt["language"] == language)]
    if utt.empty:
        raise SystemExit(f"No rows for model '{model}' ({language}) in {path.name}.")
    return utt


def error_arrays(utt: pd.DataFrame, ids: list[str]) -> dict:
    """(metric, normalisation) to (errors, units) per utterance in the order of ids"""
    out = {}
    for normalisation, group in utt.groupby("normalisation"):
        group = group.set_index("id").reindex(ids)
        out[("wer", normalisation)] = (group["word_errors"].to_numpy(float), group["ref_words"].to_numpy(float))
        out[("cer", normalisation)] = (group["char_errors"].to_numpy(float), group["ref_chars"].to_numpy(float))
    return out


def draws(idx: np.ndarray, base: tuple, tuned: tuple) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Error rates of both models and the reduction for the data (row 0) and each bootstrap resample"""
    rate_base = base[0][idx].sum(axis=1) / base[1][idx].sum(axis=1)
    rate_tuned = tuned[0][idx].sum(axis=1) / tuned[1][idx].sum(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        relative = 1 - rate_tuned / rate_base
    return rate_base, rate_tuned, rate_base - rate_tuned, relative


def interval(x: np.ndarray) -> tuple[float, float, float]:
    low, high = np.nanpercentile(x[1:], [2.5, 97.5])
    return x[0], low, high


def resample_index(n: int, rng: np.random.Generator) -> np.ndarray:
    return np.vstack([np.arange(n), rng.integers(0, n, (N_BOOT, n))])


def gains_table(subset: str, idx: np.ndarray, base: dict, tuned: dict) -> tuple[pd.DataFrame, dict]:
    """One row per metric with error rates of both models and the reduction with bootstrap intervals"""
    rows, relative = [], {}
    for (metric, normalisation), b in base.items():
        t = tuned[(metric, normalisation)]
        rate_b, rate_t, absolute, rel = draws(idx, b, t)
        relative[(metric, normalisation)] = rel
        a, a_low, a_high = interval(absolute)
        r, r_low, r_high = interval(rel)
        rows.append({"subset": subset, "metric": metric, "normalisation": normalisation, "n": idx.shape[1],
                     "error_baseline": rate_b[0], "error_tuned": rate_t[0], "abs_gain": a, "abs_ci_low": a_low, "abs_ci_high": a_high,
                     "rel_reduction": r, "rel_ci_low": r_low, "rel_ci_high": r_high})
    return pd.DataFrame(rows), relative


def comparison(rows: list, subset: str, label: str, difference: np.ndarray) -> None:
    estimate, low, high = interval(difference)
    rows.append({"subset": subset, "comparison": label, "estimate": estimate, "ci_low": low, "ci_high": high,
                 "excludes_zero": bool(low > 0 or high < 0)})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", required=True)
    parser.add_argument("--baseline", default="small")
    parser.add_argument("--tuned", required=True, help="fine-tuned model label, e.g. small-lora-el")
    parser.add_argument("--baseline-metrics", type=Path, default=METRICS_DIR / "baseline_per_utterance.csv")
    parser.add_argument("--tuned-metrics", type=Path, default=METRICS_DIR / "finetuned_per_utterance.csv")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "analysis_q3")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    base_utt = load_metrics(args.baseline_metrics, args.baseline, args.language)
    tuned_utt = load_metrics(args.tuned_metrics, args.tuned, args.language)
    ids = sorted(set(base_utt["id"]) & set(tuned_utt["id"]))
    print(f"{len(ids)} evaluation utterances have both models ({args.baseline} vs {args.tuned}, {args.language})")
    base, tuned = error_arrays(base_utt, ids), error_arrays(tuned_utt, ids)

    comparisons = []
    idx = resample_index(len(ids), rng)
    gains, relative = gains_table("eval_set", idx, base, tuned)
    if ("wer", "content") in relative:
        comparison(comparisons, "eval_set", "WER reduction, content minus raw", relative[("wer", "content")] - relative[("wer", "raw")])
        comparison(comparisons, "eval_set", "WER reduction, content minus standard", relative[("wer", "content")] - relative[("wer", "standard")])

    for column, order in (("metric", ["wer", "cer"]), ("normalisation", NORM_ORDER)):
        gains[column] = pd.Categorical(gains[column], order, ordered=True)
    gains = gains.sort_values(["metric", "normalisation"], ignore_index=True).astype({"metric": str, "normalisation": str})
    gains.to_csv(args.out_dir / "gains.csv", index=False)
    comp = pd.DataFrame(comparisons)
    comp.to_csv(args.out_dir / "comparisons.csv", index=False)
    show = gains.assign(
        reduction=lambda g: (100 * g["rel_reduction"]).round(1).astype(str) + "% [" + (100 * g["rel_ci_low"]).round(1).astype(str) + ", " + (100 * g["rel_ci_high"]).round(1).astype(str) + "]")
    print("\nError reduction from fine-tuning (95% CI)\n" + show[["metric", "normalisation", "n", "error_baseline", "error_tuned", "reduction"]].round(3).to_string(index=False))
    print("\nDifference in relative reduction (95% CI)\n" + comp.round(3).to_string(index=False))
    print(f"\nWrote results to {args.out_dir}")


if __name__ == "__main__":
    main()
