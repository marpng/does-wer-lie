"""Which word errors did fine-tuning remove
Compares baseline and fine-tuned output on the same evaluation sentences

    python src/analyse_error_types.py --language el --baseline small --tuned small-lora-el

Aligns reference and output word by word with jiwer and sorts every error into one type

    punct_case   same word apart from punctuation or capitals
    accent       same word apart from accents
    number       digits on one side
    latin        Latin letters in the output where the reference has none
    inflection   same word start with a different ending
    spelling     small character difference with a different start
    other_word   a different word
    deleted      reference word missing in the output
    inserted     extra word in the output

Counts are per 100 reference words with a paired bootstrap CI of tuned minus baseline
Writes results/analysis_q3/error_types_{language}.csv and error_types_{language}_examples.csv
"""

import argparse
import re
import unicodedata
from pathlib import Path

import jiwer
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
TYPES = ["punct_case", "accent", "number", "latin", "inflection", "spelling", "other_word", "deleted", "inserted"]
N_BOOT = 2000
SEED = 42


def strip_punct(word: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFC", word) if not unicodedata.category(c).startswith(("P", "S"))).casefold()


def strip_accents(word: str) -> str:
    base = "".join(c for c in unicodedata.normalize("NFD", word) if unicodedata.category(c) != "Mn")
    return base.replace("ς", "σ")


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        new = [i]
        for j, cb in enumerate(b, 1):
            new.append(min(row[j] + 1, new[j - 1] + 1, row[j - 1] + (ca != cb)))
        row = new
    return row[-1]


def classify(ref: str, hyp: str) -> str:
    r, h = strip_punct(ref), strip_punct(hyp)
    if r == h:
        return "punct_case"
    if strip_accents(r) == strip_accents(h):
        return "accent"
    if re.search(r"\d", r + h):
        return "number"
    if re.search(r"[a-z]", h) and not re.search(r"[a-z]", r):
        return "latin"
    r, h = strip_accents(r), strip_accents(h)
    prefix = len(next((r[:k] for k in range(min(len(r), len(h)), -1, -1) if r[:k] == h[:k]), ""))
    if prefix >= 3 and prefix >= 0.5 * min(len(r), len(h)):
        return "inflection"
    if edit_distance(r, h) <= 2:
        return "spelling"
    return "other_word"


def count_errors(ref: str, hyp: str) -> tuple[dict, int, list]:
    counts = dict.fromkeys(TYPES, 0)
    examples = []
    out = jiwer.process_words(ref, hyp)
    for chunk in out.alignments[0]:
        r_words = out.references[0][chunk.ref_start_idx:chunk.ref_end_idx]
        h_words = out.hypotheses[0][chunk.hyp_start_idx:chunk.hyp_end_idx]
        if chunk.type == "delete":
            counts["deleted"] += len(r_words)
            examples += [("deleted", w, "") for w in r_words]
        elif chunk.type == "insert":
            counts["inserted"] += len(h_words)
            examples += [("inserted", "", w) for w in h_words]
        elif chunk.type == "substitute":
            for r, h in zip(r_words, h_words):
                kind = classify(r, h)
                counts[kind] += 1
                examples.append((kind, r, h))
    return counts, len(ref.split()), examples


def analyse(refs: pd.Series, hyps: pd.Series) -> tuple[np.ndarray, np.ndarray, list]:
    rows, words, examples = [], [], []
    for ref, hyp in zip(refs, hyps):
        counts, n, ex = count_errors(ref, hyp if hyp.strip() else "<empty>")
        rows.append([counts[t] for t in TYPES])
        words.append(n)
        examples.append(ex)
    return np.array(rows), np.array(words), examples


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", default="el")
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--baseline", default="small")
    parser.add_argument("--tuned", required=True)
    parser.add_argument("--baseline-dir", type=Path, default=ROOT / "results" / "baseline")
    parser.add_argument("--tuned-dir", type=Path, default=ROOT / "results" / "finetuned")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "analysis_q3")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest or ROOT / "data" / "manifests" / f"fleurs_{args.language}_gr.csv", keep_default_na=False)
    base = pd.read_csv(args.baseline_dir / f"{args.baseline}_{args.language}.csv", keep_default_na=False)[["id", "hypothesis"]]
    tuned = pd.read_csv(args.tuned_dir / f"{args.tuned}_{args.language}.csv", keep_default_na=False)[["id", "hypothesis"]]
    df = (manifest[["id", "reference_raw"]].merge(base, on="id").merge(tuned, on="id", suffixes=("_base", "_tuned")))
    df["reference_raw"] = df["reference_raw"].str.split().str.join(" ")
    for col in ("hypothesis_base", "hypothesis_tuned"):
        df[col] = df[col].str.split().str.join(" ")

    b_counts, words, b_ex = analyse(df["reference_raw"], df["hypothesis_base"])
    t_counts, _, t_ex = analyse(df["reference_raw"], df["hypothesis_tuned"])
    total_words = words.sum()
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(df), size=(N_BOOT, len(df)))

    table = []
    for k, name in enumerate(TYPES + ["all"]):
        b = b_counts.sum(axis=1) if name == "all" else b_counts[:, k]
        t = t_counts.sum(axis=1) if name == "all" else t_counts[:, k]
        diff = (t[idx].sum(axis=1) - b[idx].sum(axis=1)) / words[idx].sum(axis=1) * 100
        table.append({"error_type": name, "baseline_per_100_words": b.sum() / total_words * 100,
                      "tuned_per_100_words": t.sum() / total_words * 100,
                      "change_per_100_words": (t.sum() - b.sum()) / total_words * 100,
                      "ci_low": np.percentile(diff, 2.5), "ci_high": np.percentile(diff, 97.5),
                      "relative_change_%": (t.sum() / b.sum() - 1) * 100 if b.sum() else np.nan})
    table = pd.DataFrame(table).round(2)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.out_dir / f"error_types_{args.language}.csv", index=False)
    print(f"{len(df)} sentences, {total_words} reference words, errors by type (raw text, punctuation and case count):\n")
    print(table.to_string(index=False))

    rows = []
    for i in range(len(df)):
        for model, ex in (("baseline", b_ex[i]), ("tuned", t_ex[i])):
            rows += [{"id": df["id"][i], "model": model, "type": k, "reference": r, "output": h} for k, r, h in ex]
    pd.DataFrame(rows).to_csv(args.out_dir / f"error_types_{args.language}_examples.csv", index=False)

    # words the baseline got wrong and the tuned model got right in the same sentence
    print("\nExamples of reference words the tuned model fixed (baseline error not repeated), by type:")
    fixed = {}
    for i in range(len(df)):
        tuned_refs = {r for _, r, _ in t_ex[i]}
        for kind, r, h in b_ex[i]:
            if r and r not in tuned_refs:
                fixed.setdefault(kind, []).append(f"{r} <- {h}")
    for kind in TYPES:
        sample = fixed.get(kind, [])
        print(f"  {kind:11s} {len(sample):4d}  " + " | ".join(sample[:6]))


if __name__ == "__main__":
    main()
