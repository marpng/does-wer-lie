"""Pick the evaluation set: one recording per unique test sentence, gender balanced where possible

    python src/select_eval_set.py --manifest data/manifests/fleurs_en_us.csv

Writes eval_sets/{language}.csv with id and sentence_id
Keep this file fixed once chosen
"""

import argparse
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "eval_sets"
SEED = 42


def select(manifest: pd.DataFrame) -> pd.DataFrame:
    shuffled = manifest.sample(frac=1, random_state=SEED)
    counts = {g: 0 for g in shuffled["gender"].unique()}
    chosen = []
    # Single gender sentences first since they leave no choice
    groups = sorted(shuffled.groupby("sentence_id", sort=False), key=lambda kv: kv[1]["gender"].nunique())
    for _, recordings in groups:
        pick = min(recordings.itertuples(), key=lambda r: counts[r.gender])
        counts[pick.gender] += 1
        chosen.append(pick.id)
    return manifest[manifest["id"].isin(chosen)][["id", "sentence_id"]].sort_values("id")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--split", default="test")
    parser.add_argument("--force", action="store_true", help="replace an existing evaluation set")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, dtype={"sentence_id": str}, keep_default_na=False)
    manifest = manifest[(manifest["split"] == args.split) & (manifest["audio_path"] != "")].copy()
    if manifest.empty:
        raise SystemExit(f"No '{args.split}' rows with audio in {args.manifest}.")
    # Without sentence IDs every recording is its own sentence
    no_sentence = manifest["sentence_id"] == ""
    manifest.loc[no_sentence, "sentence_id"] = manifest.loc[no_sentence, "id"]
    language = manifest["language"].iloc[0]

    out = OUT_DIR / f"{language}.csv"
    if out.exists() and not args.force:
        raise SystemExit(f"{out.relative_to(ROOT)} already exists; use --force to replace it.")

    selected = select(manifest)
    OUT_DIR.mkdir(exist_ok=True)
    selected.to_csv(out, index=False)

    info = manifest[manifest["id"].isin(selected["id"])]
    print(f"[{language}] {len(selected)} utterances, {info['duration'].sum() / 60:.1f} min, "
          f"gender {info['gender'].value_counts().to_dict()} -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
