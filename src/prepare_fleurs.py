"""Download FLEURS and write one manifest per language

    python src/prepare_fleurs.py --langs en_us el_gr
    python src/prepare_fleurs.py --langs el_gr --splits train dev test

Writes data/raw/fleurs/{lang}/ and data/manifests/fleurs_{lang}.csv
Splits not in --splits get transcript rows with an empty audio_path
"""

import argparse
import csv
import tarfile
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

REPO = "google/fleurs"
ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "fleurs"
MANIFEST_DIR = ROOT / "data" / "manifests"
SAMPLE_RATE = 16_000  # FLEURS audio is 16 kHz mono

TSV_COLUMNS = ["sentence_id", "file_name", "reference_raw", "reference", "chars", "num_samples", "gender"]
MANIFEST_COLUMNS = [
    "id", "language", "dataset", "split", "audio_path", "reference", "reference_raw",
    "duration", "sentence_id", "gender", "dialect",
]


def fetch(lang: str, filename: str) -> Path:
    return Path(hf_hub_download(REPO, f"data/{lang}/{filename}", repo_type="dataset", local_dir=RAW_DIR.parent / "_hf"))


def read_tsv(path: Path) -> pd.DataFrame:
    # No header and transcripts contain stray quote characters
    return pd.read_csv(path, sep="\t", header=None, names=TSV_COLUMNS, quoting=csv.QUOTE_NONE, dtype=str)


def prepare_split(lang: str, split: str, with_audio: bool) -> pd.DataFrame:
    lang_dir = RAW_DIR / lang
    lang_dir.mkdir(parents=True, exist_ok=True)

    tsv_path = lang_dir / f"{split}.tsv"
    if not tsv_path.exists():
        tsv_path.write_bytes(fetch(lang, f"{split}.tsv").read_bytes())
    df = read_tsv(tsv_path)

    audio_dir = lang_dir / "audio" / split
    if with_audio and not audio_dir.exists():
        archive = fetch(lang, f"audio/{split}.tar.gz")
        with tarfile.open(archive) as tar:
            tar.extractall(lang_dir / "audio", filter="data")
        archive.unlink()

    df["id"] = f"fleurs_{lang}_{split}_" + df["file_name"].str.removesuffix(".wav")
    df["language"] = lang.split("_")[0]
    df["dataset"] = "fleurs"
    df["split"] = split
    df["audio_path"] = [
        str((audio_dir / f).relative_to(ROOT)) if (audio_dir / f).exists() else "" for f in df["file_name"]
    ]
    df["duration"] = (df["num_samples"].astype(int) / SAMPLE_RATE).round(3)
    df["gender"] = df["gender"].str.lower()
    df["dialect"] = ""

    missing = int((df["audio_path"] == "").sum())
    if with_audio and missing:
        print(f"[{lang}] WARNING {split}: {missing} of {len(df)} audio files missing. "
              f"If the download was interrupted, delete {audio_dir.relative_to(ROOT)} and rerun.")
    return df[MANIFEST_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langs", nargs="+", required=True, help="e.g. en_us el_gr it_it de_de")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["train", "dev", "test"],
                        help="splits whose audio is downloaded")
    args = parser.parse_args()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for lang in args.langs:
        manifest = pd.concat(
            [prepare_split(lang, split, split in args.splits) for split in ["train", "dev", "test"]],
            ignore_index=True,
        )
        out = MANIFEST_DIR / f"fleurs_{lang}.csv"
        manifest.to_csv(out, index=False)
        print(f"[{lang}] {len(manifest)} rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
