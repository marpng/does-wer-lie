"""Download FLEURS for the given languages and write one manifest per language.

FLEURS audio is already 16 kHz mono WAV, so no resampling is needed here.

Usage:
    python src/prepare_fleurs.py --langs en_us el_gr
    python src/prepare_fleurs.py --langs it_it de_de --splits test
    python src/prepare_fleurs.py --langs el_gr --splits train dev test   # train is ~2 GB

Output:
    data/raw/fleurs/{lang}/{split}.tsv            original metadata
    data/raw/fleurs/{lang}/audio/{split}/*.wav    extracted audio
    data/manifests/fleurs_{lang}.csv              shared manifest format
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
SAMPLE_RATE = 16_000

TSV_COLUMNS = ["sentence_id", "file_name", "reference_raw", "reference", "chars", "num_samples", "gender"]
MANIFEST_COLUMNS = [
    "id", "language", "dataset", "split", "audio_path", "reference", "reference_raw",
    "duration", "sentence_id", "gender", "dialect",
]


def fetch(lang: str, filename: str) -> Path:
    return Path(hf_hub_download(REPO, f"data/{lang}/{filename}", repo_type="dataset", local_dir=RAW_DIR.parent / "_hf"))


def read_tsv(path: Path) -> pd.DataFrame:
    # No header, and transcripts contain stray quote characters, so quoting must be off.
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
        archive.unlink()  # the extracted WAVs are all we need

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
    return df[MANIFEST_COLUMNS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--langs", nargs="+", required=True, help="FLEURS language codes, e.g. en_us el_gr it_it de_de")
    parser.add_argument("--splits", nargs="+", default=["dev", "test"], choices=["train", "dev", "test"],
                        help="splits whose AUDIO is downloaded (default: dev test)")
    parser.add_argument("--no-metadata-only-splits", action="store_true",
                        help="skip the transcript-only rows for splits whose audio was not requested")
    args = parser.parse_args()

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    for lang in args.langs:
        frames = []
        for split in ["train", "dev", "test"]:
            with_audio = split in args.splits
            if not with_audio and args.no_metadata_only_splits:
                continue
            print(f"[{lang}] {split}: {'audio + metadata' if with_audio else 'metadata only'}")
            frames.append(prepare_split(lang, split, with_audio))
        manifest = pd.concat(frames, ignore_index=True)
        out = MANIFEST_DIR / f"fleurs_{lang}.csv"
        manifest.to_csv(out, index=False)
        print(f"[{lang}] wrote {len(manifest)} rows -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
