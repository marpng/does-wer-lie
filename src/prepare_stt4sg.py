"""Build the STT4SG-350 manifest and the Swiss German evaluation set

    python src/prepare_stt4sg.py --stt4sg-dir path/to/STT4SG-350 --test-size 5000 --train-size 5000 --dev-size 300

The corpus is licensed (META-SHARE NonCommercial NoRedistribution) and is not in the repo: request it from SwissNLP
--test-size 5000 stops at all 3515 unique test sentences and always contains the 350-sentence subset the ratings come from
--train-size and --dev-size draw fixed subsets for fine-tuning that share no sentence with the test subset
Only those subsets are converted to 16 kHz mono WAV in data/raw/stt4sg/, see src/swiss_german.py
Writes data/manifests/stt4sg_gsw.csv, then continue as for FLEURS:

    python src/select_eval_set.py --manifest data/manifests/stt4sg_gsw.csv
    python src/transcribe.py --manifest data/manifests/stt4sg_gsw.csv --whisper-language de
"""

import argparse
import hashlib
from pathlib import Path

import pandas as pd

from swiss_german import ROOT, TEST_SIZE, build_manifest, durations, get, normalise_gender, read_tsv, sentence_ids

DEFAULT_DIR = ROOT / "data" / "STT4SG-350"  # the corpus as downloaded from SwissNLP (not in the repo, see README)
# split -> (TSV, folder the clip paths are relative to)
SPLITS = {
    "test": ("test.tsv", "clips__test"),
    "dev": ("valid.tsv", "clips__train_valid"),
    "train": ("train_balanced.tsv", "clips__train_valid"),  # --train-tsv train_all.tsv for the full set
}


def resolve(corpus_dir: Path, clips_dir: Path, raw: str) -> Path | None:
    path = corpus_dir / raw if raw.startswith("clips__") else clips_dir / raw
    if path.exists():
        return path
    for ext in (".flac", ".mp3", ".wav"):
        candidate = path.with_name(path.name + ext)
        if candidate.exists():
            return candidate
    return None


def read_split(corpus_dir: Path, split: str, tsv_name: str, clips_folder: str) -> pd.DataFrame:
    raw = read_tsv(corpus_dir / tsv_name)
    paths = get(raw, ["path", "clip_path", "file_name"], "audio path", required=True)
    texts = get(raw, ["sentence", "text", "transcript"], "transcript", required=True)
    files = pd.Series([resolve(corpus_dir, corpus_dir / clips_folder, p) for p in paths], index=raw.index, dtype=object)
    clip_ids = get(raw, ["clip_id", "id"], "clip id")
    clip_ids = clip_ids.where(clip_ids != "", paths.map(lambda p: Path(p).name.split(".")[0]))
    if clip_ids.duplicated().any():
        # File names repeat across speakers (same sentence): build the ID from the full path instead.
        clip_ids = paths.map(lambda p: hashlib.sha1(p.encode()).hexdigest()[:16])

    df = pd.DataFrame({
        "clip_id": clip_ids,
        "split": split,
        "audio_file": files,
        "reference_raw": texts,
        "duration": durations(files, get(raw, ["duration"], "duration")),
        "sentence_id": sentence_ids(get(raw, ["sentence_id"], "sentence id"), texts),
        "gender": normalise_gender(get(raw, ["gender"], "gender")),
        "dialect": get(raw, ["dialect_region", "dialect", "canton"], "dialect"),
    })
    missing = int(files.isna().sum())
    print(f"[stt4sg] {tsv_name}: {len(df)} rows" + (f", WARNING {missing} audio files not found" if missing else ""))
    return df


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stt4sg-dir", type=Path, default=DEFAULT_DIR, help="STT4SG-350 root folder")
    parser.add_argument("--test-size", type=int, default=TEST_SIZE, help="recordings in the test subset")
    parser.add_argument("--train-tsv", default=SPLITS["train"][0], help="train_balanced.tsv or train_all.tsv")
    parser.add_argument("--language-tag", default="gsw",
                        help="language code in IDs, file names and results (gsw5k for the 5,000-clip version)")
    parser.add_argument("--train-size", type=int, default=0, help="train recordings converted for fine-tuning")
    parser.add_argument("--dev-size", type=int, default=0, help="dev recordings converted for fine-tuning")
    parser.add_argument("--link-audio", nargs="*", default=[], choices=["train", "dev"],
                        help="splits whose rows point at the original audio (test always gets the converted subset)")
    args = parser.parse_args()

    corpus_dir = args.stt4sg_dir.resolve()
    if not corpus_dir.exists():
        raise SystemExit(f"STT4SG-350 folder not found: {corpus_dir}")

    splits = dict(SPLITS, train=(args.train_tsv, SPLITS["train"][1]))
    frames = []
    for split, (tsv_name, clips_folder) in splits.items():
        if not (corpus_dir / tsv_name).exists():
            print(f"[stt4sg] {tsv_name} not found, skipping {split}")
            continue
        frames.append(read_split(corpus_dir, split, tsv_name, clips_folder))
    build_manifest(pd.concat(frames, ignore_index=True), "stt4sg", args.test_size, args.link_audio,
                   {"train": args.train_size, "dev": args.dev_size}, args.language_tag)


if __name__ == "__main__":
    main()
