"""Shared steps for the Swiss German corpus STT4SG-350, used by prepare_stt4sg.py

The STT4SG-350 test split reads each of its 3515 sentences once in each of the 7 dialect regions (24605 recordings)
The evaluation set keeps one recording per sentence:
    first a 350-sentence subset (50 per region, female and male alternating), the one the human ratings come from
    then all remaining sentences with the regions taking turns, so a larger set always contains the 350
Only recordings of 1 to 30 s with a non-empty reference are used, seed 42
Fixed train and dev subsets for fine-tuning are drawn the same way and share no sentence with the test subset
Only the chosen subsets are converted to 16 kHz mono WAV in data/raw/stt4sg/{split}/
All other rows stay in the manifest with an empty audio_path, so the FLEURS scripts run unchanged
"""

import hashlib
import os
from collections import deque
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
RAW_DIR = ROOT / "data" / "raw"

LANGUAGE = "gsw"
SAMPLE_RATE = 16_000
SEED = 42
TEST_SIZE = 350          # FLEURS English: 350 unique test sentences, Greek: 333
MIN_SECONDS, MAX_SECONDS = 1.0, 30.0

MANIFEST_COLUMNS = [
    "id", "language", "dataset", "split", "audio_path", "reference", "reference_raw",
    "duration", "sentence_id", "gender", "dialect",
]
GENDERS = {"m": "male", "male": "male", "mann": "male", "f": "female", "w": "female", "female": "female", "frau": "female"}


def read_tsv(path: Path) -> pd.DataFrame:
    # keep_default_na=False: missing values stay "" instead of NaN (str(NaN) would give "nan").
    return pd.read_csv(path, sep="\t", dtype=str, keep_default_na=False)


def column(df: pd.DataFrame, candidates: list[str], what: str, required: bool = True) -> str | None:
    for name in candidates:
        if name in df.columns:
            return name
    if required:
        raise SystemExit(f"No column for {what} (tried {candidates}). Columns in the TSV: {list(df.columns)}")
    return None


def get(df: pd.DataFrame, candidates: list[str], what: str, required: bool = False) -> pd.Series:
    name = column(df, candidates, what, required)
    return df[name].str.strip() if name else pd.Series("", index=df.index)


def sentence_ids(ids: pd.Series, texts: pd.Series) -> pd.Series:
    """Corpus sentence ID where given; otherwise derived from the text, so repeated sentences still share an ID."""
    from_text = texts.str.strip().str.lower().map(lambda t: "txt_" + hashlib.md5(t.encode()).hexdigest()[:12])
    return ids.where(ids != "", from_text)


def normalise_gender(values: pd.Series) -> pd.Series:
    lower = values.str.strip().str.lower()
    return lower.map(GENDERS).fillna(lower)


def durations(files: pd.Series, from_tsv: pd.Series) -> pd.Series:
    """Duration column of the TSV where present, else read from the file header."""
    out = pd.to_numeric(from_tsv, errors="coerce")
    # Some corpora store milliseconds.
    if out.notna().any() and out.median() > 1000:
        out = out / 1000
    for i in out.index[out.isna() & files.notna()]:
        try:
            out.at[i] = sf.info(str(files.at[i])).duration
        except Exception:
            pass
    return out.round(3)


def pick_subset(test: pd.DataFrame, size: int, seed: int = SEED, exclude: set | None = None) -> list:
    """Row labels of a subset (see the module docstring for the rules). Sentences in exclude are never picked."""
    ok = (
        test["audio_file"].notna()
        & test["duration"].between(MIN_SECONDS, MAX_SECONDS)
        & (test["reference_raw"].str.strip() != "")
    )
    pool = test[ok].sample(frac=1, random_state=seed)
    dialects = sorted(pool["dialect"].unique())
    rng = np.random.default_rng(seed)
    quota = {d: size // len(dialects) for d in dialects}
    for d in rng.choice(dialects, size % len(dialects), replace=False):
        quota[d] += 1

    used, chosen = set(exclude or ()), []

    def take(rows: pd.DataFrame, n: int) -> None:
        known = [deque(r.index) for g, r in rows.groupby("gender", sort=True) if g in ("female", "male")]
        other = deque(rows.index[~rows["gender"].isin(["female", "male"])])
        picked = 0
        # Alternate female / male; unknown gender only once both are used up.
        while picked < n and any(known):
            for queue in known:
                while queue and picked < n:
                    i = queue.popleft()
                    if pool.at[i, "sentence_id"] not in used:
                        used.add(pool.at[i, "sentence_id"])
                        chosen.append(i)
                        picked += 1
                        break
        while other and picked < n:
            i = other.popleft()
            if pool.at[i, "sentence_id"] not in used:
                used.add(pool.at[i, "sentence_id"])
                chosen.append(i)
                picked += 1

    for d in dialects:
        take(pool[pool["dialect"] == d], quota[d])
    if len(chosen) < size:  # a region ran out of sentences: top up from the others
        take(pool[~pool.index.isin(chosen)], size - len(chosen))
    if len(chosen) < size:
        print(f"WARNING: only {len(chosen)} usable unique sentences, fewer than the {size} asked for.")
    return chosen


def convert(source: Path, target: Path) -> float:
    """Downmix and resample to 16 kHz mono WAV (polyphase filter, no aliasing). Returns seconds."""
    if not target.exists():
        from math import gcd

        from scipy.signal import resample_poly

        audio, sample_rate = sf.read(str(source), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if sample_rate != SAMPLE_RATE:
            g = gcd(sample_rate, SAMPLE_RATE)
            audio = resample_poly(audio, SAMPLE_RATE // g, sample_rate // g).astype("float32")
        target.parent.mkdir(parents=True, exist_ok=True)
        sf.write(target, audio, SAMPLE_RATE, subtype="PCM_16")
    return round(sf.info(str(target)).frames / SAMPLE_RATE, 3)


def convert_rows(df: pd.DataFrame, rows: list, dataset: str, split: str) -> None:
    audio_dir = RAW_DIR / dataset / split
    print(f"[{dataset}] converting {len(rows)} {split} clips to 16 kHz mono WAV in {audio_dir.relative_to(ROOT)} ...")
    step = max(50, len(rows) // 10)
    for n, i in enumerate(rows, 1):
        target = audio_dir / f"{df.at[i, 'clip_id']}.wav"
        df.at[i, "duration"] = convert(df.at[i, "audio_file"], target)
        df.at[i, "audio_path"] = str(target.relative_to(ROOT))
        if n % step == 0:
            print(f"  {n}/{len(rows)}")


def describe(chosen: pd.DataFrame, dataset: str, split: str) -> None:
    print(f"[{dataset}] {split} subset: {len(chosen)} recordings, {chosen['sentence_id'].nunique()} sentences, "
          f"{chosen['duration'].sum() / 3600:.2f} h")
    print(f"  gender  {chosen['gender'].replace('', 'unknown').value_counts().to_dict()}")
    print(f"  dialect {chosen['dialect'].replace('', 'unknown').value_counts().sort_index().to_dict()}")


def pick_spread(rows: pd.DataFrame, size: int, exclude: set, seed: int = SEED) -> list:
    """Like pick_subset, but the dialect regions take turns (one clip each per round, female/male alternating
    within a region). Used where regions compete for the same sentences: in the STT4SG test split every
    sentence is read once in each of the 7 regions, so filling one region after the other starves the last ones."""
    ok = (
        rows["audio_file"].notna()
        & rows["duration"].between(MIN_SECONDS, MAX_SECONDS)
        & (rows["reference_raw"].str.strip() != "")
    )
    pool = rows[ok].sample(frac=1, random_state=seed)
    queues = {}
    for dialect, d in pool.groupby("dialect", sort=True):
        groups = list(d.groupby("gender", sort=True))
        known = [deque(g.index) for name, g in groups if name in ("female", "male")]
        other = [deque(g.index) for name, g in groups if name not in ("female", "male")]
        merged = deque()  # interleave genders: f, m, f, m, ... (unknown last)
        while any(known):
            for q in known:
                if q:
                    merged.append(q.popleft())
        for q in other:
            merged.extend(q)
        queues[dialect] = merged

    used, chosen = set(exclude), []
    dialects = sorted(queues)
    start = 0
    while len(chosen) < size and any(queues.values()):
        order = dialects[start:] + dialects[:start]  # rotate who picks first
        start = (start + 1) % len(dialects)
        for dialect in order:
            queue = queues[dialect]
            while queue:
                i = queue.popleft()
                if pool.at[i, "sentence_id"] not in used:
                    used.add(pool.at[i, "sentence_id"])
                    chosen.append(i)
                    break
            if len(chosen) == size:
                break
    if len(chosen) < size:
        print(f"NOTE: only {len(chosen)} more unique sentences available ({len(chosen) + len(exclude)} in total); "
              f"one recording per sentence, so the set stops there.")
    return chosen


def pick_test(test: pd.DataFrame, size: int) -> list:
    """The test subset. Sets larger than TEST_SIZE are built around the TEST_SIZE set, so a 5,000-clip set
    always contains the original 350 clips (and with them every rated item)."""
    base = pick_subset(test, min(size, TEST_SIZE))
    if size <= TEST_SIZE:
        return base
    return base + pick_spread(test, size - len(base), exclude=set(test.loc[base, "sentence_id"]))


def build_manifest(df: pd.DataFrame, dataset: str, test_size: int, link_splits: list[str],
                   subset_sizes: dict[str, int] | None = None, language: str = LANGUAGE) -> Path:
    """df: one row per recording with clip_id, split, audio_file (Path or None), reference_raw,
    duration, sentence_id, gender, dialect. Writes data/manifests/{dataset}_gsw.csv.

    subset_sizes, e.g. {"train": 5000, "dev": 300}: like the test subset, a fixed sample of those splits
    is converted to 16 kHz WAV (for fine-tuning). Never contains a test-subset sentence."""
    df = df.copy()
    no_duration = df["duration"].isna()
    if no_duration.any():
        print(f"[{dataset}] dropping {int(no_duration.sum())} rows with neither audio nor a duration in the TSV")
        df = df[~no_duration]

    df["language"] = language
    df["dataset"] = dataset
    df["id"] = f"{dataset}_{language}_" + df["split"] + "_" + df["clip_id"]
    if df["id"].duplicated().any():
        raise SystemExit(f"[{dataset}] duplicate IDs, e.g. {df.loc[df['id'].duplicated(), 'id'].head(3).tolist()}")
    # The corpus has no separate normalised transcript; compute_metrics.py normalises reference_raw itself.
    df["reference"] = df["reference_raw"]
    df["audio_path"] = ""

    # Other splits only point at the original files when asked (e.g. train audio for fine-tuning later).
    for split in link_splits:
        rows = (df["split"] == split) & df["audio_file"].notna()
        df.loc[rows, "audio_path"] = [os.path.relpath(f, ROOT) for f in df.loc[rows, "audio_file"]]

    test = df[df["split"] == "test"]
    if test.empty:
        raise SystemExit(f"[{dataset}] no test rows.")
    subset = pick_test(test, test_size)  # picked first and unchanged, so eval_sets/gsw.csv stays valid
    convert_rows(df, subset, dataset, "test")

    used = set(df.loc[subset, "sentence_id"])
    extra = {}
    for split, size in (subset_sizes or {}).items():
        if size <= 0:
            continue
        rows = df[df["split"] == split]
        if rows.empty:
            raise SystemExit(f"[{dataset}] no {split} rows for a {split} subset.")
        df.loc[rows.index, "audio_path"] = ""  # only the subset has audio, even if --link-audio named this split
        extra[split] = pick_subset(rows, size, exclude=used)
        used |= set(df.loc[extra[split], "sentence_id"])
        convert_rows(df, extra[split], dataset, split)

    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)
    out = MANIFEST_DIR / f"{dataset}_{language}.csv"
    df[MANIFEST_COLUMNS].to_csv(out, index=False)

    print(f"[{dataset}] {len(df)} rows -> {out.relative_to(ROOT)}")
    print(df.groupby("split").agg(rows=("id", "size"), hours=("duration", lambda s: s.sum() / 3600),
                                  with_audio=("audio_path", lambda s: int((s != "").sum()))).round(2).to_string())
    describe(df.loc[subset], dataset, "test")
    for split, rows in extra.items():
        describe(df.loc[rows], dataset, split)
    return out
