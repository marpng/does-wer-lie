"""Transcribe the evaluation set with Whisper (faster-whisper)

    python src/transcribe.py --manifest data/manifests/fleurs_en_us.csv
    python src/transcribe.py --manifest data/manifests/stt4sg_gsw.csv --whisper-language de
    python src/transcribe.py --manifest data/manifests/fleurs_el_gr.csv --models small-lora-el=models/small-lora-el/ct2 \
        --out-dir results/finetuned

Writes {out-dir}/{model}_{language}.csv and skips files that already exist
A model is a Whisper size name or label=path to a converted fine-tuned model
--only-ids takes a CSV with an id column to transcribe only those utterances
"""

import argparse
import time
from pathlib import Path

import pandas as pd
import soundfile as sf
from faster_whisper import WhisperModel
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
EVAL_DIR = ROOT / "eval_sets"
MODELS = ["small", "medium", "large-v3"]
SAMPLE_RATE = 16_000

DECODING = dict(
    task="transcribe",
    beam_size=5,
    condition_on_previous_text=False,
    vad_filter=False,  # a VAD could drop very quiet recordings
    without_timestamps=True,
)


def load_eval_rows(manifest_path: Path, only_ids: Path | None = None) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    language = manifest["language"].iloc[0]
    eval_path = EVAL_DIR / f"{language}.csv"
    if not eval_path.exists():
        raise SystemExit(f"No evaluation set at {eval_path.relative_to(ROOT)}. Run src/select_eval_set.py first.")
    eval_ids = pd.read_csv(eval_path)["id"]
    rows = manifest[manifest["id"].isin(eval_ids)]
    if len(rows) != len(eval_ids) or (rows["audio_path"] == "").any():
        raise SystemExit("Some evaluation utterances are missing from the manifest or have no audio.")
    if only_ids:
        wanted = set(pd.read_csv(only_ids)["id"])
        rows = rows[rows["id"].isin(wanted)]
        if len(rows) != len(wanted):
            raise SystemExit(f"{len(wanted) - len(rows)} ids in {only_ids.name} are not in the evaluation set of this manifest.")
    return rows


def transcribe(model: WhisperModel, rows: pd.DataFrame, whisper_language: str, label: str) -> list[dict]:
    out = []
    for row in tqdm(rows.itertuples(), total=len(rows), desc=label):
        audio, sample_rate = sf.read(ROOT / row.audio_path, dtype="float32")
        if sample_rate != SAMPLE_RATE or audio.ndim != 1:
            raise SystemExit(f"{row.audio_path} is not 16 kHz mono.")
        start = time.perf_counter()
        segments, _ = model.transcribe(audio, language=whisper_language, **DECODING)
        segments = list(segments)
        out.append({
            "id": row.id,
            "hypothesis": " ".join(s.text.strip() for s in segments),
            # worst segment and utterances under 30 s have one
            "avg_logprob": min((s.avg_logprob for s in segments), default=float("nan")),
            "no_speech_prob": max((s.no_speech_prob for s in segments), default=float("nan")),
            "seconds": round(time.perf_counter() - start, 2),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=MODELS, help="size names or label=path for a converted model")
    parser.add_argument("--only-ids", type=Path, help="CSV with an id column to transcribe only those utterances")
    parser.add_argument("--whisper-language", help="default is the manifest language")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--compute-type", default="default", help="ctranslate2 compute type")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "baseline")
    args = parser.parse_args()

    rows = load_eval_rows(args.manifest, args.only_ids)
    language = rows["language"].iloc[0]
    whisper_language = args.whisper_language or language
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for spec in args.models:
        name, _, path = spec.partition("=")
        out_path = args.out_dir / f"{name}_{language}.csv"
        if out_path.exists():
            print(f"skip {out_path.name}: already exists")
            continue
        model = WhisperModel(str(ROOT / path) if path else name, device=args.device, compute_type=args.compute_type)
        result = pd.DataFrame(transcribe(model, rows, whisper_language, f"{name} {language}"))
        result.insert(1, "language", language)
        result.insert(2, "model", name)
        result["compute_type"] = model.model.compute_type
        result.to_csv(out_path, index=False)
        print(f"wrote {len(result)} rows -> {out_path}")
        del model


if __name__ == "__main__":
    main()
