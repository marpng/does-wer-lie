"""Transcribe an evaluation set with zero-shot Whisper (faster-whisper).

The language is always set explicitly. For Swiss German pass --whisper-language de.
Existing output files are never overwritten: a model/language pair that is already done is skipped.

Usage:
    python src/transcribe.py --manifest data/manifests/fleurs_en_us.csv
    python src/transcribe.py --manifest data/manifests/fleurs_el_gr.csv --models small medium
    python src/transcribe.py --manifest data/manifests/stt4sg_gsw.csv --whisper-language de

Output:
    results/baseline/{model}_{language}.csv
    columns: id, language, model, hypothesis, avg_logprob, no_speech_prob, seconds, compute_type
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

# Decoding settings, identical for every model and language.
DECODING = dict(
    task="transcribe",
    beam_size=5,
    condition_on_previous_text=False,  # utterances are independent sentences
    vad_filter=False,  # some recordings are very quiet; do not let a VAD drop them
    without_timestamps=True,
)


def load_eval_rows(manifest_path: Path) -> pd.DataFrame:
    manifest = pd.read_csv(manifest_path, keep_default_na=False)
    language = manifest["language"].iloc[0]
    eval_path = EVAL_DIR / f"{language}.csv"
    if not eval_path.exists():
        raise SystemExit(f"No evaluation set at {eval_path.relative_to(ROOT)}. Run src/select_eval_set.py first.")
    rows = manifest[manifest["id"].isin(pd.read_csv(eval_path)["id"])]
    if (rows["audio_path"] == "").any():
        raise SystemExit("Some evaluation utterances have no audio on disk.")
    return rows


def transcribe(model: WhisperModel, rows: pd.DataFrame, whisper_language: str, label: str) -> list[dict]:
    out = []
    for row in tqdm(rows.itertuples(), total=len(rows), desc=label):
        audio, sample_rate = sf.read(ROOT / row.audio_path, dtype="float32")
        if sample_rate != SAMPLE_RATE or audio.ndim != 1:
            raise SystemExit(f"{row.audio_path} is not 16 kHz mono; fix it in the manifest step.")
        start = time.perf_counter()
        segments, _ = model.transcribe(audio, language=whisper_language, **DECODING)
        segments = list(segments)
        out.append({
            "id": row.id,
            "hypothesis": " ".join(s.text.strip() for s in segments),
            "avg_logprob": min((s.avg_logprob for s in segments), default=float("nan")),
            "no_speech_prob": max((s.no_speech_prob for s in segments), default=float("nan")),
            "seconds": round(time.perf_counter() - start, 2),
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--models", nargs="+", default=MODELS)
    parser.add_argument("--whisper-language", help="language code passed to Whisper (default: the manifest language)")
    parser.add_argument("--device", default="auto", help="auto, cpu or cuda")
    parser.add_argument("--compute-type", default="default",
                        help="ctranslate2 compute type; 'default' keeps the model's own precision where the device allows")
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results" / "baseline")
    args = parser.parse_args()

    rows = load_eval_rows(args.manifest)
    language = rows["language"].iloc[0]
    whisper_language = args.whisper_language or language
    args.out_dir.mkdir(parents=True, exist_ok=True)

    for name in args.models:
        out_path = args.out_dir / f"{name}_{language}.csv"
        if out_path.exists():
            print(f"skip {out_path.name}: already exists")
            continue
        model = WhisperModel(name, device=args.device, compute_type=args.compute_type)
        result = pd.DataFrame(transcribe(model, rows, whisper_language, f"{name} {language}"))
        result.insert(1, "language", language)
        result.insert(2, "model", name)
        result["compute_type"] = model.model.compute_type
        result.to_csv(out_path, index=False)
        print(f"wrote {len(result)} rows -> {out_path.relative_to(ROOT)}")
        del model


if __name__ == "__main__":
    main()
