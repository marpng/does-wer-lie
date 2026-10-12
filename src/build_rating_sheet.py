"""Build an offline rating sheet (one HTML page plus audio) for one language.

Raters see audio, reference and model output. Model name and WER are never written to the sheet.
Audio is peak-normalised so that quiet recordings are audible.

Usage:
    python src/build_rating_sheet.py --language el
    python src/build_rating_sheet.py --language en --overlap-only        # the shared agreement set
    python src/build_rating_sheet.py --language el --items ratings/pilot/rated_set_ids.csv --name pilot_el

Output (not committed: contains audio and reference text):
    ratings/sheets/{name}/index.html
    ratings/sheets/{name}/audio/{item_id}.wav

Open index.html in a browser, rate, then press "Download CSV" and save the file to ratings/scores/.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
MANIFEST_DIR = ROOT / "data" / "manifests"
BASELINE_DIR = ROOT / "results" / "baseline"
TEMPLATE = Path(__file__).with_name("rating_sheet_template.html")
PEAK = 0.9


def load_items(items_path: Path, language: str, overlap_only: bool) -> pd.DataFrame:
    items = pd.read_csv(items_path, dtype={"sentence_id": str})
    items = items[items["language"] == language]
    if overlap_only:
        items = items[items["overlap"]]
    if items.empty:
        raise SystemExit(f"No items for language '{language}' in {items_path}.")

    manifests = pd.concat([pd.read_csv(p, keep_default_na=False) for p in MANIFEST_DIR.glob("*.csv")])
    items = items.merge(manifests[["id", "audio_path", "reference_raw"]], on="id", validate="many_to_one")
    hyps = pd.concat([pd.read_csv(p, keep_default_na=False) for p in BASELINE_DIR.glob(f"*_{language}.csv")])
    return items.merge(hyps[["id", "model", "hypothesis"]], on=["id", "model"], validate="one_to_one")


def export_audio(source: Path, target: Path) -> None:
    audio, sample_rate = sf.read(source, dtype="float32")
    peak = float(np.abs(audio).max())
    if peak > 0:
        audio = audio / peak * PEAK
    sf.write(target, audio, sample_rate, subtype="PCM_16")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--language", required=True)
    parser.add_argument("--items", type=Path, default=ROOT / "ratings" / "rated_set_ids.csv")
    parser.add_argument("--overlap-only", action="store_true")
    parser.add_argument("--name", help="sheet folder name (default: the language, or {language}_overlap)")
    args = parser.parse_args()

    items = load_items(args.items, args.language, args.overlap_only)
    name = args.name or (f"{args.language}_overlap" if args.overlap_only else args.language)
    sheet_dir = ROOT / "ratings" / "sheets" / name
    (sheet_dir / "audio").mkdir(parents=True, exist_ok=True)

    payload = []
    for item in items.sort_values("item_id").itertuples():
        export_audio(ROOT / item.audio_path, sheet_dir / "audio" / f"{item.item_id}.wav")
        payload.append({"item_id": item.item_id, "reference": item.reference_raw.strip(), "hypothesis": item.hypothesis.strip()})

    data = json.dumps({"sheet": name, "language": args.language, "items": payload}, ensure_ascii=False).replace("</", "<\\/")
    (sheet_dir / "index.html").write_text(TEMPLATE.read_text(encoding="utf-8").replace("__SHEET_DATA__", data), encoding="utf-8")
    print(f"wrote {len(payload)} items -> {(sheet_dir / 'index.html').relative_to(ROOT)}")


if __name__ == "__main__":
    main()
