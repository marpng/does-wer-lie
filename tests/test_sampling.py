import pandas as pd

from sample_rated_set import BAND_LABELS, band
from select_eval_set import select


def test_band_edges():
    wer = pd.Series([0.0, 0.01, 0.05, 0.10, 0.15, 0.20, 0.30, 0.31, 1.5])
    assert list(band(wer)) == ["0", "0–5%", "0–5%", "5–15%", "5–15%", "15–30%", "15–30%", ">30%", ">30%"]


def test_there_are_five_bands():
    assert len(BAND_LABELS) == 5


def manifest(rows):
    return pd.DataFrame(rows, columns=["id", "sentence_id", "gender"])


def test_one_recording_per_sentence_and_genders_are_balanced():
    rows = [(f"{s}_{g}", s, g) for s in "abcd" for g in ("female", "male")]
    chosen = select(manifest(rows))
    assert chosen["sentence_id"].is_unique and len(chosen) == 4
    assert chosen["id"].str.endswith("female").sum() == 2


def test_sentences_read_by_one_gender_only_are_still_included():
    rows = [("a_f", "a", "female"), ("a_m", "a", "male"), ("b_f", "b", "female")]
    chosen = select(manifest(rows))
    assert set(chosen["sentence_id"]) == {"a", "b"}


def test_selection_is_reproducible():
    rows = [(f"{s}_{g}", s, g) for s in "abcdefgh" for g in ("female", "male")]
    assert select(manifest(rows)).equals(select(manifest(rows)))
