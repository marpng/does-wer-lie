import matplotlib

matplotlib.use("Agg")

import pandas as pd

from plot_wer_vs_human import BANDS, LANGUAGES, make_figure, panel_values


def sample() -> pd.DataFrame:
    rows = [("en", b, 10, 1.9) for b in BANDS] + [("el", b, 16, 1.0) for b in BANDS[:-1]]
    return pd.DataFrame(rows, columns=["language", "wer_band", "n", "mean_human"])


def test_missing_band_gives_zero_items_and_no_bar():
    values, counts = panel_values(sample(), "el")
    assert counts[-1] == 0 and values[-1] is None
    assert counts[0] == 16 and values[0] == 1.0


def test_figure_has_one_panel_per_language():
    fig = make_figure(sample())
    assert len(fig.axes) == len(LANGUAGES)
