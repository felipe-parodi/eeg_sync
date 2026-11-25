"""The gaze dashboard shows the whole-body synchrony windows, not the old xcorr."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from gaze_analysis.config import SessionBlock  # noqa: E402
from gaze_analysis.plotting import plot_dashboard  # noqa: E402


def _inputs():
    t = np.arange(0.0, 100.0, 0.2)
    frames = range(len(t))
    prox = pd.DataFrame(
        {"frame_idx": frames, "timestamp_s": t, "torso_distance_px": 50.0}
    )
    gaze = pd.DataFrame(
        {"frame_idx": frames, "timestamp_s": t, "gaze_category": "independent"}
    )
    conv = pd.DataFrame(columns=["frame_idx", "timestamp_s", "gaze_convergence_score"])
    xcorr = pd.DataFrame(
        {
            "window_start_s": [0.0],
            "window_end_s": [5.0],
            "peak_xcorr": [0.5],
            "peak_lag_s": [0.0],
        }
    )
    sync = pd.DataFrame(
        {
            "window_start_s": [0.0, 10.0],
            "window_end_s": [20.0, 30.0],
            "sync_r": [0.2, 0.4],
            "block": "grocery",
        }
    )
    blocks = [SessionBlock("grocery", 0.0, 100.0, "#4CAF50")]
    return prox, xcorr, gaze, conv, sync, blocks


def test_dashboard_uses_sync_windows_when_available() -> None:
    prox, xcorr, gaze, conv, sync, blocks = _inputs()

    fig = plot_dashboard(
        prox, xcorr, gaze, conv, blocks, sync_df=sync, sync_chance={"grocery": 0.05}
    )

    titles = [ax.get_title() for ax in fig.axes]
    assert any("Movement synchrony" in t for t in titles)
    assert not any("Cross-Correlation" in t or "Peak Lag" in t for t in titles)


def test_dashboard_falls_back_to_xcorr_for_old_runs() -> None:
    prox, xcorr, gaze, conv, _sync, blocks = _inputs()

    fig = plot_dashboard(prox, xcorr, gaze, conv, blocks)

    assert any("Cross-Correlation" in ax.get_title() for ax in fig.axes)


def test_dashboard_keeps_sync_panel_when_selection_has_no_windows() -> None:
    prox, xcorr, gaze, conv, sync, blocks = _inputs()

    fig = plot_dashboard(prox, xcorr, gaze, conv, blocks, sync_df=sync.iloc[0:0])

    titles = [ax.get_title() for ax in fig.axes]
    assert any("Movement synchrony" in t for t in titles)
    assert not any("Cross-Correlation" in t for t in titles)
