import numpy as np
import pandas as pd
import pytest

from bot.labels import HORIZON, compute_labels

H = HORIZON


def synth(n=500, seed=7, gaps=((100, 1), (300, 3))):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2021-01-01", periods=n, freq="h", tz="UTC", name="open_time")
    o = 100 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    df = pd.DataFrame({"open": o, "high": o * 1.01, "low": o * 0.99, "close": o,
                       "volume": 1.0, "missing": False}, index=idx)
    for s, k in gaps:
        df.iloc[s:s + k, :5] = np.nan
        df.iloc[s:s + k, 5] = True
    return df


def test_horizon_value():
    assert H == 4


def test_label_definition():
    df = synth(gaps=())
    lab = compute_labels(df)
    o = df["open"].to_numpy()
    for t in range(len(df) - 1 - H):
        assert lab["label"].iloc[t] == float(o[t + 1 + H] > o[t + 1])
        assert lab["fwd_ret"].iloc[t] == pytest.approx(np.log(o[t + 1 + H] / o[t + 1]), abs=1e-15)
    assert lab.iloc[len(df) - 1 - H:].isna().all().all()  # hors données


def test_label_nan_if_missing_in_window():
    df = synth()
    lab = compute_labels(df)
    miss = df["missing"].to_numpy()
    for t in range(len(df)):
        need = range(t + 1, t + 2 + H)
        bad = need.stop - 1 >= len(df) or miss[t + 1: t + 2 + H].any()
        assert np.isnan(lab["label"].iloc[t]) == bad, t
        assert np.isnan(lab["fwd_ret"].iloc[t]) == bad, t


@pytest.mark.parametrize("t", [10, 150, 250, 400])
def test_label_no_lookahead_beyond_horizon(t):
    df = synth()
    ref = compute_labels(df)
    noisy = df.copy()
    rng = np.random.default_rng(t)
    k = len(df) - (t + 2 + H)
    noisy.iloc[t + 2 + H:, :5] = rng.uniform(1, 1e4, size=(k, 5))
    out = compute_labels(noisy)
    assert np.array_equal(ref.iloc[: t + 1].to_numpy(), out.iloc[: t + 1].to_numpy(), equal_nan=True)


def test_label_changes_when_exit_open_changes():
    df = synth(gaps=())
    t = 200
    ref = compute_labels(df)["label"].iloc[t]
    mod = df.copy()
    entry = mod["open"].iloc[t + 1]
    mod.iloc[t + 1 + H, mod.columns.get_loc("open")] = entry * (0.5 if ref == 1.0 else 2.0)
    assert compute_labels(mod)["label"].iloc[t] == 1.0 - ref
