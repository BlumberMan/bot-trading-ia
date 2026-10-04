import numpy as np
import pandas as pd
import pytest

from bot.features import (FEATURES, LOOKBACK, MAX_SPAN, SPANS, compute_features,
                          compute_features_live)

SEED = 12345
N = 3000


def synth(n=N, seed=SEED, gaps=((400, 1), (1000, 3), (1700, 8), (2500, 1))):
    """OHLCV synthétique horaire (seed fixe) avec trous marqués missing."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-01", periods=n, freq="h", tz="UTC", name="open_time")
    r = rng.normal(0, 0.01, n)
    close = 10_000 * np.exp(np.cumsum(r))
    open_ = np.concatenate([[10_000.0], close[:-1]]) * np.exp(rng.normal(0, 0.001, n))
    high = np.maximum(open_, close) * np.exp(np.abs(rng.normal(0, 0.003, n)))
    low = np.minimum(open_, close) * np.exp(-np.abs(rng.normal(0, 0.003, n)))
    vol = np.exp(rng.normal(5, 1, n))
    vol[50] = 0.0  # volume nul
    df = pd.DataFrame({"open": open_, "high": high, "low": low, "close": close,
                       "volume": vol, "missing": False}, index=idx)
    for start, length in gaps:
        df.iloc[start:start + length, :5] = np.nan
        df.iloc[start:start + length, 5] = True
    return df


def _eq(a, b):
    return np.array_equal(np.asarray(a, dtype=float), np.asarray(b, dtype=float), equal_nan=True)


def test_lookback_covers_spans():
    assert LOOKBACK >= MAX_SPAN
    assert set(SPANS) == set(FEATURES)


def test_live_equals_batch_500_instants():
    df = synth()
    batch = compute_features(df)
    rng = np.random.default_rng(SEED + 1)
    ts = rng.choice(np.arange(LOOKBACK - 1, len(df)), size=600, replace=False)
    n_nan = 0
    for t in ts:
        live = compute_features_live(df.iloc[t - LOOKBACK + 1: t + 1]).to_numpy()
        ref = batch.iloc[t].to_numpy()
        assert (np.isnan(live) == np.isnan(ref)).all(), t
        ok = ~np.isnan(ref)
        assert np.max(np.abs(live[ok] - ref[ok]), initial=0.0) <= 1e-9, t
        n_nan += np.isnan(ref).any()
    assert len(ts) >= 500
    assert n_nan > 0  # des instants près des trous sont bien testés


def test_live_uses_only_last_lookback_rows():
    df = synth()
    t = 2000
    a = compute_features_live(df.iloc[: t + 1])
    b = compute_features_live(df.iloc[t - LOOKBACK + 1: t + 1])
    assert _eq(a, b)
    with pytest.raises(ValueError):
        compute_features_live(df.iloc[t - LOOKBACK + 2: t + 1])


@pytest.mark.parametrize("t", [300, 999, 1500, 2200, 2998])
def test_no_lookahead_features(t):
    df = synth()
    ref = compute_features(df)
    noisy = df.copy()
    rng = np.random.default_rng(t)
    k = len(df) - (t + 1)
    noisy.iloc[t + 1:, :4] = rng.uniform(1, 1e6, size=(k, 4))
    noisy.iloc[t + 1:, 4] = rng.uniform(0, 1e6, size=k)
    noisy.iloc[t + 1:, 5] = rng.random(k) < 0.3
    out = compute_features(noisy)
    assert np.array_equal(ref.iloc[: t + 1].to_numpy(), out.iloc[: t + 1].to_numpy(), equal_nan=True)
    # et la perturbation a bien un effet après t (le test n'est pas vide)
    assert not np.array_equal(ref.iloc[t + 1:].to_numpy(), out.iloc[t + 1:].to_numpy(), equal_nan=True)


def test_missing_window_gives_nan():
    df = synth()
    feats = compute_features(df)
    miss = df["missing"].to_numpy()
    for name, span in SPANS.items():
        col = feats[name].to_numpy()
        if span == 0:
            assert not np.isnan(col).any(), name
            continue
        for t in range(len(df)):
            lo = t - span + 1
            expect_nan = lo < 0 or miss[max(lo, 0): t + 1].any()
            if expect_nan:
                assert np.isnan(col[t]), (name, t)
            else:
                assert np.isfinite(col[t]), (name, t)


def test_formulas_spot_check():
    df = synth(gaps=())
    f = compute_features(df)
    t = 1234
    c = df["close"].to_numpy()
    lr = np.diff(np.log(c))
    assert f["ret_4"].iloc[t] == pytest.approx(np.log(c[t] / c[t - 4]), abs=1e-12)
    assert f["vol_24"].iloc[t] == pytest.approx(np.std(lr[t - 24:t], ddof=1), abs=1e-12)
    assert f["sma_dev_168"].iloc[t] == pytest.approx(c[t] / c[t - 167: t + 1].mean() - 1, abs=1e-12)
    d = np.diff(c[t - 14: t + 1])
    g, l = d[d > 0].sum() / 14, -d[d < 0].sum() / 14
    assert f["rsi_14"].iloc[t] == pytest.approx(100 * g / (g + l), abs=1e-9)
    lv = np.log1p(df["volume"].to_numpy()[t - 167: t + 1])
    assert f["logvol_z_168"].iloc[t] == pytest.approx((lv[-1] - lv.mean()) / lv.std(ddof=1), abs=1e-9)
    rg = ((df["high"] - df["low"]) / df["close"]).to_numpy()
    assert f["range_24"].iloc[t] == pytest.approx(rg[t - 23: t + 1].mean(), abs=1e-12)
    ts = df.index[t]
    assert f["hour_sin"].iloc[t] == pytest.approx(np.sin(2 * np.pi * ts.hour / 24))
    assert f["dow_cos"].iloc[t] == pytest.approx(np.cos(2 * np.pi * ts.dayofweek / 7))
