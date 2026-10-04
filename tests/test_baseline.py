import numpy as np
import pandas as pd
import pytest

from bot.baseline import (COST_PER_SIDE, EXEC_DELAY, SMA_WINDOW, baseline_signal, run_baseline,
                          run_buy_and_hold, sma)

H1 = pd.Timedelta(hours=1)


def frame(close, missing=None):
    n = len(close)
    idx = pd.date_range(pd.Timestamp("2020-01-01 00:00", tz="UTC"), periods=n, freq="h",
                        name="open_time")
    c = np.asarray(close, dtype=np.float64)
    df = pd.DataFrame({"open": c, "high": c, "low": c, "close": c, "volume": 1.0,
                       "missing": False}, index=idx)
    if missing is not None:
        df.loc[df.index[missing], ["open", "high", "low", "close", "volume"]] = np.nan
        df.loc[df.index[missing], "missing"] = True
    return df


def test_frozen_parameters():
    assert SMA_WINDOW == 168
    assert COST_PER_SIDE == 0.0015
    assert EXEC_DELAY == 1


def test_sma_matches_naive_mean_and_needs_168_bars():
    rng = np.random.default_rng(1)
    c = 100 + np.cumsum(rng.normal(0, 1, 400))
    df = frame(c)
    s = sma(df).to_numpy()
    assert np.isnan(s[:167]).all() and not np.isnan(s[167])
    for t in (167, 250, 399):
        assert s[t] == pytest.approx(c[t - 167:t + 1].mean(), rel=1e-12)


def test_signal_rule_strict_greater():
    c = np.full(200, 100.0)
    c[180] = 101.0  # close > SMA -> 1
    c[190] = 99.0  # close < SMA -> 0
    sig = baseline_signal(frame(c))
    assert np.isnan(sig.iloc[:167]).all()
    assert sig.iloc[170] == 0.0  # égalité close == SMA -> 0 (strictement supérieur requis)
    assert sig.iloc[180] == 1.0
    assert sig.iloc[190] == 0.0


def test_missing_bar_gives_nan_for_window():
    c = np.linspace(100, 200, 500)
    sig = baseline_signal(frame(c, missing=[300]))
    assert np.isnan(sig.iloc[300:300 + 168]).all()
    assert sig.iloc[299] == 1.0 and sig.iloc[300 + 168] == 1.0


def test_signal_uses_only_past_bars():
    rng = np.random.default_rng(2)
    df = frame(100 + np.cumsum(rng.normal(0, 1, 600)))
    full = baseline_signal(df)
    for t in (200, 350, 599):
        trunc = baseline_signal(df.iloc[:t + 1])
        assert trunc.iloc[-1] == full.iloc[t]
    # modifier le futur ne change pas le passé
    df2 = df.copy()
    df2.iloc[400:, :4] *= 3
    pd.testing.assert_series_equal(baseline_signal(df2).iloc[:400], full.iloc[:400])


def test_baseline_executes_next_open_with_costs():
    c = np.concatenate([np.full(200, 100.0), np.linspace(101, 150, 50)])
    df = frame(c)
    r = run_baseline(df)
    first_long_signal = int(np.nanargmax(baseline_signal(df).to_numpy() == 1.0))
    assert r.position.iloc[first_long_signal] == 0.0
    assert r.position.iloc[first_long_signal + 1] == 1.0
    assert r.cost.iloc[first_long_signal + 1] == pytest.approx(0.0015, abs=1e-18)


def test_buy_and_hold_pays_both_sides():
    c = np.linspace(100, 200, 300)
    df = frame(c)
    a, b = df.index[100], df.index[200]
    r = run_buy_and_hold(df, a, b)
    expected = (1 - 0.0015) * (c[201] / c[100]) * (1 - 0.0015)  # marquage : open de 201
    assert r.final_equity == pytest.approx(expected, rel=1e-12)
    assert len(r.trades) == 1
