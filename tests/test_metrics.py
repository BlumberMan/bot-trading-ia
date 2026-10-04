import math

import numpy as np
import pandas as pd
import pytest

from bot.backtest import run_backtest
from bot.metrics import (annualized_return, daily_returns, exposure, max_drawdown,
                         profit_factor, sharpe_daily, summarize, total_return, win_rate)


def hourly(values, start="2021-03-01 00:00"):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(values), freq="h")
    return pd.Series(np.asarray(values, dtype=np.float64), index=idx)


def three_day_equity():
    # 72 bougies ; fins de jour : 1.01, 1.01*0.99, 1.01*0.99*1.02 ; bruit intrajournalier
    d1, d2, d3 = 1.01, 1.01 * 0.99, 1.01 * 0.99 * 1.02
    v = np.empty(72)
    v[:24] = np.linspace(0.97, d1, 24)
    v[24:48] = np.linspace(1.05, d2, 24)
    v[48:] = np.linspace(0.95, d3, 24)
    return hourly(v), [0.01, -0.01, 0.02]


def test_daily_returns_by_hand():
    eq, expected = three_day_equity()
    np.testing.assert_allclose(daily_returns(eq).to_numpy(), expected, rtol=0, atol=1e-15)


def test_sharpe_by_hand():
    eq, r = three_day_equity()
    mean = sum(r) / 3
    sd = math.sqrt(sum((x - mean) ** 2 for x in r) / 2)  # ddof = 1
    expected = mean / sd * math.sqrt(365)
    assert sharpe_daily(eq) == pytest.approx(expected, rel=1e-12)
    assert expected == pytest.approx(8.3381, abs=1e-4)


def test_sharpe_utc_days_and_degenerate_cases():
    # 23:00 et 00:00 appartiennent à deux jours UTC différents
    eq = hourly([1.0, 1.1], start="2021-03-01 23:00")
    np.testing.assert_allclose(daily_returns(eq).to_numpy(), [0.0, 0.1], atol=1e-15)
    assert math.isnan(sharpe_daily(hourly([1.0] * 48)))  # écart-type nul
    assert math.isnan(sharpe_daily(hourly([1.0, 1.1])))  # un seul jour


def test_max_drawdown_by_hand():
    eq = hourly([1.1, 0.88, 0.99, 1.2, 0.9])
    # pics : 1.1, 1.1, 1.1, 1.2, 1.2 -> DD : 0, 0.2, 0.1, 0, 0.25
    assert max_drawdown(eq) == pytest.approx(0.25, abs=1e-15)
    # l'equity initiale (1) compte dans le pic
    assert max_drawdown(hourly([0.8, 0.9])) == pytest.approx(0.2, abs=1e-15)
    assert max_drawdown(hourly([1.0, 1.1, 1.2])) == 0.0


def test_profit_factor_and_win_rate_by_hand():
    tr = [0.10, -0.05, 0.02, -0.03]
    assert profit_factor(tr) == pytest.approx((0.10 + 0.02) / (0.05 + 0.03), rel=1e-12)
    assert profit_factor(tr) == pytest.approx(1.5, rel=1e-12)
    assert win_rate(tr) == 0.5
    assert win_rate([0.1, 0.0, -0.1]) == pytest.approx(1 / 3)  # 0 n'est pas un gain
    assert profit_factor([0.1, 0.2]) == math.inf
    assert math.isnan(profit_factor([]))
    assert math.isnan(win_rate([]))


def test_exposure_total_and_annualized_return():
    assert exposure(hourly([0, 1, 1, 0.5, 0])) == pytest.approx(0.6)
    eq = hourly(np.full(8760, 1.0))
    eq.iloc[-1] = 1.21
    assert total_return(eq) == pytest.approx(0.21, abs=1e-15)
    assert annualized_return(eq) == pytest.approx(0.21, abs=1e-12)  # exactement 1 an
    eq2 = hourly(np.full(2 * 8760, 1.21))
    assert annualized_return(eq2) == pytest.approx(0.1, abs=1e-12)  # 2 ans -> sqrt(1.21) - 1


def test_summarize_consistency():
    o = hourly([100.0, 110.0, 99.0, 108.9, 120.0, 114.0])
    r = run_backtest(hourly([1, 1, 0, 0, 1, 1]), o)
    s = summarize(r)
    assert s["trades"] == len(r.trades) == 2
    assert s["total_return"] == pytest.approx(r.final_equity - 1, abs=1e-15)
    assert s["exposure"] == pytest.approx(3 / 6)
