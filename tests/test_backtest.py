import numpy as np
import pandas as pd
import pytest

from bot.backtest import run_backtest, target_positions

C = 0.0015


def series(values, start="2021-01-01 00:00"):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=len(values), freq="h",
                        name="open_time")
    return pd.Series(np.asarray(values, dtype=np.float64), index=idx)


OPENS = [100.0, 110.0, 99.0, 108.9, 120.0, 114.0]


def test_execution_t_plus_1_signal_does_not_affect_its_own_bar():
    o = series(OPENS)
    sig = series([0, 0, 1, 1, 1, 1])  # décision longue à la clôture de t=2
    r = run_backtest(sig, o, cost_per_side=0.0)
    # position tenue : 0 sur les bougies 0..2, 1 à partir de la bougie 3
    assert r.position.tolist() == [0, 0, 0, 1, 1, 1]
    # le rendement de [2, 3] (108.9/99 - 1 = +10 %) n'est PAS capté
    assert r.bar_return.iloc[2] == 0.0
    assert r.bar_return.iloc[3] == pytest.approx(120.0 / 108.9 - 1.0, abs=1e-15)
    assert r.bar_return.iloc[4] == pytest.approx(114.0 / 120.0 - 1.0, abs=1e-15)
    assert r.bar_return.iloc[5] == 0.0  # dernière bougie : pas d'open suivant
    assert r.final_equity == pytest.approx(114.0 / 108.9, abs=1e-15)


def test_signal_change_at_t_changes_position_only_from_t_plus_1():
    o = series(OPENS)
    a = run_backtest(series([1, 1, 1, 1, 1, 1]), o, cost_per_side=0.0)
    b = run_backtest(series([1, 1, 0, 1, 1, 1]), o, cost_per_side=0.0)
    # signal modifié à t=2 : bougies 0..2 identiques, bougie 3 diffère
    assert (a.bar_return.iloc[:3] == b.bar_return.iloc[:3]).all()
    assert a.bar_return.iloc[3] != b.bar_return.iloc[3]


def test_exec_delay_2_shifts_one_more_bar():
    o = series(OPENS)
    sig = series([0, 1, 1, 1, 1, 1])
    r1 = run_backtest(sig, o, exec_delay=1, cost_per_side=0.0)
    r2 = run_backtest(sig, o, exec_delay=2, cost_per_side=0.0)
    assert r1.position.tolist() == [0, 0, 1, 1, 1, 1]
    assert r2.position.tolist() == [0, 0, 0, 1, 1, 1]
    assert r2.position.iloc[1:].tolist() == r1.position.iloc[:-1].tolist()
    assert r2.final_equity == pytest.approx(114.0 / 108.9, abs=1e-15)
    assert r1.final_equity == pytest.approx(114.0 / 99.0, abs=1e-15)


def test_exec_delay_below_1_refused():
    with pytest.raises(ValueError):
        target_positions(series([1, 1]), exec_delay=0)


def test_costs_equal_cost_per_side_times_abs_delta_position():
    o = series([100.0] * 6)  # prix constants : seul le coût agit
    sig = series([1, 0.5, 0, 1, 1, 1])
    r = run_backtest(sig, o)
    # positions : 0, 1, 0.5, 0, 1, 1 -> |dp| = 0, 1, 0.5, 0.5, 1, 0
    assert r.position.tolist() == [0, 1, 0.5, 0, 1, 1]
    expected = [C * d for d in (0, 1, 0.5, 0.5, 1, 0)]
    np.testing.assert_allclose(r.cost.to_numpy(), expected, rtol=0, atol=1e-18)
    eq = np.prod([1 - c for c in expected])
    assert r.final_equity == pytest.approx(eq, abs=1e-15)


def test_cost_multiplier_2_doubles_costs():
    o = series(OPENS)
    sig = series([1, 0, 1, 0, 1, 1])
    r1 = run_backtest(sig, o)
    r2 = run_backtest(sig, o, cost_multiplier=2.0)
    np.testing.assert_allclose(r2.cost.to_numpy(), 2 * r1.cost.to_numpy(), rtol=0, atol=1e-18)
    assert r1.cost.sum() > 0
    r3 = run_backtest(sig, o, cost_per_side=2 * C)
    np.testing.assert_allclose(r2.equity.to_numpy(), r3.equity.to_numpy(), rtol=0, atol=1e-15)


def test_net_equity_by_hand():
    o = series(OPENS)
    sig = series([1, 1, 0, 0, 1, 1])
    r = run_backtest(sig, o)
    # positions : 0, 1, 1, 0, 0, 1
    assert r.position.tolist() == [0, 1, 1, 0, 0, 1]
    e1 = (1 - C) * (99.0 / 110.0)
    e2 = e1 * (108.9 / 99.0)
    e3 = e2 * (1 - C)
    e5 = e3 * (1 - C)  # entrée à l'open de 5, pas d'open suivant -> rendement 0
    np.testing.assert_allclose(r.equity.to_numpy(), [1, e1, e2, e3, e3, e5], rtol=0, atol=1e-15)
    t = r.trades
    assert len(t) == 2
    assert t.loc[0, "net_return"] == pytest.approx((1 - C) * (108.9 / 110.0) * (1 - C) - 1, abs=1e-15)
    assert t.loc[0, "entry_price"] == 110.0 and t.loc[0, "exit_price"] == 108.9
    assert not t.loc[0, "open_at_end"] and t.loc[1, "open_at_end"]


def test_nan_signal_keeps_previous_position():
    o = series([100.0] * 6)
    sig = series([1, np.nan, np.nan, 0, np.nan, np.nan])
    r = run_backtest(sig, o)
    assert r.position.tolist() == [0, 1, 1, 1, 0, 0]
    # aucun trade forcé : un seul aller-retour (entrée à 1, sortie à 4)
    assert len(r.trades) == 1
    assert (r.cost > 0).sum() == 2


def test_nan_signal_before_first_valid_is_flat():
    o = series([100.0] * 4)
    r = run_backtest(series([np.nan, np.nan, 1, 1]), o)
    assert r.position.tolist() == [0, 0, 0, 1]


def test_price_gap_position_held_and_pnl_realized_at_first_open():
    # bougies 2 et 3 manquantes (open NaN)
    o = series([100.0, 100.0, np.nan, np.nan, 130.0, 130.0])
    sig = series([1, 1, np.nan, np.nan, 1, 1])
    r = run_backtest(sig, o, cost_per_side=0.0)
    assert r.position.tolist() == [0, 1, 1, 1, 1, 1]
    # P&L du trou (130/100 - 1) réalisé sur la bougie 3, juste avant l'open de 4
    np.testing.assert_allclose(r.bar_return.to_numpy(), [0, 0, 0, 0.3, 0, 0], atol=1e-15)
    np.testing.assert_allclose(r.equity.to_numpy(), [1, 1, 1, 1.3, 1.3, 1.3], atol=1e-15)


def test_trade_decided_before_gap_executed_at_first_available_open():
    o = series([100.0, 100.0, np.nan, np.nan, 130.0, 140.0])
    sig = series([0, 1, np.nan, np.nan, 1, 1])  # décision longue à t=1, open de 2 manquant
    r = run_backtest(sig, o)
    assert r.position.tolist() == [0, 0, 0, 0, 1, 1]
    assert r.cost.iloc[4] == pytest.approx(C, abs=1e-18)
    assert r.final_equity == pytest.approx((1 - C) * 140.0 / 130.0, abs=1e-15)


def test_windows_chained_equal_continuous_run():
    rng = np.random.default_rng(0)
    n = 400
    o = series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, n))))
    o.iloc[[50, 51, 199, 200, 201]] = np.nan  # trous, dont un à cheval sur une frontière
    sig = series((rng.random(n) > 0.5).astype(float))
    sig.iloc[[50, 51, 199, 200, 201, 300]] = np.nan
    bounds = [(100, 199), (200, 299), (300, 399)]
    full = run_backtest(sig, o, start=o.index[100], end=o.index[399])
    chained = []
    acc = 1.0
    n_trades = 0
    straddling = 0  # trade ouvert en fin de fenêtre k ET en début de fenêtre k+1
    prev_open_at_end = False
    for a, b in bounds:
        w = run_backtest(sig, o, start=o.index[a], end=o.index[b])
        chained.append(w.equity * acc)
        acc *= w.final_equity
        n_trades += len(w.trades)
        if prev_open_at_end and len(w.trades) and w.trades["open_at_start"].iloc[0]:
            straddling += 1
        prev_open_at_end = bool(len(w.trades) and w.trades["open_at_end"].iloc[-1])
    ch = pd.concat(chained)
    assert np.max(np.abs(ch.to_numpy() - full.equity.to_numpy())) <= 1e-12
    assert n_trades - straddling == len(full.trades)


def test_initial_position_and_close_at_end_buy_and_hold():
    o = series([90.0, 100.0, 110.0, 121.0])
    r = run_backtest(series([1, 1, 1, 1]), o, start=o.index[1], initial_position=0.0,
                     close_at_end=True)
    # acheté à l'open de la 1re bougie évaluée, vendu au marquage final (open de 3)
    assert r.position.tolist() == [1, 1, 1]
    assert r.final_equity == pytest.approx((1 - C) * 1.21 * (1 - C), abs=1e-15)
    assert len(r.trades) == 1 and not r.trades.loc[0, "open_at_end"]
    # sans initial_position, la position tenue avant la fenêtre (1) est héritée sans coût
    h = run_backtest(series([1, 1, 1, 1]), o, start=o.index[2])
    assert h.initial_position == 1.0 and h.cost.sum() == 0.0
    assert bool(h.trades.loc[0, "open_at_start"])


def test_signal_out_of_range_refused():
    with pytest.raises(ValueError):
        run_backtest(series([0, -1, 1]), series([1.0, 1.0, 1.0]))
    with pytest.raises(ValueError):
        run_backtest(series([0, 1.5, 1]), series([1.0, 1.0, 1.0]))


def test_irregular_grid_refused():
    o = series([1.0, 1.0, 1.0, 1.0])
    o2 = o.drop(o.index[1])
    with pytest.raises(ValueError):
        run_backtest(series([1, 1, 1]).set_axis(o2.index), o2)
