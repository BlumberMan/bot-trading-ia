"""Tests de l'agrégation 4 h / journalière (sans réseau, données synthétiques)."""

import numpy as np
import pandas as pd
import pytest

from bot.resample import OHLCV, close_time, resample_ohlcv, resample_ohlcv_live

H1 = pd.Timedelta(hours=1)
K = {"4h": 4, "1d": 24}


def _hours(n=24 * 120, seed=0, start="2021-03-01 00:00", holes=()):
    idx = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n, freq="h", name="open_time")
    rng = np.random.default_rng(seed)
    c = 30000 * np.exp(np.cumsum(rng.normal(0, 0.01, n)))
    o = np.concatenate([[c[0]], c[:-1]])
    hi = np.maximum(o, c) * (1 + np.abs(rng.normal(0, 0.002, n)))
    lo = np.minimum(o, c) * (1 - np.abs(rng.normal(0, 0.002, n)))
    v = np.exp(rng.normal(5, 1, n))
    df = pd.DataFrame({"open": o, "high": hi, "low": lo, "close": c, "volume": v}, index=idx)
    df["missing"] = False
    for a in holes:
        df.iloc[a, :5] = np.nan
        df.iloc[a, df.columns.get_loc("missing")] = True
    return df


def _hand_df():
    # 48 heures à partir de 2024-01-01 00:00 UTC, valeurs entières faciles à agréger à la main
    idx = pd.date_range("2024-01-01 00:00", periods=48, freq="h", tz="UTC", name="open_time")
    i = np.arange(48, dtype=np.float64)
    df = pd.DataFrame({"open": 100 + i, "high": 200 + i, "low": 50 + i, "close": 101 + i,
                       "volume": 1 + i}, index=idx)
    df.loc[idx[2], "high"] = 999.0   # pic dans la 1re bougie 4 h
    df.loc[idx[1], "low"] = 1.0      # creux dans la 1re bougie 4 h
    df["missing"] = False
    return df


# ------------------------------------------------------------------ cas calculés à la main

def test_hand_4h():
    a = resample_ohlcv(_hand_df(), "4h")
    assert len(a) == 12
    assert list(a.columns) == OHLCV + ["missing"]
    # 1re bougie 00:00 : heures 0..3
    r = a.iloc[0]
    assert a.index[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == (100.0, 999.0, 1.0, 104.0, 10.0)
    # 2e bougie 04:00 : heures 4..7 -> open 104, high 207, low 54, close 108, volume 5+6+7+8 = 26
    r = a.iloc[1]
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == (104.0, 207.0, 54.0, 108.0, 26.0)
    assert not a["missing"].any()


def test_hand_1d():
    a = resample_ohlcv(_hand_df(), "1d")
    assert len(a) == 2
    assert list(a.index) == [pd.Timestamp("2024-01-01", tz="UTC"), pd.Timestamp("2024-01-02", tz="UTC")]
    r = a.iloc[0]  # heures 0..23 : volume = 1+...+24 = 300
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == (100.0, 999.0, 1.0, 124.0, 300.0)
    r = a.iloc[1]  # heures 24..47 : volume = 25+...+48 = 876
    assert (r["open"], r["high"], r["low"], r["close"], r["volume"]) == (124.0, 247.0, 74.0, 148.0, 876.0)


def test_naive_loop_equality():
    df = _hours(holes=(50, 51, 700, 1999))
    for rule, k in K.items():
        a = resample_ohlcv(df, rule)
        for T in a.index:
            hrs = df.loc[T: T + (k - 1) * H1]
            assert len(hrs) == k
            if hrs["missing"].any():
                assert a.loc[T, "missing"] and a.loc[T, OHLCV].isna().all()
                continue
            vol = hrs["volume"].iloc[0]
            for x in hrs["volume"].iloc[1:]:
                vol = vol + x
            assert a.loc[T, "open"] == hrs["open"].iloc[0]
            assert a.loc[T, "high"] == hrs["high"].max()
            assert a.loc[T, "low"] == hrs["low"].min()
            assert a.loc[T, "close"] == hrs["close"].iloc[-1]
            assert a.loc[T, "volume"] == vol
            assert not a.loc[T, "missing"]


# ------------------------------------------------------------------ bougies manquantes

def test_missing_bars():
    # trous : heure 5 (bougie 4 h 04:00, jour 0), heures 47-48 (à cheval sur 2 bougies 4 h et 2 jours)
    df = _hours(n=24 * 5, holes=(5, 47, 48))
    a4 = resample_ohlcv(df, "4h")
    m4 = a4.index[a4["missing"]]
    t0 = df.index[0]
    assert list(m4) == [t0 + 4 * H1, t0 + 44 * H1, t0 + 48 * H1]
    assert a4.loc[m4, OHLCV].isna().all().all()
    a1 = resample_ohlcv(df, "1d")
    assert list(a1["missing"]) == [True, True, True, False, False]
    assert a1.loc[a1["missing"], OHLCV].isna().all().all()
    assert not a1.loc[~a1["missing"], OHLCV].isna().any().any()


def test_nan_without_flag_is_missing():
    df = _hours(n=48)
    df.iloc[10, df.columns.get_loc("volume")] = np.nan  # NaN sans missing=True
    a4 = resample_ohlcv(df, "4h")
    a1 = resample_ohlcv(df, "1d")
    assert bool(a4.iloc[2]["missing"]) and a4.iloc[2][OHLCV].isna().all()
    assert int(a4["missing"].sum()) == 1
    assert list(a1["missing"]) == [True, False]


def test_single_missing_hour_kills_day():
    for hole in (0, 11, 23):
        a1 = resample_ohlcv(_hours(n=48, holes=(hole,)), "1d")
        assert list(a1["missing"]) == [True, False]


# ------------------------------------------------------------------ pas de bougie partielle

@pytest.mark.parametrize("rule", ["4h", "1d"])
def test_no_partial_bar_at_end(rule):
    k = K[rule]
    df = _hours(n=24 * 10)
    full = resample_ohlcv(df, rule)
    for cut in range(1, k + 1):
        a = resample_ohlcv(df.iloc[: len(df) - cut], rule)
        last = a.index[-1]
        # dernière bougie émise : entièrement contenue dans l'entrée
        assert close_time(last, rule) - H1 <= df.index[len(df) - cut - 1]
        assert len(a) == len(full) - 1
        assert a.equals(full.iloc[:-1])


@pytest.mark.parametrize("rule", ["4h", "1d"])
def test_no_partial_bar_at_start(rule):
    df = _hours(n=24 * 10)
    full = resample_ohlcv(df, rule)
    a = resample_ohlcv(df.iloc[1:], rule)
    assert a.equals(full.iloc[1:])


# ------------------------------------------------------------------ anti-fuite

@pytest.mark.parametrize("rule", ["4h", "1d"])
def test_no_leak_after_close(rule):
    k = K[rule]
    df = _hours(n=24 * 60, holes=(100, 800))
    base = resample_ohlcv(df, rule)
    rng = np.random.default_rng(7)
    pts = rng.choice(np.arange(len(base) - 2), size=40, replace=False)
    for i in pts:
        T = base.index[i]
        cut = df.index.get_loc(close_time(T, rule))  # 1re heure après la clôture de T
        d2 = df.copy()
        m = len(d2) - cut
        for col in ("open", "high", "low", "close"):
            d2.iloc[cut:, d2.columns.get_loc(col)] = d2[col].iloc[cut:].to_numpy() * np.exp(rng.normal(0, 0.05, m))
        d2.iloc[cut:, d2.columns.get_loc("volume")] = rng.exponential(100.0, m)
        d2.iloc[cut, d2.columns.get_loc("missing")] = True
        a2 = resample_ohlcv(d2, rule)
        assert a2.iloc[: i + 1].equals(base.iloc[: i + 1])
        assert not a2.iloc[i + 1:].equals(base.iloc[i + 1:])
        # sensibilité : la dernière heure de T est bien lue
        d3 = df.copy()
        d3.iloc[cut - 1, d3.columns.get_loc("close")] *= 1.01
        a3 = resample_ohlcv(d3, rule)
        if not base.iloc[i]["missing"]:
            assert a3.iloc[i]["close"] != base.iloc[i]["close"]
        assert a3.iloc[:i].equals(base.iloc[:i])
    assert k in (4, 24)


# ------------------------------------------------------------------ alignement UTC

def test_alignment_utc():
    df = _hours(n=24 * 20, start="2022-05-03 13:00")  # début non aligné
    a4 = resample_ohlcv(df, "4h")
    a1 = resample_ohlcv(df, "1d")
    assert set(a4.index.hour) == {0, 4, 8, 12, 16, 20}
    assert (a4.index.minute == 0).all()
    assert a4.index[0] == pd.Timestamp("2022-05-03 16:00", tz="UTC")
    assert (a1.index.hour == 0).all() and (a1.index.minute == 0).all()
    assert a1.index[0] == pd.Timestamp("2022-05-04 00:00", tz="UTC")
    assert (np.diff(a4.index.to_numpy()) == np.timedelta64(4, "h")).all()
    assert (np.diff(a1.index.to_numpy()) == np.timedelta64(24, "h")).all()
    assert str(a4.index.tz) == "UTC" and a4.index.name == "open_time"


def test_alignment_non_utc_input():
    # même instants exprimés en Europe/Paris : la grille reste UTC
    df = _hours(n=24 * 5)
    dp = df.copy()
    dp.index = dp.index.tz_convert("Europe/Paris")
    a = resample_ohlcv(dp, "1d")
    assert (a.index.tz_convert("UTC").hour == 0).all()
    assert np.array_equal(a[OHLCV].to_numpy(), resample_ohlcv(df, "1d")[OHLCV].to_numpy(), equal_nan=True)


def test_input_validation():
    df = _hours(n=48)
    with pytest.raises(ValueError):
        resample_ohlcv(df, "2h")
    with pytest.raises(ValueError):
        resample_ohlcv(df.drop(df.index[5]), "4h")  # grille non régulière
    d2 = df.copy()
    d2.index = d2.index.tz_localize(None)
    with pytest.raises(ValueError):
        resample_ohlcv(d2, "4h")


# ------------------------------------------------------------------ live = lot

@pytest.mark.parametrize("rule,n_bars", [("4h", 50), ("1d", 10)])
def test_live_equals_batch(rule, n_bars):
    k = K[rule]
    df = _hours(n=24 * 200, seed=3, holes=(300, 301, 2000, 3500))
    batch = resample_ohlcv(df, rule)
    need = k * n_bars + (k - 1)  # garantit n_bars bougies complètes dans la fenêtre
    rng = np.random.default_rng(20261005)
    js = rng.choice(np.arange(need - 1, len(df)), size=500, replace=False)
    n_eq = 0
    for j in js:
        hours = df.iloc[j - need + 1: j + 1]  # heures clôturées à la fin de l'heure j
        live = resample_ohlcv_live(hours, rule, n_bars)
        end = df.index[j] + H1  # instant courant
        # la dernière bougie live est clôturée, et c'est la dernière bougie clôturée du lot
        assert close_time(live.index[-1], rule) <= end
        ref = batch.loc[close_time(batch.index, rule) <= end].iloc[-n_bars:]
        n_eq += int(live.equals(ref))
    assert n_eq == 500


def test_live_too_short():
    df = _hours(n=30)
    with pytest.raises(ValueError):
        resample_ohlcv_live(df, "1d", 2)
    assert len(resample_ohlcv_live(df, "1d")) == 1
