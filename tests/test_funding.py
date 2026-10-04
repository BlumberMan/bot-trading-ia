"""Tests du pipeline funding (sans réseau, données synthétiques)."""

import hashlib
import io
import urllib.error
import zipfile

import numpy as np
import pandas as pd
import pytest

from bot import data, funding
from bot.split import DEV_END

H1 = pd.Timedelta(hours=1)
H8 = pd.Timedelta(hours=8)
HDR = "calc_time,funding_interval_hours,last_funding_rate\n"


def _settlements(start="2024-01-01 00:00", n=300, seed=0, jitter_ms=None):
    rng = np.random.default_rng(seed)
    t = pd.date_range(pd.Timestamp(start, tz="UTC"), periods=n, freq="8h").as_unit("ns")
    if jitter_ms is not None:
        t = t + pd.to_timedelta(rng.integers(0, jitter_ms + 1, n), unit="ms")
    return pd.DataFrame({"fundingTime": t, "fundingRate": rng.normal(1e-4, 2e-4, n)})


def _grid(start, end):
    return pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"),
                         freq="1h", name="open_time")


# ---------------------------------------------------------------- lecture / nettoyage

def test_parse_ms_and_us_give_same_times():
    ms = [1704067200000, 1704096000002]  # 2024-01-01 00:00 et 08:00:00.002
    csv_ms = HDR + "".join(f"{t},8,0.0001\n" for t in ms)
    csv_us = HDR + "".join(f"{t * 1000},8,0.0001\n" for t in ms)
    a, ua = funding.parse_funding_csv(csv_ms.encode())
    b, ub = funding.parse_funding_csv(csv_us.encode())
    assert (ua, ub) == ("ms", "us")
    assert a["fundingTime"].tolist() == b["fundingTime"].tolist()
    assert a["fundingTime"].iloc[0] == pd.Timestamp("2024-01-01 00:00", tz="UTC")
    assert a["fundingTime"].iloc[1] == pd.Timestamp("2024-01-01 08:00:00.002", tz="UTC")
    assert str(a["fundingTime"].dt.tz) == "UTC"
    assert a["fundingRate"].dtype == np.float64
    with pytest.raises(ValueError):
        funding.parse_funding_csv((HDR + f"{ms[0]},8,0.1\n{ms[1] * 1000},8,0.1\n").encode())


def test_parse_alternative_header_with_mark_price():
    raw = "symbol,fundingTime,fundingRate,markPrice\nBTCUSDT,1704067200000,-0.0002,42000.5\n"
    df, unit = funding.parse_funding_csv(raw.encode())
    assert unit == "ms"
    assert list(df.columns) == ["fundingTime", "fundingRate", "markPrice"]
    assert df["markPrice"].iloc[0] == 42000.5 and df["fundingRate"].iloc[0] == -0.0002
    with pytest.raises(ValueError):
        funding.parse_funding_csv(b"a,b\n1,2\n")


def test_clean_duplicates_intervals_outliers():
    t0 = pd.Timestamp("2024-01-01 00:00", tz="UTC")
    times = [t0, t0 + H8, t0 + H8, t0 + 2 * H8 + pd.Timedelta(milliseconds=5),
             t0 + 3 * H8, t0 + 5 * H8, t0 + 5 * H8 + 4 * H1]
    raw = pd.DataFrame({"fundingTime": pd.DatetimeIndex(times).as_unit("ns"),
                        "fundingRate": [1e-4, 2e-4, 3e-4, -0.02, 1e-4, 0.015, 1e-4]})
    raw = raw.iloc[::-1].reset_index(drop=True)  # désordonné
    clean, rep = funding.clean_funding(raw, "2024-01", "2024-01")
    assert rep.duplicates_removed == 1 and rep.duplicate_conflicts == 1
    assert rep.rows == 6 == len(clean)
    assert clean["fundingTime"].is_monotonic_increasing
    # 16 h (trou) et 4 h : signalés ; 8 h ± 5 ms : gigue tolérée, comptée à part
    assert [(b - a) for a, b, _ in rep.bad_intervals] == [2 * H8, 4 * H1]
    assert rep.jitter_intervals == 2 and rep.jitter_max == pd.Timedelta(milliseconds=5)
    assert [r for _, r in rep.outliers] == [-0.02, 0.015]
    assert (clean["fundingRate"].abs() > 0.01).sum() == 2  # non supprimées


def test_clean_out_of_range():
    t = pd.DatetimeIndex(["2023-12-31 16:00", "2024-01-01 00:00", "2024-02-01 00:00"], tz="UTC")
    raw = pd.DataFrame({"fundingTime": t.as_unit("ns"), "fundingRate": [0.0, 0.0, 0.0]})
    clean, rep = funding.clean_funding(raw, "2024-01", "2024-01")
    assert rep.out_of_range_removed == 2 and len(clean) == 1


def test_months_range():
    ms = funding.months()
    assert ms[0] == "2020-01" and ms[-1] == "2026-08" and len(ms) == 80


# ---------------------------------------------------------------- archives

def _zip_bytes(name, csv_text):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name.replace(".zip", ".csv"), csv_text)
    return buf.getvalue()


def test_missing_monthly_archive_fails_without_fallback(tmp_path, monkeypatch):
    seen = []

    def fake_get(url, **k):
        seen.append(url)
        raise urllib.error.HTTPError(url, 404, "Not Found", {}, None)

    monkeypatch.setattr(data, "_http_get", fake_get)
    with pytest.raises(funding.MissingArchiveError):
        funding.fetch_month("2020-01", tmp_path)
    assert seen and all("/futures/um/monthly/fundingRate/BTCUSDT/" in u for u in seen)
    assert not any("daily" in u for u in seen)
    monkeypatch.setattr(funding, "months", lambda: ["2020-01", "2020-02"])
    with pytest.raises(funding.MissingArchiveError):
        funding.build(tmp_path, tmp_path / "out.parquet", log=lambda *a: None)
    assert not (tmp_path / "out.parquet").exists()
    assert not any("daily" in u for u in seen)


def test_download_checksum_ok_and_bad(tmp_path, monkeypatch):
    arch = funding.monthly_archive("2020-01", tmp_path)
    payload = _zip_bytes(arch.name, HDR + "1577836800000,8,0.0001\n")
    good = f"{hashlib.sha256(payload).hexdigest()}  {arch.name}\n".encode()
    monkeypatch.setattr(data, "_http_get",
                        lambda url, **k: good if url.endswith(".CHECKSUM") else payload)
    a = funding.fetch_month("2020-01", tmp_path)
    assert a.status == "downloaded"
    df, unit = funding.read_archive(a.path)
    assert len(df) == 1 and unit == "ms"
    bad = f"{'0' * 64}  {arch.name}\n".encode()
    monkeypatch.setattr(data, "_http_get",
                        lambda url, **k: bad if url.endswith(".CHECKSUM") else payload)
    with pytest.raises(funding.ChecksumError):
        funding.fetch_month("2020-01", tmp_path / "autre")


# ---------------------------------------------------------------- alignement

def test_alignment_boundaries_00_08_16():
    t0 = pd.Timestamp("2024-01-02 00:00", tz="UTC")
    f = pd.DataFrame({"fundingTime": pd.DatetimeIndex([t0, t0 + H8, t0 + 2 * H8]).as_unit("ns"),
                      "fundingRate": [0.1, 0.2, 0.3]})
    g = funding.funding_on_grid(f, _grid("2024-01-01 21:00", "2024-01-02 17:00"))
    T = lambda s: pd.Timestamp(s, tz="UTC")  # noqa: E731
    # NaN avant le premier règlement ; le règlement de 00:00 est disponible à la clôture
    # de la bougie 23:00 (t + 1 h = 00:00), pas avant
    assert g.loc[T("2024-01-01 21:00"):T("2024-01-01 22:00"), "fundingRate"].isna().all()
    assert g.loc[T("2024-01-01 22:00"), "fundingTime"] is pd.NaT
    assert g.loc[T("2024-01-01 23:00"), "fundingRate"] == 0.1
    assert g.loc[T("2024-01-01 23:00"), "age_h"] == 0.0
    assert g.loc[T("2024-01-02 06:00"), "fundingRate"] == 0.1
    assert g.loc[T("2024-01-02 06:00"), "age_h"] == 7.0
    assert g.loc[T("2024-01-02 07:00"), "fundingRate"] == 0.2  # 08:00 <= 07:00 + 1 h
    assert g.loc[T("2024-01-02 14:00"), "fundingRate"] == 0.2
    assert g.loc[T("2024-01-02 15:00"), "fundingRate"] == 0.3  # 16:00 <= 15:00 + 1 h
    assert g.loc[T("2024-01-02 17:00"), "age_h"] == 2.0
    assert (g["fundingTime"].dropna() <= g["fundingTime"].dropna().index + H1).all()


def test_alignment_jitter_is_not_available_early():
    t8 = pd.Timestamp("2024-01-02 08:00:00.002", tz="UTC")
    f = pd.DataFrame({"fundingTime": pd.DatetimeIndex(
        [pd.Timestamp("2024-01-02 00:00", tz="UTC"), t8]).as_unit("ns"), "fundingRate": [0.1, 0.2]})
    g = funding.funding_on_grid(f, _grid("2024-01-02 06:00", "2024-01-02 09:00"))
    assert g["fundingRate"].tolist() == [0.1, 0.1, 0.2, 0.2]  # 07:00 : 08:00:00.002 > 08:00
    assert g["age_h"].iloc[2] == pytest.approx(1.0 - 0.002 / 3600, abs=1e-12)


def test_alignment_never_uses_future_random():
    f = _settlements(n=400, seed=1, jitter_ms=50)
    idx = _grid("2023-12-30 00:00", "2024-05-20 00:00")
    g = funding.funding_on_grid(f, idx)
    ok = g["fundingTime"].notna()
    assert (g.loc[ok, "fundingTime"] <= g.index[ok] + H1).all()
    assert (g.loc[ok, "age_h"] >= 0).all()
    inside = ok & (g.index + H1 <= f["fundingTime"].iloc[-1] + H8)
    assert (g.loc[inside, "age_h"] < 8.001).all()
    # c'est bien le DERNIER règlement disponible
    ft = pd.DatetimeIndex(f["fundingTime"]).as_unit("ns").asi8
    for t in g.index[ok][::37]:
        last = ft[ft <= (t + H1).as_unit("ns").value][-1]
        assert g.loc[t, "fundingTime"].as_unit("ns").value == last
    assert g.loc[~ok].index.max() + H1 < f["fundingTime"].iloc[0]


def test_anti_leak_future_settlements_do_not_change_past():
    f = _settlements(n=400, seed=2, jitter_ms=30)
    idx = _grid("2024-01-01 00:00", "2024-05-10 00:00")
    base_g = funding.funding_on_grid(f, idx)
    base_x = funding.funding_features(f, idx)
    rng = np.random.default_rng(3)
    for t in rng.choice(idx[100:-10], size=15, replace=False):
        cut = t + H1
        g2 = f.copy()
        fut = g2["fundingTime"] > cut
        g2.loc[fut, "fundingRate"] = rng.normal(0, 1, int(fut.sum()))  # valeurs futures changées
        extra = pd.DataFrame({"fundingTime": pd.DatetimeIndex([cut + pd.Timedelta(milliseconds=1)]),
                              "fundingRate": [9.9]})
        if not (g2["fundingTime"] == extra["fundingTime"].iloc[0]).any():
            g2 = pd.concat([g2, extra]).sort_values("fundingTime").reset_index(drop=True)
        past = idx <= t
        pd.testing.assert_frame_equal(funding.funding_on_grid(g2, idx).loc[past], base_g.loc[past])
        pd.testing.assert_frame_equal(funding.funding_features(g2, idx).loc[past], base_x.loc[past])
        # règlements futurs supprimés : passé inchangé aussi
        g3 = f.loc[f["fundingTime"] <= cut].reset_index(drop=True)
        pd.testing.assert_frame_equal(funding.funding_features(g3, idx).loc[past], base_x.loc[past])


# ---------------------------------------------------------------- features

def test_features_formulas():
    r = np.arange(1, 101, dtype=np.float64) * 1e-5
    out = funding.funding_features_core(r)
    w = r[-90:]
    assert out[0] == r[-1]
    assert out[1] == pytest.approx(r[-3:].mean(), rel=1e-14)
    assert out[2] == pytest.approx(r[-21:].mean(), rel=1e-14)
    assert out[3] == pytest.approx((r[-1] - w.mean()) / w.std(ddof=1), rel=1e-12)
    assert out[4] == pytest.approx(r[-21:].sum(), rel=1e-14)
    short = funding.funding_features_core(r[:20])
    assert short[0] == r[19] and not np.isnan(short[1])
    assert np.isnan(short[2]) and np.isnan(short[3]) and np.isnan(short[4])
    assert np.isnan(funding.funding_features_core(np.full(90, 1e-4))[3])  # std nulle (arrondi)
    near = np.full(90, 1e-4)
    near[0] = 1e-4 + 1e-8  # plus petit écart publiable : z-score défini
    assert np.isfinite(funding.funding_features_core(near)[3])
    assert np.isnan(funding.funding_features_core(np.array([])) ).all()
    # seuls les 90 derniers règlements sont lus
    assert np.array_equal(funding.funding_features_core(np.r_[np.full(50, 7.0), r]), out)


def test_features_nan_before_first_settlement():
    f = _settlements(start="2024-01-01 00:00", n=50)
    x = funding.funding_features(f, _grid("2023-12-31 00:00", "2024-01-03 00:00"))
    assert x.loc[: pd.Timestamp("2023-12-31 22:00", tz="UTC")].isna().all().all()
    assert not np.isnan(x.loc[pd.Timestamp("2023-12-31 23:00", tz="UTC"), "fr_cur"])


def test_live_equals_batch_500_instants():
    f = _settlements(start="2024-01-01 00:00", n=600, seed=4, jitter_ms=40)
    idx = _grid("2023-12-31 12:00", "2024-07-30 00:00")
    batch = funding.funding_features(f, idx)
    rng = np.random.default_rng(20261005)
    ts = rng.choice(len(idx), size=500, replace=False)
    ft = pd.DatetimeIndex(f["fundingTime"]).as_unit("ns").asi8
    n_eq = 0
    for i in ts:
        t = idx[i]
        p = int(np.searchsorted(ft, (t + H1).as_unit("ns").value, side="right"))
        recent = f.iloc[max(0, p - funding.FUNDING_LOOKBACK - 5): p + 3]  # inclut du futur, ignoré
        live = funding.funding_features_live(recent, t).to_numpy()
        ref = batch.iloc[i].to_numpy()
        assert np.array_equal(np.isnan(live), np.isnan(ref))
        assert np.array_equal(live[~np.isnan(ref)], ref[~np.isnan(ref)])  # bit à bit
        n_eq += 1
    assert n_eq == 500


# ---------------------------------------------------------------- verrou période réservée

def test_holdout_inaccessible_by_default(tmp_path):
    t = pd.date_range(DEV_END - 10 * H8, periods=30, freq="8h").as_unit("ns")
    df = pd.DataFrame({"fundingTime": t, "fundingRate": np.linspace(0, 1e-3, 30)})
    p = tmp_path / "f.parquet"
    data.write_parquet(df, p)
    dev = funding.load_funding(p)
    assert len(dev) and dev["fundingTime"].max() <= DEV_END
    assert len(dev) == int((t <= DEV_END).sum())
    for flag in (False, 1, "yes", None):
        assert funding.load_funding(p, allow_holdout=flag)["fundingTime"].max() <= DEV_END
    full = funding.load_funding(p, allow_holdout=True)
    assert full["fundingTime"].max() > DEV_END and len(full) == 30
    funding.assert_no_holdout(dev)
    with pytest.raises(funding.HoldoutAccessError):
        funding.assert_no_holdout(full)
