import hashlib
import io
import urllib.error
import zipfile

import numpy as np
import pandas as pd
import pytest

from bot import data


def _row(ot, o=100.0, h=101.0, l=99.0, c=100.5, v=10.0, unit_mult=1):
    ct = ot + 3_600_000 * unit_mult - 1
    return f"{ot},{o},{h},{l},{c},{v},{ct},1000.0,5,1.0,100.0,0"


def test_detect_time_unit():
    ms = np.array([1546300800000, 1735689600000])  # 2019-01-01, 2025-01-01 en ms
    us = ms * 1000
    assert data.detect_time_unit(ms) == "ms"
    assert data.detect_time_unit(us) == "us"
    with pytest.raises(ValueError):
        data.detect_time_unit(np.array([ms[0], us[1]]))


def test_parse_ms_and_us_give_same_timestamps():
    ot_ms = 1735689600000  # 2025-01-01 00:00 UTC
    csv_ms = (_row(ot_ms) + "\n" + _row(ot_ms + 3_600_000) + "\n").encode()
    csv_us = (_row(ot_ms * 1000, unit_mult=1000) + "\n"
              + _row((ot_ms + 3_600_000) * 1000, unit_mult=1000) + "\n").encode()
    a, ua = data.parse_klines_csv(csv_ms)
    b, ub = data.parse_klines_csv(csv_us)
    assert (ua, ub) == ("ms", "us")
    assert a["open_time"].tolist() == b["open_time"].tolist()
    assert a["open_time"].iloc[0] == pd.Timestamp("2025-01-01 00:00", tz="UTC")
    # close_time = fin de bougie - 1 unité (1 ms ou 1 µs) : égaux à la ms près
    assert (a["close_time"].dt.floor("ms") == b["close_time"].dt.floor("ms")).all()
    assert a["open_time"].dt.tz is not None


def test_parse_with_header():
    hdr = "open_time,open,high,low,close,volume,close_time,quote_volume,count,tb_base,tb_quote,ignore\n"
    df, unit = data.parse_klines_csv((hdr + _row(1546300800000) + "\n").encode())
    assert len(df) == 1 and unit == "ms"
    assert df["close"].dtype == np.float64


def _raw(times, **over):
    df = pd.DataFrame({
        "open_time": pd.DatetimeIndex(times, tz="UTC"),
        "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5, "volume": 10.0,
    })
    df["close_time"] = df["open_time"] + pd.Timedelta(hours=1) - pd.Timedelta(milliseconds=1)
    for k, v in over.items():
        df[k] = v
    return df


def test_clean_duplicates_gaps_and_no_ffill():
    start = pd.Timestamp("2020-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2020-01-01 09:00", tz="UTC")
    hours = [0, 1, 1, 2, 3, 7, 8, 9]  # doublon à 1h, trou 4h-6h (3 bougies)
    raw = _raw([start + pd.Timedelta(hours=h) for h in hours])
    raw.loc[2, "close"] = 200.0  # doublon en conflit
    clean, rep = data.clean_klines(raw, start, end, now=pd.Timestamp("2030-01-01", tz="UTC"))
    assert rep.duplicates_removed == 1
    assert rep.duplicate_conflicts == 1
    assert len(clean) == 10 == rep.rows
    assert rep.missing == 3
    assert rep.gaps == [(start + pd.Timedelta(hours=4), start + pd.Timedelta(hours=6), 3)]
    assert clean["missing"].tolist() == [False] * 4 + [True] * 3 + [False] * 3
    assert clean.loc[clean["missing"], ["open", "high", "low", "close", "volume"]].isna().all().all()
    assert clean.loc[start + pd.Timedelta(hours=1), "close"] == 100.5  # premier gardé
    assert str(clean.index.tz) == "UTC"
    assert (np.diff(clean.index.asi8) == 3_600 * 10**6 * (1000 if clean.index.unit == "ns" else 1)).all()
    assert list(clean.columns) == ["open", "high", "low", "close", "volume", "missing"]
    assert all(clean[c].dtype == np.float64 for c in ["open", "high", "low", "close", "volume"])


def test_clean_violations_range_and_unclosed():
    start = pd.Timestamp("2020-01-01 00:00", tz="UTC")
    end = pd.Timestamp("2020-01-01 04:00", tz="UTC")
    times = [start + pd.Timedelta(hours=h) for h in range(-1, 6)]
    raw = _raw(times)
    raw.loc[1, "high"] = 99.5  # high < max(open, close)
    raw.loc[2, "low"] = 100.7  # low > min(open, close)
    raw.loc[3, ["open", "low"]] = [-1.0, -2.0]  # prix <= 0
    raw.loc[4, "volume"] = -5.0  # volume < 0
    now = start + pd.Timedelta(hours=4, minutes=30)  # bougie 04:00 non clôturée
    clean, rep = data.clean_klines(raw, start, end, now=now)
    assert rep.out_of_range_removed == 2
    assert rep.not_closed_removed == 1
    assert rep.violations == {"high<max(open,close)": 1, "low>min(open,close)": 1,
                              "prix<=0": 1, "volume<0": 1}
    assert bool(clean["missing"].iloc[-1]) is True  # 04:00 exclue car non clôturée


def test_misaligned_raises():
    start = pd.Timestamp("2020-01-01 00:00", tz="UTC")
    raw = _raw([start, start + pd.Timedelta(minutes=30)])
    with pytest.raises(ValueError):
        data.clean_klines(raw, start, start + pd.Timedelta(hours=1),
                          now=pd.Timestamp("2030-01-01", tz="UTC"))


def _make_archive(tmp_path, name, good=True):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr(name.replace(".zip", ".csv"), _row(1546300800000) + "\n")
    payload = buf.getvalue()
    path = tmp_path / name
    path.write_bytes(payload)
    sha = hashlib.sha256(payload).hexdigest()
    if not good:
        sha = ("0" if sha[0] != "0" else "1") + sha[1:]
    path.with_name(name + ".CHECKSUM").write_text(f"{sha}  {name}\n", encoding="ascii")
    return data.Archive(name, "http://invalid.example/" + name, path)


def test_present_archive_not_redownloaded(tmp_path, monkeypatch):
    def no_network(*a, **k):
        raise AssertionError("réseau appelé alors que l'archive est présente")
    monkeypatch.setattr(data, "_http_get", no_network)
    arch = _make_archive(tmp_path, "BTCUSDT-1h-2019-01.zip")
    assert data.fetch_archive(arch).status == "present"
    df, unit = data.read_archive(arch.path)
    assert len(df) == 1 and unit == "ms"


def test_bad_checksum_fails_explicitly(tmp_path, monkeypatch):
    monkeypatch.setattr(data, "_http_get", lambda *a, **k: (_ for _ in ()).throw(AssertionError()))
    arch = _make_archive(tmp_path, "BTCUSDT-1h-2019-02.zip", good=False)
    with pytest.raises(data.ChecksumError):
        data.fetch_archive(arch)


def test_bad_checksum_on_download(tmp_path, monkeypatch):
    name = "BTCUSDT-1h-2019-03.zip"
    payloads = {
        "http://x/" + name + ".CHECKSUM": f"{'a' * 64}  {name}".encode(),
        "http://x/" + name: b"contenu",
    }
    monkeypatch.setattr(data, "_http_get", lambda url, **k: payloads[url])
    arch = data.Archive(name, "http://x/" + name, tmp_path / name)
    with pytest.raises(data.ChecksumError):
        data.fetch_archive(arch)
    assert not arch.path.exists()


def test_months_range():
    m = data.months()
    assert len(m) == 92 and m[0] == "2019-01" and m[-1] == "2026-08"
    assert data.END == pd.Timestamp("2026-08-31 23:00", tz="UTC")


def _http_404(url, **k):
    raise urllib.error.HTTPError(url, 404, "Not Found", None, None)


def test_missing_monthly_archive_fails_without_fallback(tmp_path, monkeypatch):
    """404 sur la mensuelle -> MissingArchiveError, aucune journalière lue/téléchargée."""
    calls = []

    def fake_get(url, **k):
        calls.append(url)
        return _http_404(url)

    monkeypatch.setattr(data, "_http_get", fake_get)
    month = "2026-09"
    # journalières complètes et valides présentes en local : elles doivent être ignorées
    (tmp_path / "daily").mkdir()
    for d in range(1, 31):
        _make_archive(tmp_path / "daily", f"BTCUSDT-1h-{month}-{d:02d}.zip")
    with pytest.raises(data.MissingArchiveError):
        data.fetch_month(month, tmp_path)
    assert calls and all("/monthly/" in u for u in calls)
    assert not any("/daily/" in u for u in calls)
    assert not (tmp_path / "monthly" / f"BTCUSDT-1h-{month}.zip").exists()


def test_build_fails_when_a_monthly_archive_is_missing(tmp_path, monkeypatch):
    """build() échoue explicitement dès qu'un mois manque, sans écrire de parquet."""
    (tmp_path / "monthly").mkdir()
    _make_archive(tmp_path / "monthly", "BTCUSDT-1h-2019-01.zip")
    monkeypatch.setattr(data, "months", lambda: ["2019-01", "2019-02"])
    monkeypatch.setattr(data, "_http_get", _http_404)
    out = tmp_path / "out.parquet"
    with pytest.raises(data.MissingArchiveError, match="2019-02"):
        data.build(tmp_path, out, log=lambda *a: None)
    assert not out.exists()
