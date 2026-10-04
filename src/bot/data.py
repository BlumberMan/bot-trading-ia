"""Données : téléchargement des klines Binance spot BTCUSDT 1h et nettoyage.

Source : https://data.binance.vision (archives publiques, sans clé API).
Archives mensuelles en priorité ; si l'archive mensuelle d'un mois n'est pas
(encore) publiée (HTTP 404), repli sur les archives journalières du même mois,
chacune vérifiée par son propre fichier .CHECKSUM.
"""

from __future__ import annotations

import calendar
import csv
import hashlib
import io
import time
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

SYMBOL = "BTCUSDT"
INTERVAL = "1h"
BASE_URL = "https://data.binance.vision/data/spot"
START = pd.Timestamp("2019-01-01 00:00", tz="UTC")
END = pd.Timestamp("2026-09-30 23:00", tz="UTC")  # open_time de la dernière bougie
FREQ = pd.Timedelta(hours=1)
PRICE_COLS = ["open", "high", "low", "close"]
OHLCV_COLS = PRICE_COLS + ["volume"]

# Seuil de détection d'unité : un open_time en ms vaut ~1.5e12-1.8e12 pour
# 2019-2026, en µs ~1.5e15-1.8e15. 1e14 sépare sans ambiguïté.
_US_THRESHOLD = 10**14


class ChecksumError(RuntimeError):
    """Le SHA256 d'une archive ne correspond pas à son fichier .CHECKSUM."""


@dataclass
class Archive:
    name: str  # ex. BTCUSDT-1h-2019-01.zip
    url: str
    path: Path
    status: str = ""  # "present" | "downloaded"


@dataclass
class MonthSource:
    month: str  # YYYY-MM
    kind: str  # "monthly" | "daily"
    archives: list[Archive] = field(default_factory=list)


@dataclass
class CleanReport:
    rows_raw: int = 0
    duplicates_removed: int = 0
    duplicate_conflicts: int = 0
    out_of_range_removed: int = 0
    not_closed_removed: int = 0
    rows: int = 0
    missing: int = 0
    gaps: list[tuple[pd.Timestamp, pd.Timestamp, int]] = field(default_factory=list)
    violations: dict[str, int] = field(default_factory=dict)
    units: dict[str, int] = field(default_factory=dict)


# ---------------------------------------------------------------- réseau

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _http_get(url: str, retries: int = 3, timeout: float = 60.0) -> bytes:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise
            last = e
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            last = e
        time.sleep(2 * (attempt + 1))
    raise RuntimeError(f"échec du téléchargement {url}: {last}")


def parse_checksum(text: str, name: str) -> str:
    """Extrait le SHA256 d'un fichier .CHECKSUM ("<sha256>  <nom>")."""
    parts = text.strip().split()
    if len(parts) != 2 or parts[1] != name or len(parts[0]) != 64:
        raise ChecksumError(f"fichier CHECKSUM illisible pour {name}: {text!r}")
    return parts[0].lower()


def _checksum_path(arch: Archive) -> Path:
    return arch.path.with_name(arch.path.name + ".CHECKSUM")


def _archive_valid(arch: Archive) -> bool:
    """True si l'archive et son CHECKSUM sont présents localement et concordent.

    False si l'un des deux est absent. Lève ChecksumError s'ils sont présents
    mais ne concordent pas (aucun écrasement silencieux).
    """
    ck = _checksum_path(arch)
    if not (arch.path.exists() and ck.exists()):
        return False
    expected = parse_checksum(ck.read_text(encoding="ascii"), arch.name)
    actual = sha256_file(arch.path)
    if actual != expected:
        raise ChecksumError(f"{arch.name}: sha256 {actual} != attendu {expected}")
    return True


def fetch_archive(arch: Archive) -> Archive:
    """Télécharge l'archive et son CHECKSUM si absents ; vérifie le SHA256.

    Lève ChecksumError si le SHA256 ne correspond pas (fichier local ou
    fraîchement téléchargé). Lève HTTPError 404 si l'archive n'existe pas.
    """
    if _archive_valid(arch):
        arch.status = "present"
        return arch
    ck_bytes = _http_get(arch.url + ".CHECKSUM")
    expected = parse_checksum(ck_bytes.decode("ascii"), arch.name)
    data = _http_get(arch.url)
    actual = hashlib.sha256(data).hexdigest()
    if actual != expected:
        raise ChecksumError(f"{arch.name}: sha256 {actual} != attendu {expected}")
    arch.path.parent.mkdir(parents=True, exist_ok=True)
    tmp = arch.path.with_name(arch.path.name + ".part")
    tmp.write_bytes(data)
    tmp.replace(arch.path)
    _checksum_path(arch).write_bytes(ck_bytes)
    arch.status = "downloaded"
    return arch


def months(start: pd.Timestamp = START, end: pd.Timestamp = END) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(start.strftime("%Y-%m"),
                                                         end.strftime("%Y-%m"), freq="M")]


def _monthly_archive(month: str, raw_dir: Path) -> Archive:
    name = f"{SYMBOL}-{INTERVAL}-{month}.zip"
    url = f"{BASE_URL}/monthly/klines/{SYMBOL}/{INTERVAL}/{name}"
    return Archive(name, url, raw_dir / "monthly" / name)


def _daily_archives(month: str, raw_dir: Path) -> list[Archive]:
    y, m = map(int, month.split("-"))
    out = []
    for d in range(1, calendar.monthrange(y, m)[1] + 1):
        name = f"{SYMBOL}-{INTERVAL}-{month}-{d:02d}.zip"
        url = f"{BASE_URL}/daily/klines/{SYMBOL}/{INTERVAL}/{name}"
        out.append(Archive(name, url, raw_dir / "daily" / name))
    return out


def fetch_month(month: str, raw_dir: Path) -> MonthSource:
    """Archive mensuelle si disponible ; sinon repli sur les journalières.

    Ordre : mensuelle locale valide > journalières locales complètes et
    valides > téléchargement mensuelle > (404) téléchargement journalières.
    """
    monthly = _monthly_archive(month, raw_dir)
    if _archive_valid(monthly):
        monthly.status = "present"
        return MonthSource(month, "monthly", [monthly])
    daily = _daily_archives(month, raw_dir)
    if all(_archive_valid(a) for a in daily):
        for a in daily:
            a.status = "present"
        return MonthSource(month, "daily", daily)
    try:
        return MonthSource(month, "monthly", [fetch_archive(monthly)])
    except urllib.error.HTTPError as e:
        if e.code != 404:
            raise
    return MonthSource(month, "daily", [fetch_archive(a) for a in daily])


# ---------------------------------------------------------------- lecture

def detect_time_unit(values: np.ndarray) -> str:
    """'ms' ou 'us' selon l'ordre de grandeur ; erreur si mélangé."""
    v = np.asarray(values, dtype=np.int64)
    is_us = v >= _US_THRESHOLD
    if is_us.all():
        return "us"
    if (~is_us).all():
        return "ms"
    raise ValueError("unités de temps mélangées dans un même fichier")


def parse_klines_csv(raw: bytes) -> tuple[pd.DataFrame, str]:
    """Parse un CSV de klines Binance (avec ou sans en-tête).

    Renvoie (DataFrame open_time/close_time UTC + OHLCV float64, unité).
    """
    rows = [r for r in csv.reader(io.StringIO(raw.decode("ascii"))) if r]
    if rows and not rows[0][0].strip().isdigit():
        rows = rows[1:]  # en-tête éventuel
    if not rows:
        raise ValueError("CSV de klines vide")
    ot = np.array([int(r[0]) for r in rows], dtype=np.int64)
    ct = np.array([int(r[6]) for r in rows], dtype=np.int64)
    unit = detect_time_unit(ot)
    if detect_time_unit(ct) != unit:
        raise ValueError("unités open_time / close_time différentes")
    df = pd.DataFrame({
        "open_time": pd.to_datetime(ot, unit=unit, utc=True).as_unit("ns"),
        "close_time": pd.to_datetime(ct, unit=unit, utc=True).as_unit("ns"),
    })
    for i, c in enumerate(OHLCV_COLS, start=1):
        df[c] = np.array([float(r[i]) for r in rows], dtype=np.float64)
    return df, unit


def read_archive(path: Path) -> tuple[pd.DataFrame, str]:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path.name}: {len(names)} CSV dans l'archive")
        return parse_klines_csv(z.read(names[0]))


# ---------------------------------------------------------------- nettoyage

def clean_klines(raw: pd.DataFrame, start: pd.Timestamp = START, end: pd.Timestamp = END,
                 now: pd.Timestamp | None = None) -> tuple[pd.DataFrame, CleanReport]:
    """Nettoie des klines brutes (colonnes open_time, close_time, OHLCV).

    - garde uniquement les bougies clôturées (close_time < now) dans [start, end] ;
    - supprime les doublons d'open_time (compte, et compte les conflits de valeurs) ;
    - réindexe sur la grille horaire complète ; bougies absentes = NaN, missing=True ;
    - compte les violations de cohérence (sans modifier les valeurs).
    """
    rep = CleanReport(rows_raw=len(raw))
    now = pd.Timestamp.now(tz="UTC") if now is None else now
    df = raw.sort_values("open_time", kind="stable").reset_index(drop=True)

    in_range = (df["open_time"] >= start) & (df["open_time"] <= end)
    rep.out_of_range_removed = int((~in_range).sum())
    df = df[in_range]
    closed = df["close_time"] < now
    rep.not_closed_removed = int((~closed).sum())
    df = df[closed]

    dup = df["open_time"].duplicated(keep="first")
    rep.duplicates_removed = int(dup.sum())
    if rep.duplicates_removed:
        nunique = df.groupby("open_time")[OHLCV_COLS].nunique()
        rep.duplicate_conflicts = int((nunique > 1).any(axis=1).sum())
    df = df[~dup]

    misaligned = (df["open_time"] - start) % FREQ != pd.Timedelta(0)
    if misaligned.any():
        raise ValueError(f"{int(misaligned.sum())} open_time hors grille horaire")

    grid = pd.date_range(start, end, freq=FREQ, name="open_time")
    out = df.set_index("open_time")[OHLCV_COLS].reindex(grid).astype("float64")
    out["missing"] = out[OHLCV_COLS].isna().any(axis=1)
    out.loc[out["missing"], OHLCV_COLS] = np.nan
    rep.rows = len(out)
    rep.missing = int(out["missing"].sum())
    rep.gaps = find_gaps(out["missing"])

    ok = ~out["missing"]
    o, h, l, c, v = (out.loc[ok, k] for k in OHLCV_COLS)
    rep.violations = {
        "high<max(open,close)": int((h < np.maximum(o, c)).sum()),
        "low>min(open,close)": int((l > np.minimum(o, c)).sum()),
        "prix<=0": int((out.loc[ok, PRICE_COLS] <= 0).any(axis=1).sum()),
        "volume<0": int((v < 0).sum()),
    }
    return out, rep


def find_gaps(missing: pd.Series) -> list[tuple[pd.Timestamp, pd.Timestamp, int]]:
    """Liste des trous : (première bougie manquante, dernière, longueur en bougies)."""
    m = missing.to_numpy(dtype=bool)
    gaps = []
    i, n = 0, len(m)
    while i < n:
        if m[i]:
            j = i
            while j + 1 < n and m[j + 1]:
                j += 1
            gaps.append((missing.index[i], missing.index[j], j - i + 1))
            i = j + 1
        else:
            i += 1
    return gaps


def build(raw_dir: Path, out_path: Path, log=print) -> tuple[pd.DataFrame, CleanReport, list[MonthSource]]:
    """Télécharge/vérifie toutes les archives, nettoie, écrit le parquet."""
    sources = []
    frames = []
    units: dict[str, int] = {"ms": 0, "us": 0}
    for m in months():
        src = fetch_month(m, raw_dir)
        sources.append(src)
        for a in src.archives:
            df, unit = read_archive(a.path)
            units[unit] += 1
            frames.append(df)
        statuses = ",".join(sorted({a.status for a in src.archives}))
        log(f"{m} {src.kind:7s} {len(src.archives):2d} archive(s) {statuses:10s} checksum OK")
    raw = pd.concat(frames, ignore_index=True)
    clean, rep = clean_klines(raw)
    rep.units = units
    write_parquet(clean, out_path)
    return clean, rep, sources


def write_parquet(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path, engine="pyarrow", compression="snappy", index=True)


def load_parquet(path: Path) -> pd.DataFrame:
    return pd.read_parquet(path, engine="pyarrow")
