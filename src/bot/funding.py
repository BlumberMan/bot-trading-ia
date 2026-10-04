"""Funding du perpétuel BTCUSDT (Binance USDⓈ-M futures) : données, alignement 1 h, features.

Source : https://data.binance.vision/data/futures/um/monthly/fundingRate/BTCUSDT/
(archives publiques, sans clé API). Archives mensuelles uniquement, chacune vérifiée
par son .CHECKSUM (logique de bot.data réutilisée telle quelle). Aucun repli sur
des archives journalières : une archive mensuelle absente (404) fait échouer le build.

Le contrat a été lancé en 2019-09, mais la première archive mensuelle publiée est
2020-01 (2019-09 -> 2019-12 renvoient 404, constaté le 2026-10-05) : FIRST_MONTH.

Le funding ne sert que de feature (on trade le spot). Règle de disponibilité : un
taux réglé à fundingTime n'est utilisable qu'à la clôture d'une bougie 1 h t telle
que fundingTime <= t + 1 h (la bougie t, d'index open_time, clôture à t + 1 h).

Période réservée : load_funding() exclut par défaut tout règlement postérieur à
DEV_END ; allow_holdout=True est exigé pour y accéder (comme bot.split).
"""

from __future__ import annotations

import csv
import io
import urllib.error
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from bot import data
from bot.data import Archive, ChecksumError, MissingArchiveError  # noqa: F401 (réexport)
from bot.split import DEV_END, HOLDOUT_END

SYMBOL = "BTCUSDT"
BASE_URL = "https://data.binance.vision/data/futures/um/monthly/fundingRate"
FIRST_MONTH = "2020-01"  # première archive mensuelle publiée (2019-09..2019-12 : 404)
LAST_MONTH = "2026-08"  # même fin que les prix
PROBED_ABSENT = ["2019-09", "2019-10", "2019-11", "2019-12"]
H1 = pd.Timedelta(hours=1)
EXPECTED_INTERVAL = pd.Timedelta(hours=8)
INTERVAL_TOL = pd.Timedelta(seconds=1)  # gigue de quelques ms tolérée, signalée à part
OUTLIER_ABS = 0.01  # |rate| > 1 % par règlement : compté, jamais supprimé

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "data" / "processed" / "btcusdt_funding.parquet"

# Features : nombre de règlements (le courant inclus) nécessaires à chaque feature.
SPANS: dict[str, int] = {
    "fr_cur": 1,
    "fr_mean3": 3,
    "fr_mean21": 21,
    "fr_z90": 90,
    "fr_sum21": 21,
}
FUNDING_FEATURES: list[str] = list(SPANS)
FUNDING_LOOKBACK = max(SPANS.values())  # 90 règlements (30 jours à 8 h)
SD_MIN = 1e-12  # écart-type minimal pour le z-score (sinon NaN)


class HoldoutAccessError(PermissionError):
    """Accès aux règlements de la période réservée sans autorisation explicite."""


@dataclass
class FundingReport:
    archives: int = 0
    downloaded: int = 0
    rows_raw: int = 0
    units: dict[str, int] = field(default_factory=dict)
    out_of_range_removed: int = 0
    duplicates_removed: int = 0
    duplicate_conflicts: int = 0
    rows: int = 0
    first: pd.Timestamp | None = None
    last: pd.Timestamp | None = None
    # intervalles hors 8 h ± INTERVAL_TOL : (règlement précédent, règlement, écart)
    bad_intervals: list[tuple[pd.Timestamp, pd.Timestamp, pd.Timedelta]] = field(default_factory=list)
    jitter_intervals: int = 0  # 0 < |écart − 8 h| <= INTERVAL_TOL
    jitter_max: pd.Timedelta = pd.Timedelta(0)
    interval_col_not_8: int = 0  # colonne funding_interval_hours != 8
    outliers: list[tuple[pd.Timestamp, float]] = field(default_factory=list)
    nan_rates: int = 0
    has_mark_price: bool = False


# ---------------------------------------------------------------- réseau

def months(first: str = FIRST_MONTH, last: str = LAST_MONTH) -> list[str]:
    return [p.strftime("%Y-%m") for p in pd.period_range(first, last, freq="M")]


def monthly_archive(month: str, raw_dir: Path) -> Archive:
    name = f"{SYMBOL}-fundingRate-{month}.zip"
    url = f"{BASE_URL}/{SYMBOL}/{name}"
    return Archive(name, url, raw_dir / "monthly" / name)


def fetch_month(month: str, raw_dir: Path) -> Archive:
    """Archive mensuelle uniquement (locale valide, sinon téléchargée), SHA256 vérifié.

    Lève MissingArchiveError si l'archive n'existe pas (404), sans aucun repli ;
    ChecksumError si le SHA256 ne concorde pas.
    """
    arch = monthly_archive(month, raw_dir)
    try:
        data.fetch_archive(arch)
    except urllib.error.HTTPError as e:
        if e.code == 404:
            raise MissingArchiveError(
                f"archive mensuelle absente (404) : {arch.url} ; aucun repli") from e
        raise
    return arch


# ---------------------------------------------------------------- lecture

_TIME_COLS = ("calc_time", "fundingTime", "funding_time")
_RATE_COLS = ("last_funding_rate", "fundingRate", "funding_rate")
_MARK_COLS = ("markPrice", "mark_price")
_INTERVAL_COLS = ("funding_interval_hours",)


def _find(header: list[str], names: tuple[str, ...]) -> int | None:
    for n in names:
        if n in header:
            return header.index(n)
    return None


def parse_funding_csv(raw: bytes) -> tuple[pd.DataFrame, str]:
    """Parse un CSV de funding Binance (en-tête obligatoire).

    Colonnes reconnues : calc_time|fundingTime (temps), last_funding_rate|fundingRate
    (taux), markPrice (optionnel), funding_interval_hours (optionnel).
    Renvoie (DataFrame fundingTime UTC ns + fundingRate float64 [+ markPrice]
    [+ funding_interval_hours], unité 'ms' | 'us').
    """
    rows = [r for r in csv.reader(io.StringIO(raw.decode("ascii"))) if r]
    if not rows:
        raise ValueError("CSV de funding vide")
    header = [h.strip() for h in rows[0]]
    it, ir = _find(header, _TIME_COLS), _find(header, _RATE_COLS)
    if it is None or ir is None:
        raise ValueError(f"en-tête de funding non reconnu : {header}")
    im, ii = _find(header, _MARK_COLS), _find(header, _INTERVAL_COLS)
    body = rows[1:]
    if not body:
        raise ValueError("CSV de funding sans ligne")
    t = np.array([int(r[it]) for r in body], dtype=np.int64)
    unit = data.detect_time_unit(t)
    df = pd.DataFrame({
        "fundingTime": pd.to_datetime(t, unit=unit, utc=True).as_unit("ns"),
        "fundingRate": np.array([float(r[ir]) if r[ir].strip() else np.nan for r in body],
                                dtype=np.float64),
    })
    if im is not None:
        df["markPrice"] = np.array([float(r[im]) if r[im].strip() else np.nan for r in body],
                                   dtype=np.float64)
    if ii is not None:
        df["funding_interval_hours"] = np.array([int(r[ii]) for r in body], dtype=np.int64)
    return df, unit


def read_archive(path: Path) -> tuple[pd.DataFrame, str]:
    with zipfile.ZipFile(path) as z:
        names = [n for n in z.namelist() if n.endswith(".csv")]
        if len(names) != 1:
            raise ValueError(f"{path.name}: {len(names)} CSV dans l'archive")
        return parse_funding_csv(z.read(names[0]))


# ---------------------------------------------------------------- nettoyage

def clean_funding(raw: pd.DataFrame, first: str = FIRST_MONTH,
                  last: str = LAST_MONTH) -> tuple[pd.DataFrame, FundingReport]:
    """Trie, garde [début de first, fin de last], supprime les doublons de fundingTime
    (comptés, conflits de valeurs comptés), contrôle les intervalles et compte les
    valeurs aberrantes (|rate| > OUTLIER_ABS) sans les supprimer."""
    rep = FundingReport(rows_raw=len(raw))
    lo = pd.Period(first, freq="M").start_time.tz_localize("UTC")
    hi = (pd.Period(last, freq="M") + 1).start_time.tz_localize("UTC")
    df = raw.sort_values("fundingTime", kind="stable").reset_index(drop=True)
    keep = (df["fundingTime"] >= lo) & (df["fundingTime"] < hi)
    rep.out_of_range_removed = int((~keep).sum())
    df = df[keep]

    dup = df["fundingTime"].duplicated(keep="first")
    rep.duplicates_removed = int(dup.sum())
    if rep.duplicates_removed:
        vcols = [c for c in df.columns if c != "fundingTime"]
        nun = df.groupby("fundingTime")[vcols].nunique(dropna=False)
        rep.duplicate_conflicts = int((nun > 1).any(axis=1).sum())
    df = df[~dup].reset_index(drop=True)

    rep.rows = len(df)
    rep.has_mark_price = "markPrice" in df.columns
    if rep.rows:
        rep.first, rep.last = df["fundingTime"].iloc[0], df["fundingTime"].iloc[-1]
    ft = df["fundingTime"]
    d = ft.diff().iloc[1:]
    dev = (d - EXPECTED_INTERVAL).abs()
    bad = dev > INTERVAL_TOL
    rep.bad_intervals = [(ft.iloc[i - 1], ft.iloc[i], d.loc[i]) for i in d.index[bad.to_numpy()]]
    jit = (dev > pd.Timedelta(0)) & ~bad
    rep.jitter_intervals = int(jit.sum())
    rep.jitter_max = dev[jit].max() if jit.any() else pd.Timedelta(0)
    if "funding_interval_hours" in df.columns:
        rep.interval_col_not_8 = int((df["funding_interval_hours"] != 8).sum())
    out = df["fundingRate"].abs() > OUTLIER_ABS
    rep.outliers = list(zip(ft[out].tolist(), df.loc[out, "fundingRate"].tolist()))
    rep.nan_rates = int(df["fundingRate"].isna().sum())
    return df, rep


def build(raw_dir: Path, out_path: Path, log=print) -> tuple[pd.DataFrame, FundingReport]:
    """Télécharge/vérifie toutes les archives mensuelles, nettoie, écrit le parquet."""
    frames = []
    units = {"ms": 0, "us": 0}
    n_dl = 0
    ms = months()
    for m in ms:
        a = fetch_month(m, raw_dir)
        df, unit = read_archive(a.path)
        units[unit] += 1
        n_dl += a.status == "downloaded"
        frames.append(df)
        log(f"{m} monthly {a.status:10s} {len(df):3d} règlements checksum OK")
    raw = pd.concat(frames, ignore_index=True)
    clean, rep = clean_funding(raw)
    rep.archives, rep.downloaded, rep.units = len(ms), n_dl, units
    data.write_parquet(clean, out_path)
    return clean, rep


# ---------------------------------------------------------------- verrou

def load_funding(path: Path | str = DEFAULT_PATH, *, allow_holdout: bool = False) -> pd.DataFrame:
    """Charge le parquet de funding.

    Par défaut : uniquement les règlements avec fundingTime <= DEV_END
    (2025-09-30 23:00 UTC) ; rien de la période réservée. allow_holdout=True
    (strictement True) renvoie aussi la période réservée, jusqu'à la clôture de la
    dernière bougie du holdout (HOLDOUT_END + 1 h).
    """
    df = pd.read_parquet(path, engine="pyarrow")
    if allow_holdout is True:
        return df.loc[df["fundingTime"] <= HOLDOUT_END + H1].reset_index(drop=True)
    return df.loc[df["fundingTime"] <= DEV_END].reset_index(drop=True)


def assert_no_holdout(funding: pd.DataFrame) -> None:
    """Lève HoldoutAccessError si un règlement est postérieur à DEV_END."""
    if len(funding) and funding["fundingTime"].max() > DEV_END:
        raise HoldoutAccessError("règlements postérieurs à DEV_END sans allow_holdout=True")


# ---------------------------------------------------------------- alignement 1 h

def _ns(values) -> np.ndarray:
    return pd.DatetimeIndex(values).as_unit("ns").asi8


def _positions(funding_times, index_1h: pd.DatetimeIndex) -> np.ndarray:
    """Pour chaque bougie t : position du dernier règlement avec fundingTime <= t + 1 h
    (-1 s'il n'y en a aucun). funding_times doit être trié croissant."""
    ft = _ns(funding_times)
    if len(ft) > 1 and (np.diff(ft) <= 0).any():
        raise ValueError("fundingTime doit être strictement croissant (nettoyer avant)")
    close = _ns(pd.DatetimeIndex(index_1h) + H1)
    return np.searchsorted(ft, close, side="right") - 1


def funding_on_grid(funding: pd.DataFrame, index_1h: pd.DatetimeIndex) -> pd.DataFrame:
    """Dernier taux réglé disponible à la clôture de chaque bougie 1 h t.

    Colonnes : fundingRate (taux du dernier règlement avec fundingTime <= t + 1 h),
    fundingTime (horodatage de ce règlement), age_h (délai en heures entre ce
    règlement et la clôture t + 1 h, >= 0). NaN / NaT avant le premier règlement.
    """
    idx = pd.DatetimeIndex(index_1h)
    pos = _positions(funding["fundingTime"], idx)
    ok = pos >= 0
    rates = funding["fundingRate"].to_numpy(dtype=np.float64)
    ft = _ns(funding["fundingTime"])
    rate = np.full(len(idx), np.nan)
    rate[ok] = rates[pos[ok]]
    used = np.zeros(len(idx), dtype=np.int64)
    used[ok] = ft[pos[ok]]
    used_ts = pd.DatetimeIndex(pd.to_datetime(used, utc=True)).as_unit("ns").where(ok, pd.NaT)
    age = np.full(len(idx), np.nan)
    age[ok] = (_ns(idx + H1)[ok] - used[ok]) / 3.6e12
    return pd.DataFrame({"fundingRate": rate, "fundingTime": used_ts, "age_h": age}, index=idx)


# ---------------------------------------------------------------- features

def funding_features_core(rates: np.ndarray) -> np.ndarray:
    """Cœur unique (lot et direct) : features au dernier règlement de `rates`.

    `rates` = taux des derniers règlements disponibles, ordre chronologique, le
    dernier élément étant le règlement courant. Seuls les FUNDING_LOOKBACK (90)
    derniers sont lus. Une feature dont la fenêtre dépasse le nombre de règlements
    fournis, ou contient un NaN, vaut NaN. Sommes dans un ordre fixe (np.sum sur
    la même tranche contiguë) : résultat identique bit à bit en lot et en direct.

    Renvoie [fr_cur, fr_mean3, fr_mean21, fr_z90, fr_sum21].
    """
    r = np.ascontiguousarray(np.asarray(rates, dtype=np.float64)[-FUNDING_LOOKBACK:])
    n = len(r)
    out = np.full(len(FUNDING_FEATURES), np.nan)
    if n == 0:
        return out
    out[0] = r[-1]
    if n >= 3:
        out[1] = np.sum(r[-3:]) / 3.0
    if n >= 21:
        s21 = np.sum(r[-21:])
        out[2] = s21 / 21.0
        out[4] = s21
    if n >= 90:
        w = r[-90:]
        m = np.sum(w) / 90.0
        sd = np.sqrt(np.sum((w - m) ** 2) / 89.0)  # ddof = 1
        # taux publiés à 1e-8 près : un écart-type < SD_MIN ne peut venir que de l'arrondi
        # flottant d'une fenêtre constante (ex. 90 x 0.0001) -> NaN
        out[3] = (r[-1] - m) / sd if sd > SD_MIN else np.nan
    return out


def funding_features_per_settlement(funding: pd.DataFrame) -> np.ndarray:
    """Lot : features à chaque règlement i (fenêtre = règlements [i-89, i])."""
    r = funding["fundingRate"].to_numpy(dtype=np.float64)
    out = np.full((len(r), len(FUNDING_FEATURES)), np.nan)
    for i in range(len(r)):
        out[i] = funding_features_core(r[max(0, i - FUNDING_LOOKBACK + 1): i + 1])
    return out


def funding_features(funding: pd.DataFrame, index_1h: pd.DatetimeIndex) -> pd.DataFrame:
    """Lot (backtest) : features de funding sur la grille 1 h, + fr_age_h.

    La valeur à t est celle du dernier règlement avec fundingTime <= t + 1 h.
    NaN avant le premier règlement.
    """
    idx = pd.DatetimeIndex(index_1h)
    per = funding_features_per_settlement(funding)
    pos = _positions(funding["fundingTime"], idx)
    ok = pos >= 0
    vals = np.full((len(idx), len(FUNDING_FEATURES)), np.nan)
    if len(per):
        vals[ok] = per[pos[ok]]
    out = pd.DataFrame(vals, index=idx, columns=FUNDING_FEATURES)
    out["fr_age_h"] = funding_on_grid(funding, idx)["age_h"].to_numpy()
    return out


def funding_features_live(recent: pd.DataFrame, t: pd.Timestamp) -> pd.Series:
    """Direct : features à la clôture de la bougie t (index open_time).

    `recent` = derniers règlements connus (au moins les FUNDING_LOOKBACK derniers
    avec fundingTime <= t + 1 h pour égaler le lot). Tout règlement avec
    fundingTime > t + 1 h est ignoré.
    """
    t = pd.Timestamp(t)
    rec = recent.sort_values("fundingTime", kind="stable")
    rec = rec.loc[rec["fundingTime"] <= t + H1]
    r = rec["fundingRate"].to_numpy(dtype=np.float64)[-FUNDING_LOOKBACK:]
    vals = funding_features_core(r)
    s = pd.Series(vals, index=FUNDING_FEATURES, name=t)
    if len(rec):
        last = pd.Timestamp(rec["fundingTime"].iloc[-1])
        s["fr_age_h"] = ((t + H1).as_unit("ns").value - last.as_unit("ns").value) / 3.6e12
    else:
        s["fr_age_h"] = np.nan
    return s
