"""Découpage dev / holdout, walk-forward expanding avec purge, normalisation.

- Dev     : 2019-01-01 00:00 -> 2025-09-30 23:00 UTC (seule période accessible par défaut).
- Holdout : 2025-10-01 00:00 -> 2026-08-31 23:00 UTC, accessible uniquement avec
  `allow_holdout=True` explicite (réservé au palier 4).
- Walk-forward expanding : 9 folds de test, train depuis 2019-01-01.
- Purge : retrait du train de toute ligne t dont la fenêtre de label
  [t+1, t+1+H] chevauche [test_start, test_end] (soit les H+1 bougies
  précédant le test).
- Embargo : retrait du train des EMBARGO bougies qui suivent la fin du test.
  En expanding window, le train précède toujours le test : l'embargo est sans
  effet sur les folds. Il s'applique à la frontière dev -> holdout : les
  EMBARGO premières bougies du holdout ne sont pas évaluées
  (évaluation à partir de HOLDOUT_EVAL_START).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from bot.labels import HORIZON

H1 = pd.Timedelta(hours=1)
DEV_START = pd.Timestamp("2019-01-01 00:00", tz="UTC")
DEV_END = pd.Timestamp("2025-09-30 23:00", tz="UTC")
HOLDOUT_START = pd.Timestamp("2025-10-01 00:00", tz="UTC")
HOLDOUT_END = pd.Timestamp("2026-08-31 23:00", tz="UTC")
PURGE = HORIZON + 1  # bougies retirées avant chaque début de test
EMBARGO = 24  # bougies
HOLDOUT_EVAL_START = HOLDOUT_START + EMBARGO * H1

DEFAULT_DATASET = Path(__file__).resolve().parents[2] / "data" / "processed" / "btcusdt_1h.parquet"

_TEST_PERIODS = [
    ("2021-01-01", "2021-06-30"),
    ("2021-07-01", "2021-12-31"),
    ("2022-01-01", "2022-06-30"),
    ("2022-07-01", "2022-12-31"),
    ("2023-01-01", "2023-06-30"),
    ("2023-07-01", "2023-12-31"),
    ("2024-01-01", "2024-06-30"),
    ("2024-07-01", "2024-12-31"),
    ("2025-01-01", "2025-09-30"),
]


class HoldoutAccessError(PermissionError):
    """Accès au holdout sans autorisation explicite."""


# ---------------------------------------------------------------- dev / holdout

def dev_period(df: pd.DataFrame) -> pd.DataFrame:
    """Lignes de la période dev uniquement."""
    return df.loc[(df.index >= DEV_START) & (df.index <= DEV_END)]


def holdout_period(df: pd.DataFrame, *, allow_holdout: bool = False) -> pd.DataFrame:
    """Lignes du holdout. Exige allow_holdout=True (sinon HoldoutAccessError)."""
    if allow_holdout is not True:
        raise HoldoutAccessError("holdout inaccessible sans allow_holdout=True")
    return df.loc[(df.index >= HOLDOUT_START) & (df.index <= HOLDOUT_END)]


def load_dataset(path: Path | str = DEFAULT_DATASET, *, allow_holdout: bool = False) -> pd.DataFrame:
    """Charge le parquet. Par défaut : période dev uniquement.

    allow_holdout=True renvoie dev + holdout (palier 4 uniquement).
    """
    df = pd.read_parquet(path, engine="pyarrow")
    if allow_holdout is True:
        return df.loc[(df.index >= DEV_START) & (df.index <= HOLDOUT_END)]
    return dev_period(df)


# ---------------------------------------------------------------- folds

@dataclass(frozen=True)
class Fold:
    k: int
    train_start: pd.Timestamp
    train_end: pd.Timestamp  # dernière ligne de train utilisable (après purge)
    test_start: pd.Timestamp
    test_end: pd.Timestamp


def make_folds(horizon: int = HORIZON) -> list[Fold]:
    folds = []
    for k, (a, b) in enumerate(_TEST_PERIODS, start=1):
        ts = pd.Timestamp(f"{a} 00:00", tz="UTC")
        te = pd.Timestamp(f"{b} 23:00", tz="UTC")
        folds.append(Fold(k, DEV_START, ts - (horizon + 2) * H1, ts, te))
    return folds


def purge_mask(index: pd.DatetimeIndex, test_start: pd.Timestamp, test_end: pd.Timestamp,
               horizon: int = HORIZON, embargo: int = EMBARGO) -> np.ndarray:
    """True pour les lignes autorisées en train vis-à-vis d'un test donné.

    Exclut : le test lui-même, les lignes t dont [t+1, t+1+H] chevauche le
    test (t >= test_start - (H+1)h), et l'embargo (test_end, test_end + embargo h].
    """
    t = index
    label_overlap = (t + (1 + horizon) * H1 >= test_start) & (t + H1 <= test_end)
    in_test = (t >= test_start) & (t <= test_end)
    in_embargo = (t > test_end) & (t <= test_end + embargo * H1)
    return np.asarray(~(label_overlap | in_test | in_embargo))


def split_fold(df: pd.DataFrame, fold: Fold, horizon: int = HORIZON,
               embargo: int = EMBARGO) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(train, test) d'un fold. Refuse toute ligne postérieure à DEV_END."""
    if len(df) and df.index.max() > DEV_END:
        raise HoldoutAccessError("split_fold : données postérieures à DEV_END interdites")
    idx = df.index
    train_m = (idx >= fold.train_start) & (idx < fold.test_start) & purge_mask(
        idx, fold.test_start, fold.test_end, horizon, embargo)
    test_m = (idx >= fold.test_start) & (idx <= fold.test_end)
    return df.loc[train_m], df.loc[test_m]


# ---------------------------------------------------------------- normalisation

class StandardScaler:
    """Centrage-réduction (moyenne / écart-type ddof=0), numpy uniquement.

    fit() ne doit recevoir que le train du fold ; transform() applique les
    statistiques figées. Écart-type nul -> remplacé par 1.
    """

    def __init__(self) -> None:
        self.mean_: np.ndarray | None = None
        self.scale_: np.ndarray | None = None

    def fit(self, x) -> "StandardScaler":
        a = np.asarray(x, dtype=np.float64)
        if a.ndim != 2 or len(a) == 0:
            raise ValueError("fit attend une matrice 2D non vide")
        if np.isnan(a).any():
            raise ValueError("fit : NaN présents (exclure les lignes NaN en amont)")
        self.mean_ = a.mean(axis=0)
        sd = a.std(axis=0)
        self.scale_ = np.where(sd > 0, sd, 1.0)
        return self

    def transform(self, x):
        if self.mean_ is None:
            raise RuntimeError("scaler non ajusté")
        a = np.asarray(x, dtype=np.float64)
        out = (a - self.mean_) / self.scale_
        if isinstance(x, pd.DataFrame):
            return pd.DataFrame(out, index=x.index, columns=x.columns)
        return out
