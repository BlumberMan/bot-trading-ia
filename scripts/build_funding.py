"""Construit data/processed/btcusdt_funding.parquet depuis data.binance.vision.

Funding BTCUSDT USDⓈ-M (archives mensuelles 2020-01 -> 2026-08, SHA256 vérifiés).
Usage : .venv\\Scripts\\python scripts\\build_funding.py

Sur la période réservée, seuls le comptage des règlements et les contrôles de
cohérence du build sont affichés (aucune statistique de taux).
"""

from __future__ import annotations

import sys
import urllib.error
import urllib.request
from pathlib import Path

from bot import data, funding
from bot.split import DEV_END

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw" / "funding"
OUT = ROOT / "data" / "processed" / "btcusdt_funding.parquet"


def _probe(month: str) -> str:
    """Statut HTTP de l'archive d'un mois antérieur à FIRST_MONTH (information seule)."""
    url = funding.monthly_archive(month, RAW_DIR).url
    try:
        req = urllib.request.Request(url, method="HEAD")
        with urllib.request.urlopen(req, timeout=30) as r:
            return str(r.status)
    except urllib.error.HTTPError as e:
        return str(e.code)
    except Exception as e:  # réseau indisponible : information seulement
        return f"non vérifié ({type(e).__name__})"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    for m in funding.PROBED_ABSENT:
        print(f"{m} sonde (avant FIRST_MONTH) : HTTP {_probe(m)}")
    clean, rep = funding.build(RAW_DIR, OUT)
    ft = clean["fundingTime"]
    n_dev = int((ft <= DEV_END).sum())
    print("-" * 60)
    print(f"premier mois disponible : {funding.FIRST_MONTH} (contrat lancé en 2019-09, "
          f"archives {funding.PROBED_ABSENT[0]}..{funding.PROBED_ABSENT[-1]} absentes)")
    print(f"archives mensuelles     : {rep.archives} ({funding.FIRST_MONTH} -> {funding.LAST_MONTH}), "
          f"téléchargées: {rep.downloaded}, présentes: {rep.archives - rep.downloaded}")
    print(f"archives journalières   : 0 (aucun repli)")
    print(f"checksums OK            : {rep.archives}/{rep.archives}")
    print(f"fichiers ms / us        : {rep.units['ms']} / {rep.units['us']}")
    print(f"colonnes                : {list(clean.columns)}")
    print(f"prix de marquage        : {'présent' if rep.has_mark_price else 'absent des archives'}")
    print(f"lignes brutes           : {rep.rows_raw}")
    print(f"hors période retirées   : {rep.out_of_range_removed}")
    print(f"doublons supprimés      : {rep.duplicates_removed} (conflits de valeurs: {rep.duplicate_conflicts})")
    print(f"règlements              : {rep.rows} (dev <= {DEV_END}: {n_dev}, réservée: {rep.rows - n_dev})")
    print(f"premier règlement       : {rep.first}")
    print(f"dernier règlement       : {rep.last}")
    print(f"taux NaN                : {rep.nan_rates}")
    print(f"funding_interval_hours != 8 : {rep.interval_col_not_8}")
    print(f"intervalles 8 h à la gigue près (<= {funding.INTERVAL_TOL}) : "
          f"{rep.jitter_intervals} (gigue max {rep.jitter_max})")
    print(f"intervalles != 8 h      : {len(rep.bad_intervals)}")
    for a, b, d in rep.bad_intervals:
        print(f"    {a} -> {b}  écart {d}")
    print(f"valeurs |rate| > {funding.OUTLIER_ABS:g} : {len(rep.outliers)} (conservées)")
    for t, r in rep.outliers:
        print(f"    {t}  {'dev' if t <= DEV_END else 'réservée'}  {r:.8f}")
    print(f"parquet                 : {OUT.relative_to(ROOT)}")
    print(f"sha256 parquet          : {data.sha256_file(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
