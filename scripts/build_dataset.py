"""Construit data/processed/btcusdt_1h.parquet depuis data.binance.vision.

Usage : .venv\\Scripts\\python scripts\\build_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

from bot import data

ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = ROOT / "data" / "raw"
OUT = ROOT / "data" / "processed" / "btcusdt_1h.parquet"


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    clean, rep, sources = data.build(RAW_DIR, OUT)
    n_monthly = sum(1 for s in sources if s.kind == "monthly")
    n_daily_months = [s.month for s in sources if s.kind == "daily"]
    n_archives = sum(len(s.archives) for s in sources)
    n_dl = sum(1 for s in sources for a in s.archives if a.status == "downloaded")
    print("-" * 60)
    print(f"mois couverts           : {len(sources)} ({sources[0].month} -> {sources[-1].month})")
    print(f"archives mensuelles     : {n_monthly}")
    print(f"mois en archives jour.  : {len(n_daily_months)} {n_daily_months}")
    print(f"archives total          : {n_archives} (téléchargées: {n_dl}, présentes: {n_archives - n_dl})")
    print(f"checksums OK            : {n_archives}/{n_archives}")
    print(f"fichiers ms / us        : {rep.units['ms']} / {rep.units['us']}")
    print(f"lignes brutes           : {rep.rows_raw}")
    print(f"hors période retirées   : {rep.out_of_range_removed}")
    print(f"non clôturées retirées  : {rep.not_closed_removed}")
    print(f"doublons supprimés      : {rep.duplicates_removed} (conflits de valeurs: {rep.duplicate_conflicts})")
    print(f"lignes (grille horaire) : {rep.rows}")
    print(f"première bougie         : {clean.index[0]}")
    print(f"dernière bougie         : {clean.index[-1]}")
    print(f"bougies manquantes      : {rep.missing} en {len(rep.gaps)} trou(s)")
    big = [g for g in rep.gaps if g[2] > 3]
    print(f"trous > 3 h             : {len(big)}")
    for a, b, n in big:
        print(f"    {a} -> {b}  ({n} h)")
    print(f"violations cohérence    : {sum(rep.violations.values())} {rep.violations}")
    print(f"parquet                 : {OUT.relative_to(ROOT)}")
    print(f"sha256 parquet          : {data.sha256_file(OUT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
