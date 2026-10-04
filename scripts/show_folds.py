"""Affiche les folds walk-forward (période dev uniquement).

Usage : .venv\\Scripts\\python scripts\\show_folds.py
"""

from __future__ import annotations

import sys

import pandas as pd

from bot.features import FEATURES, compute_features
from bot.labels import HORIZON, compute_labels
from bot.split import DEV_END, EMBARGO, PURGE, load_dataset, make_folds, split_fold


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    df = load_dataset()  # dev uniquement
    assert df.index.max() <= DEV_END
    feats = compute_features(df)
    labs = compute_labels(df)
    full = pd.concat([feats, labs], axis=1)
    usable = full.dropna(subset=FEATURES + ["label"])
    print(f"dataset dev : {df.index.min()} -> {df.index.max()} ({len(df)} lignes, "
          f"{len(usable)} utilisables sans NaN)")
    print(f"H={HORIZON}  purge={PURGE} bougies  embargo={EMBARGO} bougies")
    hdr = (f"{'fold':>4} | {'train début':<16} | {'train fin':<16} | {'test début':<16} | "
           f"{'test fin':<16} | {'n train':>7} | {'n test':>6} | {'écart h':>7} | "
           f"{'y=1 train':>9} | {'y=1 test':>8}")
    print(hdr)
    print("-" * len(hdr))
    fmt = "%Y-%m-%d %H:%M"
    max_row = pd.Timestamp.min.tz_localize("UTC")
    ok = True
    for f in make_folds():
        tr, te = split_fold(usable, f)
        gap_h = (te.index.min() - tr.index.max()) / pd.Timedelta(hours=1)
        overlap = len(tr.index.intersection(te.index))
        ok &= overlap == 0 and gap_h >= HORIZON + 1 and tr.index.max() <= f.train_end
        max_row = max(max_row, te.index.max(), tr.index.max())
        print(f"{f.k:>4} | {f.train_start.strftime(fmt):<16} | {f.train_end.strftime(fmt):<16} | "
              f"{f.test_start.strftime(fmt):<16} | {f.test_end.strftime(fmt):<16} | "
              f"{len(tr):>7} | {len(te):>6} | {gap_h:>7.0f} | "
              f"{tr['label'].mean():>9.4f} | {te['label'].mean():>8.4f}")
    print("-" * len(hdr))
    print(f"chevauchement train/test : {'aucun' if ok else 'PROBLÈME'}")
    print(f"dernière ligne utilisée  : {max_row} (<= {DEV_END} : {max_row <= DEV_END})")
    print("note : 'train fin' = dernière ligne autorisée après purge ; "
          "'écart h' = premier test - dernière ligne train utilisable")
    return 0 if ok and max_row <= DEV_END else 1


if __name__ == "__main__":
    sys.exit(main())
