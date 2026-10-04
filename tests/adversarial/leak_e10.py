"""Avocat du diable P3-A1, test 8 (fuite), dynamique.

Usage : .venv\\Scripts\\python tests\\adversarial\\leak_e10.py [--seed 7]
(a) features / labels : troncature -> valeurs à t inchangées (features) ; label à t n'utilise que open <= t+1+H.
(b) empoisonnement du futur : toutes les bougies >= 2023-01-01 00:00 (début du test du fold 5) sont
    remplacées par des prix aléatoires ; la procédure E10 (harness.run_fold) des folds 1..5 doit donner
    exactement les mêmes choix (hp, mapping, scores de la grille) et des modèles identiques
    (probas identiques sur les features propres du test du fold 5).
Données : load_dataset() par défaut (index max <= 2025-09-30 23:00).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import harness  # noqa: E402
from bot.features import compute_features  # noqa: E402
from bot.labels import compute_labels  # noqa: E402
from bot.split import load_dataset, make_folds  # noqa: E402

MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()
    df = load_dataset()
    assert df.index.max() <= MAX_INDEX
    print(f"[garde] index max lu = {df.index.max()}")
    rng = np.random.default_rng(args.seed)

    # (a) troncature
    full = compute_features(df)
    worst = 0.0
    for T in rng.choice(np.arange(500, len(df) - 1), 20, replace=False):
        tr = compute_features(df.iloc[: T + 1])
        a, b = full.iloc[: T + 1].to_numpy(), tr.to_numpy()
        same = np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) == 0
        worst = max(worst, 0.0 if same else 1.0)
    print(f"(a) features : 20 troncatures aléatoires, valeurs <= t identiques bit à bit : {worst == 0.0}")
    H = 24
    lab = compute_labels(df, H)["label"]
    df2 = df.copy()
    T = 30000
    df2.iloc[T + 2 + H:, df2.columns.get_loc("open")] *= 1.5  # open après t+1+H modifiés
    lab2 = compute_labels(df2, H)["label"]
    print(f"(a) labels : open > t+1+H modifiés, label[<=T] identiques : "
          f"{lab.iloc[:T + 1].equals(lab2.iloc[:T + 1])}")
    folds = make_folds(horizon=H)
    for f in folds:
        tr_df, _ = harness.split_fold(df, f, horizon=H)
        y = lab.loc[tr_df.index].dropna()
        last_open_used = y.index.max() + (1 + H) * harness.H1
        assert last_open_used < f.test_start
    print("(a) folds : pour chaque fold, dernier open utilisé par un label de train < test_start : True")

    # (b) empoisonnement du futur
    cut = folds[4].test_start
    pois = df.copy()
    m = pois.index >= cut
    k = int(m.sum())
    fac = np.exp(np.cumsum(rng.normal(0, 0.05, k)))
    for c in ("open", "high", "low", "close"):
        pois.loc[m, c] = pois.loc[m, c].to_numpy() * fac * rng.uniform(0.9, 1.1, k)
    pois.loc[m, "volume"] = rng.permutation(pois.loc[m, "volume"].to_numpy())
    cfg = json.loads((ROOT / "experiments" / "configs" / "E10.json").read_text(encoding="utf-8"))
    hps, maps = harness.hp_grid(cfg), harness.mapping_grid(cfg)
    feats = harness.resolve_features(cfg["features"])
    log = harness.Tee()
    Xc, yc = harness.build_matrix(df, feats), compute_labels(df, H)["label"]
    Xp, yp = harness.build_matrix(pois, feats), compute_labels(pois, H)["label"]
    tmask = (df.index >= folds[4].test_start) & (df.index <= folds[4].test_end)
    ok_all = True
    for f in folds[:5]:
        rc = harness.run_fold(f, df, Xc, yc, cfg, hps, maps, 42, None, log)
        rp = harness.run_fold(f, pois, Xp, yp, cfg, hps, maps, 42, None, log)
        same_choice = rc["best_hp"] == rp["best_hp"] and rc["best_map"] == rp["best_map"]
        same_grid = rc["grid_scores"] == rp["grid_scores"]
        pc = harness.predict_proba(rc["_model"], Xc.loc[tmask])
        pp = harness.predict_proba(rp["_model"], Xc.loc[tmask])
        same_model = np.array_equal(pc, pp, equal_nan=True)
        ok_all &= same_choice and same_grid and same_model
        print(f"(b) fold {f.k} : choix identiques {same_choice}, 288 scores identiques {same_grid}, "
              f"probas du modèle sur X propre (test fold 5) identiques {same_model}")
    print(f"(b) futur empoisonné à partir de {cut} : aucune influence sur les folds 1..5 : {ok_all}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
