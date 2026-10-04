"""Avocat du diable P3-A3, T8 (fuite) pour E24 (b4.py « single », label triple barrière vol).

Usage :
  .venv\\Scripts\\python tests\\adversarial\\leak_e24.py static            (c) label TB / vol_168 + (d) purge
  .venv\\Scripts\\python tests\\adversarial\\leak_e24.py fold K            (b) empoisonnement du futur, fold K (seed 31+K)
Données : load_dataset() par défaut (index max <= 2025-09-30 23:00 UTC).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import math  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import b4  # noqa: E402
import harness  # noqa: E402
import labels_b4  # noqa: E402
from bot.features import compute_features  # noqa: E402
from bot.split import H1, load_dataset, make_folds, purge_mask  # noqa: E402

MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")
CFG = json.loads((ROOT / "experiments" / "configs" / "E24.json").read_text(encoding="utf-8"))
H = 24


def load():
    df = load_dataset()
    assert df.index.max() <= MAX_INDEX, "données réservées"
    print(f"[garde] index max lu = {df.index.max()} ({len(df)} lignes)")
    return df


def static() -> None:
    df = load()
    spec = CFG["label"]
    y0 = labels_b4.make_label(df, spec).to_numpy()
    v0 = compute_features(df)["vol_168"].to_numpy()
    rng = np.random.default_rng(33)
    Ts = np.sort(rng.choice(np.arange(500, len(df) - H - 3), 20, replace=False))
    bad_lab = bad_vol = 0
    for T in Ts:
        # (c1) données tronquées à t+1+H : labels <= T identiques
        y1 = labels_b4.make_label(df.iloc[: T + 2 + H], spec).to_numpy()
        bad_lab += 0 if np.array_equal(y0[: T + 1], y1[: T + 1], equal_nan=True) else 1
        # (c2) bougies > T modifiées : vol_168 <= T identique
        d2 = df.copy()
        d2.iloc[T + 1:, :5] = d2.iloc[T + 1:, :5].to_numpy() * np.exp(rng.normal(0, 0.05, (len(df) - T - 1, 1)))
        v2 = compute_features(d2)["vol_168"].to_numpy()
        bad_vol += 0 if np.array_equal(v0[: T + 1], v2[: T + 1], equal_nan=True) else 1
    print(f"(c1) label TB E24 : 20 troncatures à t+1+H (seed 33) -> labels <= t modifiés : {bad_lab}/20")
    print(f"(c2) vol_168 : 20 perturbations des bougies > t -> vol_168 <= t modifiée : {bad_vol}/20")

    c = b4.prep(CFG, 0)
    X = harness.build_matrix(df, b4.ALL_FEATS)
    Y = {"E24": labels_b4.make_label(df, spec)}
    worst = {"inner_label_end_minus_val_start_h": -math.inf, "final_label_end_minus_test_start_h": -math.inf,
             "val_open_max_minus_test_start_h": -math.inf}
    for f in make_folds(horizon=H):
        fH, Xtr, ytr, feats, _ = b4.config_train(df, X, Y, c, f.k, None)
        t = Xtr.index
        n = len(t)
        v0_ = int(math.floor((1.0 - harness.VAL_FRAC) * n))
        vs, ve = t[v0_], f.train_end
        inner = (np.arange(n) < v0_) & purge_mask(t, vs, ve, horizon=H)
        le_in = (t[inner].max() + (1 + H) * H1 - vs) / H1
        le_f = (t.max() + (1 + H) * H1 - f.test_start) / H1
        ov = df["open"].loc[vs: ve + H1]
        lo = (ov.index.max() - f.test_start) / H1
        worst["inner_label_end_minus_val_start_h"] = max(worst["inner_label_end_minus_val_start_h"], le_in)
        worst["final_label_end_minus_test_start_h"] = max(worst["final_label_end_minus_test_start_h"], le_f)
        worst["val_open_max_minus_test_start_h"] = max(worst["val_open_max_minus_test_start_h"], lo)
        print(f"(d) fold {f.k} : train {t[0]} -> {t.max()} ; val {vs} -> {ve} ; test_start {f.test_start} ; "
              f"fin label inner - val_start {le_in:+.0f} h ; fin label train - test_start {le_f:+.0f} h ; "
              f"dernier open validation - test_start {lo:+.0f} h")
    print("(d) pire cas (heures, doit être < 0) : " + json.dumps(worst))
    print(f"(d) RÉSULTAT : {all(v < 0 for v in worst.values())}")


def poisoned(df, cut, seed):
    rng = np.random.default_rng(seed)
    pois = df.copy()
    m = pois.index >= cut
    k = int(m.sum())
    fac = np.exp(np.cumsum(rng.normal(0, 0.05, k)))
    for col in ("open", "high", "low", "close"):
        pois.loc[m, col] = pois.loc[m, col].to_numpy() * fac * rng.uniform(0.9, 1.1, k)
    pois.loc[m, "volume"] = rng.permutation(pois.loc[m, "volume"].to_numpy())
    return pois


def fold(k: int) -> None:
    df = load()
    f = make_folds(horizon=H)[k - 1]
    pois = poisoned(df, f.test_start, 31 + k)
    c = b4.prep(CFG, 0)
    log = harness.Tee()
    res = []
    for d in (df, pois):
        X = harness.build_matrix(d, b4.ALL_FEATS)
        Y = {"E24": labels_b4.make_label(d, CFG["label"])}
        res.append((X, b4.run_fold_single(f, d, X, Y, c, 42, None, log)))
    (Xc, (rc, mc)), (_, (rp, mp)) = res
    keys = ("best_hp", "best_map_rel", "best_map", "best_val_sharpe", "best_val_trades", "n_train", "auc_val",
            "auc_train", "val_start", "train_start")
    diff = [kk for kk in keys if rc[kk] != rp[kk]]
    tm = (df.index >= f.test_start) & (df.index <= f.test_end)
    pc = harness.predict_proba(mc, Xc.loc[tm, rc["chosen_features"]])
    pp = harness.predict_proba(mp, Xc.loc[tm, rc["chosen_features"]])
    same_model = np.array_equal(pc, pp, equal_nan=True)
    print(f"(b) fold {k} (seed poison {31 + k}) : opens modifiés {int((pois['open'] != df['open']).sum())} "
          f"(>= {f.test_start}) ; champs différents {diff} ; choix {rc['best_hp']} {rc['best_map']} ; "
          f"probas modèle final identiques sur features propres du test {same_model}")
    print(f"(b) fold {k} RÉSULTAT : {not diff and same_model}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    if sys.argv[1] == "static":
        static()
    else:
        fold(int(sys.argv[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
