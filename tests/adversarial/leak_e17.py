"""Avocat du diable P3-A2, test 8 (fuite) pour la procédure emboîtée E17.

Usage :
  .venv\\Scripts\\python tests\\adversarial\\leak_e17.py static            (c) features extra + (d) purge/validation
  .venv\\Scripts\\python tests\\adversarial\\leak_e17.py fold K [--seed S] (b) empoisonnement du futur, fold K
(b) toutes les bougies >= test_start(K) remplacées par des prix / volumes aléatoires (seed S + K) ;
    nested.run_fold(K) doit donner les mêmes 16 records de validation (scores, hp, mapping, AUC, effectifs),
    le même choix et un modèle final identique (probas identiques sur les features PROPRES du test K).
(c) compute_extra : 20 troncatures aléatoires (seed S), valeurs <= t identiques bit à bit.
(d) pour chaque fold et chaque configuration de l'union : bornes des labels (t + 1 + H_c) du train interne,
    de la validation et du train final ; dernier open lu par le backtest de validation.
Données : load_dataset() par défaut (index max <= 2025-09-30 23:00 UTC).
"""

from __future__ import annotations

import json
import math
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import harness  # noqa: E402
import nested  # noqa: E402
from bot.features import FEATURES  # noqa: E402
from bot.labels import compute_labels  # noqa: E402
from bot.split import H1, load_dataset, make_folds, purge_mask, split_fold  # noqa: E402
from extra_features import EXTRA_FEATURES, compute_extra  # noqa: E402

MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")
CFG17 = json.loads((ROOT / "experiments" / "configs" / "E17.json").read_text(encoding="utf-8"))


def mats(df, union):
    X = harness.build_matrix(df, list(FEATURES) + list(EXTRA_FEATURES))
    y = {H: compute_labels(df, horizon=H)["label"] for H in sorted({c["H"] for c in union} | {24})}
    return X, y


def static(seed: int) -> None:
    df = load_dataset()
    assert df.index.max() <= MAX_INDEX
    print(f"[garde] index max lu = {df.index.max()}")
    rng = np.random.default_rng(seed)
    full = compute_extra(df)
    bad = 0
    Ts = rng.choice(np.arange(500, len(df) - 1), 20, replace=False)
    for T in Ts:
        tr = compute_extra(df.iloc[: T + 1])
        a, b = full.iloc[: T + 1].to_numpy(), tr.to_numpy()
        same = np.array_equal(np.isnan(a), np.isnan(b)) and np.nanmax(np.abs(a - b)) == 0
        bad += 0 if same else 1
    print(f"(c) features extra : 20 troncatures (seed {seed}), troncatures en écart : {bad}")
    # (c') perturbation du futur : bougies > T modifiées -> features <= T identiques
    T = int(Ts[0])
    df2 = df.copy()
    df2.iloc[T + 1:, :5] = df2.iloc[T + 1:, :5].to_numpy() * 1.37
    e2 = compute_extra(df2)
    same = full.iloc[: T + 1].equals(e2.iloc[: T + 1])
    print(f"(c') bougies > {df.index[T]} multipliées par 1,37 : features extra <= t identiques : {same}")

    log = lambda *a, **k: None  # noqa: E731
    union = nested.load_union(CFG17, log)
    X, y = mats(df, union)
    worst = {"inner_label_end_minus_val_start_h": -math.inf, "val_label_end_minus_test_start_h": -math.inf,
             "final_label_end_minus_test_start_h": -math.inf, "val_open_max_minus_test_start_h": -math.inf}
    for f24 in make_folds(horizon=24):
        k = f24.k
        tr24, _ = split_fold(df, f24, horizon=24)
        Xc, yc = X.loc[tr24.index], y[24].loc[tr24.index]
        cl = (~Xc.isna().any(axis=1)) & (~yc.isna())
        idx_c = tr24.index[cl.to_numpy()]
        v0 = int(math.floor(0.8 * len(idx_c)))
        vs, ve = idx_c[v0], f24.train_end
        open_v = df["open"].loc[vs: ve + H1]
        worst["val_open_max_minus_test_start_h"] = max(worst["val_open_max_minus_test_start_h"],
                                                       (open_v.index.max() - f24.test_start) / H1)
        for c in union:
            _, Xtr, ytr = nested.config_train(df, X, y, c, k, None)
            t = Xtr.index
            inner = np.asarray(t < vs) & purge_mask(t, vs, ve, horizon=24)
            vrows = np.asarray((t >= vs) & (t <= ve))
            Hc = c["H"] * H1
            le_in = (t[inner].max() + H1 + Hc - vs) / H1
            le_v = (t[vrows].max() + H1 + Hc - f24.test_start) / H1
            le_f = (t.max() + H1 + Hc - f24.test_start) / H1
            worst["inner_label_end_minus_val_start_h"] = max(worst["inner_label_end_minus_val_start_h"], le_in)
            worst["val_label_end_minus_test_start_h"] = max(worst["val_label_end_minus_test_start_h"], le_v)
            worst["final_label_end_minus_test_start_h"] = max(worst["final_label_end_minus_test_start_h"], le_f)
        print(f"(d) fold {k} : val {vs} -> {ve} ; test_start {f24.test_start} ; dernier open du backtest de "
              f"validation {open_v.index.max()}")
    print("(d) pire cas sur 9 folds x 16 configurations (heures ; doit être < 0) : "
          + json.dumps(worst))
    print(f"(d) aucune information du test ni de la validation dans le train interne : "
          f"{all(v < 0 for v in worst.values())}")


def poisoned(df: pd.DataFrame, cut: pd.Timestamp, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pois = df.copy()
    m = pois.index >= cut
    k = int(m.sum())
    fac = np.exp(np.cumsum(rng.normal(0, 0.05, k)))
    for col in ("open", "high", "low", "close"):
        pois.loc[m, col] = pois.loc[m, col].to_numpy() * fac * rng.uniform(0.9, 1.1, k)
    pois.loc[m, "volume"] = rng.permutation(pois.loc[m, "volume"].to_numpy())
    return pois


def fold(k: int, seed: int) -> None:
    df = load_dataset()
    assert df.index.max() <= MAX_INDEX
    log = harness.Tee()
    union = nested.load_union(CFG17, lambda *a, **kw: None)
    f24 = make_folds(horizon=24)[k - 1]
    pois = poisoned(df, f24.test_start, seed + k)
    Xc, yc = mats(df, union)
    Xp, yp = mats(pois, union)
    rc, mc = nested.run_fold(f24, df, Xc, yc, union, 42, None, log)
    rp, mp = nested.run_fold(f24, pois, Xp, yp, union, 42, None, log)
    same_cfg = rc["per_config_best"] == rp["per_config_best"]
    same_choice = (rc["chosen_config"], rc["best_hp"], rc["best_map"]) == (rp["chosen_config"], rp["best_hp"],
                                                                           rp["best_map"])
    tm = (df.index >= f24.test_start) & (df.index <= f24.test_end)
    feats = rc["chosen_features"]
    pc = harness.predict_proba(mc, Xc.loc[tm, feats])
    pp = harness.predict_proba(mp, Xc.loc[tm, feats])
    same_model = np.array_equal(pc, pp, equal_nan=True)
    n_changed_px = int((pois["open"] != df["open"]).sum())
    print(f"(b) fold {k} (seed poison {seed + k}) : opens modifiés {n_changed_px} (>= {f24.test_start}) ; "
          f"16 records de validation identiques {same_cfg} ; choix identique {same_choice} "
          f"({rc['chosen_config']} / {rp['chosen_config']}) ; probas du modèle final identiques {same_model}")
    print(f"(b) fold {k} RÉSULTAT : {same_cfg and same_choice and same_model}")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    seed = 7
    if "--seed" in sys.argv:
        seed = int(sys.argv[sys.argv.index("--seed") + 1])
    if sys.argv[1] == "static":
        static(seed)
    else:
        fold(int(sys.argv[2]), seed)
    return 0


if __name__ == "__main__":
    sys.exit(main())
