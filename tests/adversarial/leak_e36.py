"""Avocat du diable P3-A4, T8 (fuite) pour E36 (b5.py « single », bougies 4 h, TB fixe H=30).

Usage :
  .venv\\Scripts\\python tests\\adversarial\\leak_e36.py static     (a) grep + (c) labels/features 4 h + (d) purge
  .venv\\Scripts\\python tests\\adversarial\\leak_e36.py fold K     (b) empoisonnement des heures >= test_start, fold K
Données : load_dataset() par défaut (index max <= 2025-09-30 23:00 UTC).
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import math  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import agg4h  # noqa: E402
import b5  # noqa: E402
import harness  # noqa: E402
from bot.split import H1, load_dataset, make_folds, purge_mask  # noqa: E402

MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")
CFG = json.loads((ROOT / "experiments" / "configs" / "E36.json").read_text(encoding="utf-8"))
H = 30
HH = agg4h.horizon_hours(H)


def load():
    df = load_dataset()
    assert df.index.max() <= MAX_INDEX, "données réservées"
    print(f"[garde] index max lu = {df.index.max()} ({len(df)} lignes)")
    return df


def grep() -> None:
    pat = re.compile(r"allow_holdout\s*=\s*True|holdout_period\(|read_parquet\(|HOLDOUT_END|2025-1[0-2]|2026-")
    hits = []
    for base in ("src", "experiments"):
        for p in sorted((ROOT / base).rglob("*.py")):
            for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if pat.search(ln):
                    hits.append(f"{p.relative_to(ROOT)}:{i}: {ln.strip()}")
    print(f"(a) motifs d'accès potentiels au holdout dans src/ et experiments/ (*.py) : {len(hits)}")
    for h in hits:
        print("    " + h)


def static() -> None:
    grep()
    df1 = load()
    df4 = agg4h.aggregate_4h(df1)
    F = agg4h.compute_features_4h(df4)
    spec = CFG["label"]
    y0 = agg4h.make_label_4h(df4, F, spec).to_numpy()
    rng = np.random.default_rng(33)
    pts = np.sort(rng.choice(np.arange(400, len(df4) - H - 3), 20, replace=False))
    bad_lab = bad_feat = bad_bar = 0
    for i in pts:
        # (c1) heures tronquées après la fin de la bougie 4 h i+1+H
        cut = df4.index[i + 2 + H]  # première heure de la bougie i+2+H
        d1 = df1.loc[: cut - H1]
        a1 = agg4h.aggregate_4h(d1)
        y1 = agg4h.make_label_4h(a1, agg4h.compute_features_4h(a1), spec).to_numpy()
        bad_lab += 0 if np.array_equal(y0[: i + 1], y1[: i + 1], equal_nan=True) else 1
        # (c2) heures >= T+4h perturbées : bougies et features <= i identiques
        T = df4.index[i]
        d2 = df1.copy()
        m = d2.index >= T + 4 * H1
        k = int(m.sum())
        fac = np.exp(rng.normal(0, 0.05, k))
        for col in ("open", "high", "low", "close"):
            d2.loc[m, col] = d2.loc[m, col].to_numpy() * fac
        d2.loc[m, "volume"] = rng.permutation(d2.loc[m, "volume"].to_numpy())
        a2 = agg4h.aggregate_4h(d2)
        f2 = agg4h.compute_features_4h(a2)
        bad_bar += 0 if df4.iloc[: i + 1].equals(a2.iloc[: i + 1]) else 1
        bad_feat += 0 if np.array_equal(F.iloc[: i + 1].to_numpy(), f2.iloc[: i + 1].to_numpy(), equal_nan=True) else 1
    print(f"(c1) label TB 4 h E36 : 20 troncatures après la bougie t+1+H (seed 33) -> labels <= t modifiés : {bad_lab}/20")
    print(f"(c2) 20 perturbations des heures >= T+4h -> bougies 4 h <= t modifiées : {bad_bar}/20 ; "
          f"features <= t modifiées : {bad_feat}/20")
    # (c3) réimplémentation indépendante du label
    o = np.where(df4["missing"].to_numpy(), np.nan, df4["open"].to_numpy())
    v = spec["up"]["v"]
    u = max(v, agg4h.C_LOG)
    n = len(o)
    ymine = np.full(n, np.nan)
    for t in range(n - H - 1):
        e = o[t + 1]
        seg = o[t + 1: t + 2 + H]
        if np.isnan(seg).any():
            continue
        r = np.log(o[t + 2: t + 2 + H] / e)
        up = np.flatnonzero(r >= u)
        dn = np.flatnonzero(r <= -v)
        ju = up[0] if len(up) else np.inf
        jd = dn[0] if len(dn) else np.inf
        ymine[t] = 1.0 if (ju < np.inf and ju < jd) else 0.0
    ok = ~np.isnan(y0) | ~np.isnan(ymine)
    print(f"(c3) label réimplémenté indépendamment : lignes {int(ok.sum())} ; égal à agg4h.make_label_4h : "
          f"{np.array_equal(y0, ymine, equal_nan=True)}")

    c = b5.prep(CFG, 0)
    Y = {"E36": agg4h.make_label_4h(df4, F, spec)}
    worst = {"inner_label_end_minus_val_start_h": -math.inf, "final_label_bar_end_minus_test_start_h": -math.inf,
             "val_open_max_minus_test_start_h": -math.inf}
    for f in make_folds(horizon=HH):
        fH, Xtr, ytr = b5.config_train(df4, F, Y, c, f.k, None)
        t = Xtr.index
        nn = len(t)
        v0 = int(math.floor((1.0 - harness.VAL_FRAC) * nn))
        vs, ve = t[v0], fH.train_end
        inner = (np.arange(nn) < v0) & purge_mask(t, vs, ve, horizon=HH)
        le_in = (t[inner].max() + 4 * (1 + H) * H1 - vs) / H1  # open4[t+1+H] - val_start
        le_f = (t.max() + 4 * (1 + H) * H1 + 4 * H1 - f.test_start) / H1  # fin de la bougie t+1+H - test_start
        ov = df1["open"].loc[vs: ve + H1]
        lo = (ov.index.max() - f.test_start) / H1
        worst["inner_label_end_minus_val_start_h"] = max(worst["inner_label_end_minus_val_start_h"], le_in)
        worst["final_label_bar_end_minus_test_start_h"] = max(worst["final_label_bar_end_minus_test_start_h"], le_f)
        worst["val_open_max_minus_test_start_h"] = max(worst["val_open_max_minus_test_start_h"], lo)
        print(f"(d) fold {f.k} : train {t[0]} -> {t.max()} ; val {vs} -> {ve} ; test_start {f.test_start} ; "
              f"open4[t+1+H] inner - val_start {le_in:+.0f} h ; fin bougie t+1+H train - test_start {le_f:+.0f} h ; "
              f"dernier open validation - test_start {lo:+.0f} h")
    print("(d) pire cas (heures ; < 0 exigé pour inner/val, <= 0 pour la fin de bougie) : " + json.dumps(worst))
    print(f"(d) RÉSULTAT : {worst['inner_label_end_minus_val_start_h'] < 0 and worst['final_label_bar_end_minus_test_start_h'] <= 0 and worst['val_open_max_minus_test_start_h'] < 0}")


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
    df1 = load()
    f = make_folds(horizon=HH)[k - 1]
    pois = poisoned(df1, f.test_start, 31 + k)
    c = b5.prep(CFG, 0)
    log = harness.Tee()
    res = []
    for d in (df1, pois):
        df4, F, X, Y = b5.build_inputs(d, [c], "base", log)
        res.append((df4, X, b5.run_fold_single(f, d, df4, X, Y, c, 42, None, log, list(X.columns))))
    (df4c, Xc, (rc, mc)), (_, _, (rp, mp)) = res
    keys = ("best_hp", "best_map_rel", "best_map", "best_val_sharpe", "best_val_trades", "n_train", "auc_val",
            "auc_train", "val_start", "train_start", "sharpe_in_sample")
    diff = [kk for kk in keys if rc[kk] != rp[kk]]
    tm = (df4c.index >= f.test_start - 4 * H1) & (df4c.index <= f.test_end)
    pc = harness.predict_proba(mc, Xc.loc[tm, rc["chosen_features"]])
    pp = harness.predict_proba(mp, Xc.loc[tm, rc["chosen_features"]])
    same_model = np.array_equal(pc, pp, equal_nan=True)
    print(f"(b) fold {k} (seed poison {31 + k}) : opens 1 h modifiés {int((pois['open'] != df1['open']).sum())} "
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
