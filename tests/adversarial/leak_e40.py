r"""Avocat du diable P3-A5, T8 (fuite) pour E40 (b7.py, bougies 4 h, TB fixe H=30, row_filter funding).

Usage :
  .venv\Scripts\python tests\adversarial\leak_e40.py static     (a) grep + (c) labels/features/funding 4 h + (d) purge
  .venv\Scripts\python tests\adversarial\leak_e40.py fold K     (b) prix ET funding >= test_start empoisonnés, fold K
Données : load_dataset() et load_funding() par défaut (<= 2025-09-30 23:00 UTC).
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
import b7  # noqa: E402
import agg4h  # noqa: E402
import b5  # noqa: E402
import harness  # noqa: E402
from bot.funding import load_funding  # noqa: E402
from bot.split import H1, load_dataset, make_folds, purge_mask  # noqa: E402

MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")
CFG = json.loads((ROOT / "experiments" / "configs" / "E40.json").read_text(encoding="utf-8"))
H = 30
HH = agg4h.horizon_hours(H)


def load():
    df = load_dataset()
    assert df.index.max() <= MAX_INDEX, "données réservées"
    fu = load_funding()
    assert fu["fundingTime"].max() <= MAX_INDEX, "données réservées (funding)"
    print(f"[garde] index max lu = {df.index.max()} ({len(df)} lignes) ; fundingTime max {fu['fundingTime'].max()} "
          f"({len(fu)} règlements)")
    return df, fu


def grep() -> None:
    pat = re.compile(r"allow_holdout\s*=\s*True|holdout_period\(|read_parquet\(|read_csv\(|HOLDOUT_END|2025-1[0-2]|2026-")
    hits = []
    for base in ("src", "experiments"):
        for p in sorted((ROOT / base).rglob("*.py")):
            for i, ln in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
                if pat.search(ln):
                    hits.append(f"{p.relative_to(ROOT)}:{i}: {ln.strip()}")
    print(f"(a) motifs d'accès potentiels au holdout dans src/ et experiments/ (*.py) : {len(hits)}")
    for h in hits:
        print("    " + h)
    b7src = (ROOT / "experiments" / "b7.py").read_text(encoding="utf-8")
    print(f"(a) b7.py : appels load_funding {re.findall(r'load_funding\([^)]*\)', b7src)} ; "
          f"allow_holdout dans b7.py : {'allow_holdout' in b7src}")


def static() -> None:
    grep()
    df1, fund = load()
    df4 = agg4h.aggregate_4h(df1)
    F = agg4h.compute_features_4h(df4)
    spec = CFG["label"]
    y0 = agg4h.make_label_4h(df4, F, spec).to_numpy()
    rng = np.random.default_rng(43)
    pts = np.sort(rng.choice(np.arange(400, len(df4) - H - 3), 20, replace=False))
    bad_lab = bad_feat = bad_bar = 0
    for i in pts:
        cut = df4.index[i + 2 + H]
        d1 = df1.loc[: cut - H1]
        a1 = agg4h.aggregate_4h(d1)
        y1 = agg4h.make_label_4h(a1, agg4h.compute_features_4h(a1), spec).to_numpy()
        bad_lab += 0 if np.array_equal(y0[: i + 1], y1[: i + 1], equal_nan=True) else 1
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
    print(f"(c1) label TB 4 h E40 : 20 troncatures après la bougie t+1+H (seed 43) -> labels <= t modifiés : {bad_lab}/20")
    print(f"(c2) 20 perturbations des heures >= T+4h -> bougies 4 h <= t modifiées : {bad_bar}/20 ; "
          f"features <= t modifiées : {bad_feat}/20")
    # (c4) funding : règlements fundingTime > T+4h perturbés / supprimés
    G0 = b7.funding_4h(fund, df1.index, df4.index)
    rf0 = G0[b7.FU].notna().all(axis=1).to_numpy()
    rng = np.random.default_rng(44)
    i0 = int(df4.index.searchsorted(pd.Timestamp("2020-02-01", tz="UTC")))
    pts = np.sort(rng.choice(np.arange(i0 - 30, len(df4) - 10), 20, replace=False))
    ft = fund["fundingTime"]
    bad_g = bad_rf = 0
    for i in pts:
        T = df4.index[i]
        fut = (ft > T + 4 * H1).to_numpy()
        f2 = fund.copy()
        f2.loc[fut, "fundingRate"] = rng.normal(0, 0.01, int(fut.sum()))
        f2 = f2[~(fut & (rng.random(len(f2)) < 0.3))].reset_index(drop=True)
        G2 = b7.funding_4h(f2, df1.index, df4.index)
        bad_g += 0 if np.array_equal(G0.iloc[: i + 1].to_numpy(), G2.iloc[: i + 1].to_numpy(), equal_nan=True) else 1
        bad_rf += 0 if np.array_equal(rf0[: i + 1], G2[b7.FU].notna().all(axis=1).to_numpy()[: i + 1]) else 1
    print(f"(c4) funding : 20 bougies T (seed 44), règlements fundingTime > T+4h perturbés et 30 % supprimés -> "
          f"features funding <= T modifiées : {bad_g}/20 ; masque row_filter <= T modifié : {bad_rf}/20 ; première "
          f"bougie row_filter {df4.index[rf0][0]}")
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
    print(f"(c3) label réimplémenté indépendamment : égal à agg4h.make_label_4h : "
          f"{np.array_equal(y0, ymine, equal_nan=True)}")

    c = b7.prep(CFG, 0)
    log = harness.Tee()
    log.lines = []
    df4b, Fb, X, Y = b5.build_inputs(df1, [c], "base", lambda *a, **k: None)
    worst = {"inner_label_end_minus_val_start_h": -math.inf, "final_label_bar_end_minus_test_start_h": -math.inf,
             "val_open_max_minus_test_start_h": -math.inf}
    for f in make_folds(horizon=HH):
        fH, Xtr, ytr = b5.config_train(df4b, X, Y, c, f.k, None)
        t = Xtr.index
        nn = len(t)
        v0 = int(math.floor((1.0 - harness.VAL_FRAC) * nn))
        vs, ve = t[v0], fH.train_end
        inner = (np.arange(nn) < v0) & purge_mask(t, vs, ve, horizon=HH)
        le_in = (t[inner].max() + 4 * (1 + H) * H1 - vs) / H1
        le_f = (t.max() + 4 * (1 + H) * H1 + 4 * H1 - f.test_start) / H1
        ov = df1["open"].loc[vs: ve + H1]
        lo = (ov.index.max() - f.test_start) / H1
        worst["inner_label_end_minus_val_start_h"] = max(worst["inner_label_end_minus_val_start_h"], le_in)
        worst["final_label_bar_end_minus_test_start_h"] = max(worst["final_label_bar_end_minus_test_start_h"], le_f)
        worst["val_open_max_minus_test_start_h"] = max(worst["val_open_max_minus_test_start_h"], lo)
        print(f"(d) fold {f.k} : train {t[0]} -> {t.max()} ({nn} lignes) ; val {vs} -> {ve} ; test_start {f.test_start} ; "
              f"open4[t+1+H] inner - val_start {le_in:+.0f} h ; fin bougie t+1+H train - test_start {le_f:+.0f} h ; "
              f"dernier open validation - test_start {lo:+.0f} h")
    print("(d) pire cas : " + json.dumps(worst))
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
    df1, fund = load()
    f = make_folds(horizon=HH)[k - 1]
    pois = poisoned(df1, f.test_start, 41 + k)
    rng = np.random.default_rng(141 + k)
    fpois = fund.copy()
    fm = (fpois["fundingTime"] >= f.test_start).to_numpy()
    fpois.loc[fm, "fundingRate"] = rng.normal(0, 0.003, int(fm.sum()))
    c = b7.prep(CFG, 0)
    log = harness.Tee()
    res = []
    for d, fu in ((df1, None), (pois, fpois)):
        b7._FUNDING_OVERRIDE = fu
        df4, F, X, Y = b5.build_inputs(d, [c], "base", log)
        res.append((df4, X, b5.run_fold_single(f, d, df4, X, Y, c, 42, None, log, list(X.columns))))
    b7._FUNDING_OVERRIDE = None
    (df4c, Xc, (rc, mc)), (_, _, (rp, mp)) = res
    keys = ("best_hp", "best_map_rel", "best_map", "best_val_sharpe", "best_val_trades", "n_train", "auc_val",
            "auc_train", "val_start", "train_start", "sharpe_in_sample")
    diff = [kk for kk in keys if rc[kk] != rp[kk]]
    tm = (df4c.index >= f.test_start - 4 * H1) & (df4c.index <= f.test_end)
    pc = harness.predict_proba(mc, Xc.loc[tm, rc["chosen_features"]])
    pp = harness.predict_proba(mp, Xc.loc[tm, rc["chosen_features"]])
    same_model = np.array_equal(pc, pp, equal_nan=True)
    print(f"(b) fold {k} (seed poison prix {41 + k}, funding {141 + k}) : opens 1 h modifiés "
          f"{int((pois['open'] != df1['open']).sum())} ; taux de funding modifiés {int(fm.sum())} (>= {f.test_start}) ; "
          f"champs différents {diff} ; choix {rc['best_hp']} {rc['best_map']} ; train_start {rc['train_start']} n_train "
          f"{rc['n_train']} ; probas modèle final identiques sur features propres du test {same_model}")
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
