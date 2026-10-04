r"""Corrélations entre features 4 h, funding inclus (brief P3-B7, point 4 ; n'est pas un essai).

Usage : .venv\Scripts\python experiments\corr_b7.py
Pour chaque fold k de make_folds(horizon=123) : lignes du TRAIN du fold sur la grille 4 h (split_fold, aucune ligne de
test), features f4 (14) + funding (6 : fr_cur, fr_mean3, fr_mean21, fr_z90, fr_sum21, fr_age_h ; fr_sum21 et fr_age_h
exclus du modèle a priori), lignes sans NaN ; Pearson et Spearman ; paires |corr| > 0,9.
Sorties : experiments/results/corr_b7.txt et corr_b7.json.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agg4h  # noqa: E402
import b7  # noqa: E402
import harness  # noqa: E402
from bot.split import make_folds, split_fold  # noqa: E402

THR = 0.9


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    log = harness.Tee()
    df1 = harness.load_dev(log)
    fund = b7.load_fund(log)
    df4 = agg4h.aggregate_4h(df1)
    feats = list(agg4h.F4) + list(b7.FUND_ALL)
    X = agg4h.compute_features_4h(df4)
    G = b7.funding_4h(fund, df1.index, df4.index)
    for c in b7.FUND_ALL:
        X[c] = G[c].to_numpy()
    X = X[feats]
    out = {"threshold": THR, "features": feats, "folds": []}
    log(f"features ({len(feats)}) : {feats}")
    hh = agg4h.horizon_hours(30)
    for f in make_folds(horizon=hh):
        tr, _ = split_fold(df4, f, horizon=hh)
        Xt = X.loc[tr.index].dropna()
        assert Xt.index.max() < f.test_start
        rec = {"fold": f.k, "train_start": Xt.index.min(), "train_end": Xt.index.max(), "n_rows": len(Xt)}
        log(f"\nfold {f.k} : train {Xt.index.min()} -> {Xt.index.max()} ({len(Xt)} bougies 4 h sans NaN ; test "
            f"{f.test_start} exclu)")
        for meth in ("pearson", "spearman"):
            C = Xt.corr(method=meth)
            pairs = []
            for i in range(len(feats)):
                for j in range(i + 1, len(feats)):
                    v = float(C.iloc[i, j])
                    if abs(v) > THR:
                        pairs.append([feats[i], feats[j], v])
            rec[meth] = {"matrix": C.round(6).to_numpy().tolist(), "pairs_gt_thr": pairs}
            log(f"  {meth:<8} : {len(pairs)} paire(s) |corr| > {THR}" + ("" if not pairs else " : " + " ; ".join(
                f"{a} / {b} = {v:+.4f}" for a, b, v in pairs)))
            fi = [k for k, n in enumerate(feats) if n.startswith("fr_")]
            fo = [k for k, n in enumerate(feats) if not n.startswith("fr_")]
            M = np.abs(C.to_numpy())
            sub = M[np.ix_(fi, fo)]
            a, b = np.unravel_index(np.argmax(sub), sub.shape)
            log(f"             max |corr| funding / f4 : {feats[fi[a]]} / {feats[fo[b]]} = {C.iloc[fi[a], fo[b]]:+.4f}")
        out["folds"].append(rec)
    log("\n=== Synthèse : paires |corr| > 0,9 (nombre de folds sur 9) ===")
    for meth in ("pearson", "spearman"):
        cnt = {}
        for r in out["folds"]:
            for a, b, v in r[meth]["pairs_gt_thr"]:
                cnt.setdefault((a, b), []).append(v)
        for (a, b), vs in sorted(cnt.items(), key=lambda kv: -len(kv[1])):
            log(f"  {meth:<8} {a} / {b} : {len(vs)}/9 folds, min {min(vs):+.4f} max {max(vs):+.4f}")
        if not cnt:
            log(f"  {meth:<8} aucune paire")
    res = harness.RESULTS
    with open(res / "corr_b7.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(harness.jsonable(out), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    with open(res / "corr_b7.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log.lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
