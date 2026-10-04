r"""Corrélations entre features 4 h (brief P3-B5, point 3 ; n'est pas un essai).

Usage : .venv\Scripts\python experiments\corr_b5.py
Pour chaque fold k de make_folds(horizon=123) (H=30 bougies 4 h, la purge la plus large du brief) : lignes du
TRAIN du fold sur la grille 4 h (split_fold, aucune ligne de test), features f4 (14) + calendrier (4, hors f4),
lignes sans NaN ; matrices de Pearson et de Spearman (pandas) ; paires |corr| > 0,9 (fold, méthode, paire, valeur).
Sorties : experiments/results/corr_b5.txt (brut) et corr_b5.json (matrices complètes).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agg4h  # noqa: E402
import harness  # noqa: E402
from bot.split import make_folds, split_fold  # noqa: E402

THR = 0.9


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    log = harness.Tee()
    df1 = harness.load_dev(log)
    df4 = agg4h.aggregate_4h(df1)
    feats = list(agg4h.ALL4)
    X = agg4h.compute_features_4h(df4)[feats]
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
            mx = np.abs(C.to_numpy() - np.eye(len(feats)))
            i, j = np.unravel_index(np.argmax(mx), mx.shape)
            log(f"             max |corr| hors diagonale : {feats[i]} / {feats[j]} = {C.iloc[i, j]:+.4f}")
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
    with open(res / "corr_b5.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(harness.jsonable(out), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    with open(res / "corr_b5.txt", "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log.lines) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
