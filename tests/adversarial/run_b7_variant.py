"""Avocat du diable P3-A5 : relance COMPLÈTE de la procédure de E40 (experiments/b7.py importé, non modifié) avec
une perturbation. Sorties : reports/adversarial/E40_adv_<variante>.json/.txt (+ parquet pour base et quelques variantes).

Usage : .venv\\Scripts\\python tests\\adversarial\\run_b7_variant.py VARIANTE
VARIANTE :
  base | shufS | delay2 | featlag1 | cost2 | cost3 | drop_<f> | rule_<f> | tbdir | tbscale0.8 | tbscale1.2   (b7)
  onlyvols                lgbm, features [vol_42, vol_180]
  startYYYY-MM-DD         row_filter retiré, train restreint aux bougies 4 h >= date (tous les folds)
  seedN                   E40, seed LightGBM N
  e36seedN                configuration E36 (b5, sans row_filter), seed N, via le même chemin b7
Données : load_dataset() et load_funding() par défaut ; gardes prix et funding <= 2025-09-30 23:00 UTC.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import pandas as pd  # noqa: E402

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import b7  # noqa: E402  (remplace b5.build_inputs / b5.config_train)
import b5  # noqa: E402
import harness  # noqa: E402
from bot.funding import load_funding  # noqa: E402
from bot.split import load_dataset  # noqa: E402

OUT = ROOT / "reports" / "adversarial"
MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")
CFG40 = json.loads((ROOT / "experiments" / "configs" / "E40.json").read_text(encoding="utf-8"))
CFG36 = json.loads((ROOT / "experiments" / "configs" / "E36.json").read_text(encoding="utf-8"))
SAVE = ("base", "delay2", "cost2", "cost3", "start2019-01-01", "e36seed42")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    var = sys.argv[1]
    log = harness.Tee()
    log(f"commit {harness.git('rev-parse', 'HEAD')} ; [adv P3-A5] variante {var}")
    df1 = load_dataset()
    assert df1.index.max() <= MAX_INDEX, "données réservées (prix)"
    fu = load_funding()
    assert fu["fundingTime"].max() <= MAX_INDEX, "données réservées (funding)"
    log(f"[garde adv] prix index max = {df1.index.max()} ; funding fundingTime max = {fu['fundingTime'].max()} "
        f"<= {MAX_INDEX}")
    cfg = json.loads(json.dumps(CFG36 if var.startswith("e36seed") else CFG40))
    if var.startswith("seed") or var.startswith("e36seed"):
        cfg["seed"] = int(var.split("seed")[1])
    if var.startswith("e36seed"):
        configs = [b7.prep(cfg, 0)]
        assert configs[0]["row_filter"] == []
    else:
        configs = [b7.prep(cfg, 0)]
    c = configs[0]
    shuffle_seed = int(var[4:]) if var.startswith("shuf") else None
    bvar = "base"
    if var == "onlyvols":
        c["features"] = ["vol_42", "vol_180"]
    elif var.startswith("start"):
        start = pd.Timestamp(var[5:], tz="UTC")
        c["row_filter"] = []
        orig = b5.config_train

        def ct(df4, X, Y, cc, k, sh):
            fH, Xtr, ytr = orig(df4, X, Y, cc, k, sh)
            m = (Xtr.index >= start)
            return fH, Xtr.loc[m], ytr[m]
        b5.config_train = ct
    elif var.startswith("seed") or var.startswith("e36seed"):
        pass
    else:
        b7.apply_variant(var, configs)
        bvar = var
    log(f"[adv] id {cfg['id']} seed {cfg['seed']} ; features {c['features']} ; row_filter {c['row_filter']} ; modèle "
        f"{c['model']} ; hps {len(c['hps'])} ; maps {len(c['maps'])} ; label {json.dumps(c['label'])} ; "
        f"exec_delay={b5.EXEC_DELAY_H} h ; cost_mult={b5.COST_MULTIPLIER}")
    out = b5.run(cfg, configs, False, shuffle_seed, log, var=bvar, df1=df1)
    sig = out.pop("_signal")
    proba = out.pop("_proba")
    assert pd.Timestamp(out["index_max_read"]) <= MAX_INDEX
    m = out["concatenated"]["model"]
    log(f"[adv] RÉSULTAT {var} : Sharpe concat {m['sharpe']:.6f} trades {m['trades']} expo {m['exposure']:.4f} "
        f"DD {m['max_drawdown']:.4f} rdt {m['total_return']:.4f} folds>0 {out['folds_sharpe_positive']}/9 ; "
        "Sharpe/fold " + ";".join("%.3f" % float(r["model"]["sharpe"]) for r in out["folds"])
        + " ; trades/fold " + ";".join(str(r["model"]["trades"]) for r in out["folds"])
        + " ; train_start/fold " + ";".join(str(r["train_start"])[:10] for r in out["folds"]))
    name = f"E40_adv_{var.replace(':', '')}"
    with open(OUT / f"{name}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(harness.jsonable({"commit": harness.git("rev-parse", "HEAD"), "variant": var, **out}), fh,
                  indent=1, ensure_ascii=False)
    (OUT / f"{name}.txt").write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    if var in SAVE:
        sig.to_frame("signal").to_parquet(OUT / f"{name}_signal_1h.parquet", engine="pyarrow")
        proba.to_parquet(OUT / f"{name}_proba_4h.parquet", engine="pyarrow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
