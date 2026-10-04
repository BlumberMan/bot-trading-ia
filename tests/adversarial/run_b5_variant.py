"""Avocat du diable P3-A4 : relance COMPLÈTE de la procédure b5 « single » de E36 (experiments/b5.py importé,
non modifié) avec une perturbation. Sorties : reports/adversarial/E36_adv_<variante>.json/.txt (+ parquet pour base).

Usage : .venv\\Scripts\\python tests\\adversarial\\run_b5_variant.py VARIANTE
VARIANTE :
  base | shufS | delay2 | featlag1 | cost2 | cost3 | drop_<f> | rule_<f> | tbdir      (b5.apply_variant)
  tbv08 | tbv12      barrières fixes x0,8 / x1,2 (v = 0,049521 / 0,074281), même procédure
  onlyvols           lgbm, features [vol_42, vol_180]
Données : harness.load_dev() (load_dataset() par défaut) + garde index max <= 2025-09-30 23:00 UTC.
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
import b5  # noqa: E402
import harness  # noqa: E402
from bot.split import load_dataset  # noqa: E402

OUT = ROOT / "reports" / "adversarial"
MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")
CFG = json.loads((ROOT / "experiments" / "configs" / "E36.json").read_text(encoding="utf-8"))
V0 = 0.061901


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    var = sys.argv[1]
    log = harness.Tee()
    log(f"commit {harness.git('rev-parse', 'HEAD')} ; [adv] variante {var}")
    df1 = load_dataset()
    assert df1.index.max() <= MAX_INDEX, "données réservées"
    log(f"[garde adv] index max = {df1.index.max()} <= {MAX_INDEX}")
    cfg = json.loads(json.dumps(CFG))
    configs = [b5.prep(cfg, 0)]
    c = configs[0]
    shuffle_seed = int(var[4:]) if var.startswith("shuf") else None
    if var in ("tbv08", "tbv12"):
        v = round(V0 * (0.8 if var == "tbv08" else 1.2), 6)
        c["label"]["up"] = {"type": "fixed_log", "v": v}
        c["label"]["down"] = {"type": "fixed_log", "v": v}
    elif var == "onlyvols":
        c["features"] = ["vol_42", "vol_180"]
    else:
        b5.apply_variant(var, configs)
    log(f"[adv] features {c['features']} ; modèle {c['model']} ; hps {len(c['hps'])} ; maps {len(c['maps'])} ; "
        f"label {json.dumps(c['label'])} ; exec_delay={b5.EXEC_DELAY_H} h ; cost_mult={b5.COST_MULTIPLIER}")
    out = b5.run(cfg, configs, False, shuffle_seed, log, var=var, df1=df1)
    sig = out.pop("_signal")
    proba = out.pop("_proba")
    assert pd.Timestamp(out["index_max_read"]) <= MAX_INDEX
    m = out["concatenated"]["model"]
    log(f"[adv] RÉSULTAT {var} : Sharpe concat {m['sharpe']:.6f} trades {m['trades']} expo {m['exposure']:.4f} "
        f"DD {m['max_drawdown']:.4f} rdt {m['total_return']:.4f} folds>0 {out['folds_sharpe_positive']}/9 ; "
        "Sharpe/fold " + ";".join("%.3f" % r["model"]["sharpe"] for r in out["folds"]))
    name = f"E36_adv_{var}"
    with open(OUT / f"{name}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(harness.jsonable({"commit": harness.git("rev-parse", "HEAD"), "variant": var, **out}), fh,
                  indent=1, ensure_ascii=False)
    (OUT / f"{name}.txt").write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    if var in ("base", "delay2", "cost2", "cost3"):
        sig.to_frame("signal").to_parquet(OUT / f"{name}_signal_1h.parquet", engine="pyarrow")
        proba.to_parquet(OUT / f"{name}_proba_4h.parquet", engine="pyarrow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
