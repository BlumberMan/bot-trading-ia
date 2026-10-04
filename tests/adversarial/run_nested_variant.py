"""Avocat du diable P3-A2 : relance COMPLÈTE de la procédure emboîtée E17 (experiments/nested.py importé,
non modifié) avec une perturbation. Aucune écriture dans experiments/ : sorties dans reports/adversarial/.

Usage : .venv\\Scripts\\python tests\\adversarial\\run_nested_variant.py VARIANTE
VARIANTE : base | shufS | delay2 | featlag1 | cost2 | cost3 | drop_<feat>  (nested.apply_variant)
Données : load_dataset() par défaut via harness.load_dev (assertion index max <= 2025-09-30 23:00 UTC).
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["OMP_NUM_THREADS"] = "1"
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import harness  # noqa: E402
import nested  # noqa: E402

OUT = ROOT / "reports" / "adversarial"
MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    var = sys.argv[1]
    cfg17 = json.loads((ROOT / "experiments" / "configs" / "E17.json").read_text(encoding="utf-8"))
    log = harness.Tee()
    log(f"commit {harness.git('rev-parse', 'HEAD')}")
    union = nested.load_union(cfg17, log)
    shuffle_seed = int(var[4:]) if var.startswith("shuf") else None
    nested.apply_variant(var, union)
    log(f"[adv] variante {var} : exec_delay={harness.EXEC_DELAY} cost_mult={harness.COST_MULTIPLIER}")
    out = nested.run_nested(cfg17, union, shuffle_seed, log)
    sig = out.pop("_signal")
    assert out["index_max_read"] <= MAX_INDEX
    m = out["concatenated"]["model"]
    log(f"[adv] RÉSULTAT {var} : Sharpe concat {m['sharpe']:.6f} trades {m['trades']} expo {m['exposure']:.4f} "
        f"DD {m['max_drawdown']:.4f} rdt {m['total_return']:.4f} folds>0 {out['folds_sharpe_positive']}/9 ; "
        f"choix {';'.join(r['chosen_config'] for r in out['folds'])}")
    OUT.mkdir(parents=True, exist_ok=True)
    name = f"E17_adv_{var}"
    with open(OUT / f"{name}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(harness.jsonable(out), fh, indent=1, ensure_ascii=False)
    (OUT / f"{name}.txt").write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    if var == "base":
        sig.to_frame("signal").to_csv(OUT / "E17_adv_base_signal_full.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
