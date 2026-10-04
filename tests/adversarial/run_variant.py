"""Avocat du diable P3-A1 : relance de la procédure COMPLÈTE de E10 (harnais importé, non modifié)
avec une perturbation, et sauvegarde du signal / de l'equity concaténés.

Usage : .venv\\Scripts\\python tests\\adversarial\\run_variant.py VARIANTE
VARIANTE :
  base           reproduction de E10 (aucune perturbation)
  shufS          labels du train mélangés, seed S (procédure complète)
  delay2         exec_delay = 2 (tuning interne ET évaluation)
  featlag1       toutes les features retardées d'une bougie supplémentaire (X.shift(1))
  cost2 / cost3  cost_multiplier = 2 / 3 (tuning interne ET évaluation)
  drop_<feat>    E10 sans la feature <feat>
Aucune écriture dans experiments/ (ni registre, ni results/) : sorties dans reports/adversarial/.
Données : load_dataset() par défaut via harness.load_dev (assertion index max <= 2025-09-30 23:00).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "experiments"))
import harness  # noqa: E402

OUT = ROOT / "reports" / "adversarial"
MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    var = sys.argv[1]
    cfg = json.loads((ROOT / "experiments" / "configs" / "E10.json").read_text(encoding="utf-8"))
    shuffle_seed = None
    if var == "base":
        pass
    elif var.startswith("shuf"):
        shuffle_seed = int(var[4:])
    elif var == "delay2":
        harness.EXEC_DELAY = 2
    elif var == "featlag1":
        orig = harness.build_matrix

        def lagged(df, feats):
            return orig(df, feats).shift(1)
        harness.build_matrix = lagged
    elif var in ("cost2", "cost3"):
        harness.COST_MULTIPLIER = float(var[4:])
    elif var.startswith("drop_"):
        feat = var[5:]
        feats = harness.resolve_features(cfg["features"])
        assert feat in feats
        cfg["features"] = [f for f in feats if f != feat]
    else:
        raise SystemExit(f"variante inconnue {var}")
    cfg["id"] = f"E10_adv_{var}"
    # dérivée de la config écrite dans tests/adversarial/ (traçabilité)
    (ROOT / "tests" / "adversarial" / "configs").mkdir(parents=True, exist_ok=True)
    (ROOT / "tests" / "adversarial" / "configs" / f"{cfg['id']}.json").write_text(
        json.dumps({**cfg, "_perturbation": var, "_exec_delay": harness.EXEC_DELAY,
                    "_cost_multiplier": harness.COST_MULTIPLIER}, ensure_ascii=False) + "\n",
        encoding="utf-8")

    captured = {}
    orig_bt = harness.run_backtest

    def spy(signal, open_, **kw):
        res = orig_bt(signal, open_, **kw)
        if kw.get("start") == harness.CONCAT_START and kw.get("end") == harness.CONCAT_END:
            captured["signal"] = signal.copy()
            captured["res"] = res
            captured["open_max"] = open_.index.max()
        return res
    harness.run_backtest = spy

    log = harness.Tee()
    out = harness.run_trial(cfg, shuffle_seed, log)
    assert captured["open_max"] <= MAX_INDEX
    log(f"[adv garde] index max des prix passés au backtest concaténé = {captured['open_max']} <= {MAX_INDEX}")
    m = out["concatenated"]["model"]
    log(f"[adv] variante {var} : exec_delay={harness.EXEC_DELAY} cost_mult={harness.COST_MULTIPLIER} "
        f"Sharpe concat {m['sharpe']:.4f} trades {m['trades']} expo {m['exposure']:.4f} "
        f"DD {m['max_drawdown']:.4f} folds>0 {out['folds_sharpe_positive']}/9")
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{cfg['id']}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(harness.jsonable(out), fh, indent=1, ensure_ascii=False)
    (OUT / f"{cfg['id']}.txt").write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    res = captured["res"]
    pd.DataFrame({"signal": captured["signal"].reindex(res.equity.index),
                  "position": res.position, "equity": res.equity}).to_csv(OUT / f"{cfg['id']}_series.csv")
    if var == "base":
        captured["signal"].to_frame("signal").to_csv(OUT / "E10_adv_base_signal_full.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
