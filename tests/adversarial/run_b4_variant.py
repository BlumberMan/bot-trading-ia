"""Avocat du diable P3-A3 : relance COMPLÈTE de la procédure b4 « single » de E24 (experiments/b4.py importé,
non modifié) avec une perturbation. Sorties : reports/adversarial/E24_adv_<variante>.json/.txt uniquement.

Usage : .venv\\Scripts\\python tests\\adversarial\\run_b4_variant.py VARIANTE
VARIANTE :
  base | shufS | delay2 | featlag1 | cost2 | cost3 | drop_<feat>   (b4.apply_variant)
  only_vol168      features = [vol_168]
  rule_vol         pas d'apprentissage : score = -vol_168 ou +vol_168 (2 "hp"), mêmes 36 mappings
  tbdir            label TB E24 avec expiration -> NaN (direction seule)
  tbfixed          label TB à seuils fixes r = médiane(vol_168 x sqrt(24)) sur les bougies < 2021-01-01
  nodow            sans dow_sin ni dow_cos
  dowplacS         dow_sin / dow_cos remplacés par un jour aléatoire par jour civil (seed S)
Données : load_dataset() par défaut via harness.load_dev (assertion index max <= 2025-09-30 23:00 UTC).
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import b4  # noqa: E402
import harness  # noqa: E402
import labels_b4  # noqa: E402
from bot.features import compute_features  # noqa: E402
from bot.split import load_dataset  # noqa: E402

OUT = ROOT / "reports" / "adversarial"
MAX_INDEX = pd.Timestamp("2025-09-30 23:00", tz="UTC")
CFG = json.loads((ROOT / "experiments" / "configs" / "E24.json").read_text(encoding="utf-8"))


def tb_paths(df: pd.DataFrame, H: int, u: np.ndarray, d: np.ndarray):
    """Réimplémentation indépendante : j_up, j_dn (inf si non touchée) sur les ouvertures."""
    o = df["open"].to_numpy(dtype=np.float64)
    miss = np.isnan(df[["open", "high", "low", "close", "volume"]].to_numpy(dtype=np.float64)).any(axis=1)
    if "missing" in df.columns:
        miss |= df["missing"].to_numpy(dtype=bool)
    o = np.where(miss, np.nan, o)
    n = len(o)
    entry = np.full(n, np.nan)
    entry[: n - 1] = o[1:]
    j_up, j_dn = np.full(n, np.inf), np.full(n, np.inf)
    with np.errstate(invalid="ignore", divide="ignore"):
        for j in range(1, H + 1):
            px = np.full(n, np.nan)
            px[: n - 1 - j] = o[1 + j:]
            r = np.log(px / entry)
            j_up[(r >= u) & np.isinf(j_up)] = j
            j_dn[(r <= -d) & np.isinf(j_dn)] = j
    return j_up, j_dn


def e24_barriers(df):
    s = compute_features(df)["vol_168"].to_numpy()
    H = 24
    return np.maximum(s * math.sqrt(H), labels_b4.C_LOG), s * math.sqrt(H)


class _Booster:
    def feature_importance(self, importance_type="gain"):
        return np.ones(1)


class SignModel:
    """Règle sans apprentissage : proba = sigmoïde(sign x z), z = vol_168 standardisée sur le fit."""

    def __init__(self, sign):
        self.sign = sign
        self.booster_ = _Booster()

    def fit(self, X, y):
        self.mu, self.sd = float(np.mean(X[:, 0])), float(np.std(X[:, 0]))
        return self

    def predict_proba(self, X):
        z = self.sign * (X[:, 0] - self.mu) / self.sd
        p = 1.0 / (1.0 + np.exp(-z))
        return np.column_stack([1 - p, p])


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    var = sys.argv[1]
    log = harness.Tee()
    log(f"commit {harness.git('rev-parse', 'HEAD')} ; [adv] variante {var}")
    df = load_dataset()
    assert df.index.max() <= MAX_INDEX, "données réservées"
    log(f"[garde adv] index max = {df.index.max()} <= {MAX_INDEX}")
    cfg = json.loads(json.dumps(CFG))
    configs = [b4.prep(cfg, 0)]
    c = configs[0]
    shuffle_seed = int(var[4:]) if var.startswith("shuf") else None
    if var in ("base", "delay2", "featlag1", "cost2", "cost3") or var.startswith("shuf") or var.startswith("drop_"):
        b4.apply_variant(var, configs)
    elif var == "only_vol168":
        c["features"] = ["vol_168"]
    elif var == "rule_vol":
        c["features"] = ["vol_168"]
        c["hps"] = [{"sign": -1.0}, {"sign": 1.0}]
        harness.make_model = lambda kind, hp, seed: SignModel(hp["sign"])
    elif var == "nodow":
        c["features"] = [f for f in c["features"] if f not in ("dow_sin", "dow_cos")]
    elif var.startswith("dowplac"):
        ps = int(var[7:])
        orig = harness.build_matrix

        def plac(d, feats):
            X = orig(d, feats).copy()
            days = d.index.normalize()
            ud = days.unique()
            rd = pd.Series(np.random.default_rng(ps).integers(0, 7, len(ud)).astype(float), index=ud)
            dw = rd.reindex(days).to_numpy()
            if "dow_sin" in X.columns:
                X["dow_sin"] = np.sin(2 * np.pi * dw / 7)
                X["dow_cos"] = np.cos(2 * np.pi * dw / 7)
            return X
        harness.build_matrix = plac
    elif var in ("tbdir", "tbfixed"):
        orig_ml = labels_b4.make_label
        if var == "tbdir":
            def ml(d, spec):
                y = orig_ml(d, spec)
                u, dd = e24_barriers(d)
                ju, jd = tb_paths(d, 24, u, dd)
                yy = y.to_numpy().copy()
                mine = ((ju < np.inf) & (ju < jd)).astype(float)
                ok = ~np.isnan(yy)
                assert np.array_equal(mine[ok], yy[ok]), "réimplémentation TB != labels_b4"
                expired = np.isinf(ju) & np.isinf(jd)
                log(f"[tbdir] lignes label défini {int(ok.sum())}, expirées -> NaN {int((expired & ok).sum())}, "
                    f"taux y=1 restant {np.nanmean(np.where(expired, np.nan, yy)):.4f}")
                yy[expired] = np.nan
                return pd.Series(yy, index=d.index, name="label")
        else:
            s = compute_features(df)["vol_168"]
            rlog = float(np.nanmedian((s.loc[: pd.Timestamp("2020-12-31 23:00", tz="UTC")] * math.sqrt(24)).to_numpy()))
            spec_fixed = {"kind": "tb", "H": 24, "up": {"type": "fixed", "r": math.exp(rlog) - 1.0},
                          "down": {"type": "fixed", "r": 1.0 - math.exp(-rlog)}}
            log(f"[tbfixed] r_log = médiane(vol_168 x sqrt(24)) < 2021 = {rlog:.6f} ; spec {spec_fixed}")
            c["label"] = spec_fixed

            def ml(d, spec):
                return orig_ml(d, spec)
        labels_b4.make_label = ml
    else:
        raise SystemExit(f"variante inconnue {var}")
    log(f"[adv] features {c['features']} ; hps {len(c['hps'])} ; maps {len(c['maps'])} ; label {c['label']} ; "
        f"exec_delay={harness.EXEC_DELAY} cost_mult={harness.COST_MULTIPLIER}")

    captured = []
    orig_fold = b4.run_fold_single

    def cap(f, *a, **k):
        r, model = orig_fold(f, *a, **k)
        captured.append((f, r, model))
        return r, model
    b4.run_fold_single = cap
    out = b4.run(cfg, configs, False, shuffle_seed, log, df=df)
    sig = out.pop("_signal")
    assert out["index_max_read"] <= MAX_INDEX
    m = out["concatenated"]["model"]
    log(f"[adv] RÉSULTAT {var} : Sharpe concat {m['sharpe']:.6f} trades {m['trades']} expo {m['exposure']:.4f} "
        f"DD {m['max_drawdown']:.4f} rdt {m['total_return']:.4f} folds>0 {out['folds_sharpe_positive']}/9 ; "
        "Sharpe/fold " + ";".join("%.3f" % r["model"]["sharpe"] for r in out["folds"]))
    name = f"E24_adv_{var}"
    with open(OUT / f"{name}.json", "w", encoding="utf-8", newline="\n") as fh:
        json.dump(harness.jsonable(out), fh, indent=1, ensure_ascii=False)
    (OUT / f"{name}.txt").write_text("\n".join(log.lines) + "\n", encoding="utf-8")
    if var == "base":
        sig.to_frame("signal").to_csv(OUT / "E24_adv_base_signal_full.csv")
        X = harness.build_matrix(df, b4.ALL_FEATS)
        pt = pd.Series(np.nan, index=df.index)
        for f, r, model in captured:
            tm = (df.index >= f.test_start) & (df.index <= f.test_end)
            pt[tm] = harness.predict_proba(model, X.loc[tm, r["chosen_features"]])
        pt.to_frame("p").to_csv(OUT / "E24_adv_base_proba_test.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
