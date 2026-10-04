"""Brief P3-B7 : apport du funding (bougies 4 h), essais E40-E47 ; réutilise experiments/b5.py par import.

Usage :
    .venv\\Scripts\\python experiments\\b7.py experiments\\configs\\E40.json                  (essai -> results/E40.json)
    .venv\\Scripts\\python experiments\\b7.py experiments\\configs\\E40.json --variant shuf3   (labels mélangés, hors registre)
    .venv\\Scripts\\python experiments\\b7.py CONFIG --variant <variante b5>|tbscale0.8|tbscale1.2|fplac<S>
                                                                                             (contrôles, hors registre)
    .venv\\Scripts\\python experiments\\b7.py CONFIG --out PATH                               (relance de reproductibilité)
    .venv\\Scripts\\python experiments\\b7.py --register E40                                   (ajoute LA ligne de E40)
    .venv\\Scripts\\python experiments\\b7.py --selftest                                       (données SYNTHÉTIQUES)

DONNÉES
- Prix : harness.load_dev() = bot.split.load_dataset() par défaut + garde affichée « index max <= 2025-09-30 23:00 UTC ».
- Funding : bot.funding.load_funding() par défaut (règlements fundingTime <= DEV_END) + bot.funding.assert_no_holdout +
  garde affichée « fundingTime max <= 2025-09-30 23:00 UTC ». bot.funding n'est pas modifié.

FUNDING SUR LA GRILLE 4 h (funding_4h)
- Valeur à la bougie 4 h T (open_time T, clôture T+4h) = valeur de bot.funding.funding_features sur la grille 1 h à la
  4e heure de la bougie, T+3h. La règle 1 h de bot.funding (fundingTime <= t + 1 h) donne donc fundingTime <= T + 4 h :
  dernier règlement publié à la clôture de la bougie 4 h. Règlements à 00/08/16 h UTC avec une gigue de quelques ms : un
  règlement horodaté 08:00:00.005 n'est pas <= 08:00 et n'est utilisé qu'à la clôture suivante (règle stricte, JOURNAL).
- Features de funding au modèle (jeu « fu », déclaré a priori, avant tout résultat) : fr_cur, fr_mean3, fr_mean21, fr_z90.
  Exclues a priori : fr_sum21 (= 21 x fr_mean21, corrélation 1 par construction) et fr_age_h (fonction de l'heure du
  jour : une feature calendaire, exclue comme dans f4 depuis P3-A3). Les 6 sont calculées et figurent dans les
  corrélations.
- Lignes : toutes les configurations de ce brief (row_filter = fu) n'utilisent que les lignes où les 4 features fu sont
  définies (fr_z90 : 90 règlements, soit à partir de ~2020-01-31) ; « f4 seul » et « f4 + fu » d'une même paire ont donc
  exactement les mêmes lignes de train, de validation interne et la même période. La règle s'applique au train propre
  du fold (features du modèle + row_filter + label non NaN). Procédure emboîtée : lignes propres de la validation
  commune = f4, fu et fwd_ret H=30 non NaN (le fwd_ret commun est mis à NaN là où fu n'est pas défini).
- Placebo fplac<S> : taux de funding permutés par blocs de 30 jours (blocs = floor((fundingTime - premier) / 30 j)),
  ordre des blocs tiré par numpy default_rng([S, 30]), taux réaffectés dans l'ordre aux horodatages d'origine ; les
  features de funding sont recalculées sur ces taux. Contrôle, hors registre.
- Sensibilité des barrières tbscale<f> : barrières des labels triple barrière multipliées par f (vol : k x f ;
  fixed_log : v x f, arrondi 1e-6). Contrôle, hors registre.

PROCÉDURE : celle de b5.py (docstring), inchangée ; b7 remplace seulement b5.build_inputs (ajout des colonnes de
funding à X, fwd_ret commun masqué) et b5.config_train (row_filter), par affectation dans le module b5.
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ[_v] = "1"

import argparse  # noqa: E402
import copy  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parent))
import agg4h  # noqa: E402
import b5  # noqa: E402
import harness  # noqa: E402
from b4 import git_blob_sha1  # noqa: E402
from bot import funding as bf  # noqa: E402
from bot.split import H1, make_folds, split_fold  # noqa: E402

ROOT = harness.ROOT
RESULTS = harness.RESULTS
REGISTRE = harness.REGISTRE
CONFIGS = ROOT / "experiments" / "configs"
CODE_PATHS = b5.CODE_PATHS + ("experiments/b7.py",)
REG_MAX = 49  # budget P3-B7 : 10 essais au maximum (39 + 10)
FU: list[str] = ["fr_cur", "fr_mean3", "fr_mean21", "fr_z90"]
FUND_ALL: list[str] = list(bf.FUNDING_FEATURES) + ["fr_age_h"]
BRIEF_DECL = ("label repris de P3-B5 et choisi au vu de son OOS (déclaré ; DSR à N cumulé)")

_FUNDING_OVERRIDE: pd.DataFrame | None = None  # auto-test synthétique uniquement
_orig_build_inputs = b5.build_inputs


# ------------------------------------------------------------------ funding 4 h

def funding_4h(fund: pd.DataFrame, idx1: pd.DatetimeIndex, idx4: pd.DatetimeIndex) -> pd.DataFrame:
    """Lot : features de funding à la bougie 4 h T = bot.funding.funding_features (grille 1 h) à T+3h."""
    g = bf.funding_features(fund, idx1)
    pos = idx1.get_indexer(idx4 + 3 * H1)
    assert (pos >= 0).all(), "4e heure d'une bougie 4 h absente de la grille 1 h"
    out = g.iloc[pos].copy()
    out.index = idx4
    return out[FUND_ALL]


def funding_4h_live(recent: pd.DataFrame, T: pd.Timestamp) -> pd.Series:
    """Direct : features de funding à la clôture de la bougie 4 h T (règlements fundingTime <= T + 4 h)."""
    return bf.funding_features_live(recent, pd.Timestamp(T) + 3 * H1)[FUND_ALL]


def placebo_blocks(fund: pd.DataFrame, seed: int, days: int = 30) -> pd.DataFrame:
    ft = fund["fundingTime"]
    blk = ((ft - ft.iloc[0]) // pd.Timedelta(days=days)).to_numpy().astype(np.int64)
    ub = np.unique(blk)
    order = np.random.default_rng([int(seed), days]).permutation(ub)
    r = fund["fundingRate"].to_numpy()
    out = fund.copy()
    out["fundingRate"] = np.concatenate([r[blk == b] for b in order])
    return out


def load_fund(log) -> pd.DataFrame:
    if _FUNDING_OVERRIDE is not None:
        log("[funding] SYNTHÉTIQUE (auto-test)")
        return _FUNDING_OVERRIDE
    fund = bf.load_funding()
    bf.assert_no_holdout(fund)
    mx = fund["fundingTime"].max()
    assert mx <= harness.MAX_INDEX, f"fundingTime max {mx} > {harness.MAX_INDEX} : période réservée interdite"
    log(f"[garde funding] fundingTime max lu = {mx} <= {harness.MAX_INDEX} : OK ({len(fund)} règlements, min "
        f"{fund['fundingTime'].min()})")
    return fund


def build_inputs(df1, configs, var, log):
    df4, F, X, Y = _orig_build_inputs(df1, configs, var, log)
    fund = load_fund(log)
    if var.startswith("fplac"):
        fund = placebo_blocks(fund, int(var[5:]))
        log(f"[placebo funding] taux permutés par blocs de 30 jours, seed {int(var[5:])}")
    G = funding_4h(fund, df1.index, df4.index)
    if var == "featlag1":
        G = G.shift(1)
    assert G.index.max() <= harness.MAX_INDEX
    for col in FUND_ALL:
        X[col] = G[col].to_numpy()
    rf = X[FU].notna().all(axis=1)
    Y["__common__"] = Y["__common__"].where(rf)
    first = X.index[rf.to_numpy()][0]
    log(f"[funding 4 h] bougies 4 h avec fu défini : {int(rf.sum())}/{len(rf)} ; première {first}")
    return df4, F, X, Y


def config_train(df4, X, Y, c, k, shuffle_seed):
    """b5.config_train + row_filter (features de funding requises non NaN, même si non utilisées par le modèle)."""
    fH = make_folds(horizon=c["Hh"])[k - 1]
    assert fH.k == k
    tr, _ = split_fold(df4, fH, horizon=c["Hh"])
    need = list(dict.fromkeys(c["features"] + c.get("row_filter", [])))
    Xa = X.loc[tr.index, need]
    ya = Y[c["id"]].loc[tr.index]
    clean = (~Xa.isna().any(axis=1)) & (~ya.isna())
    Xtr = X.loc[tr.index[clean.to_numpy()], c["features"]]
    ytr = ya.loc[clean].to_numpy().copy()
    assert Xtr.index.max() + 4 * (c["H"] + 1) * H1 < fH.test_start
    if shuffle_seed is not None:
        ytr = np.random.default_rng([shuffle_seed, k, c["ci"]]).permutation(ytr)
    return fH, Xtr, ytr


b5.build_inputs = build_inputs
b5.config_train = config_train


# ------------------------------------------------------------------ configurations

def resolve_features(spec) -> list[str]:
    if spec == "f4":
        return list(agg4h.F4)
    if spec == "f4fu":
        return list(agg4h.F4) + list(FU)
    if spec == "fu":
        return list(FU)
    if spec in (None, []):
        return []
    return list(spec)


def prep(cfg: dict, ci: int = 0) -> dict:
    c2 = copy.deepcopy(cfg)
    c2["features"] = resolve_features(cfg["features"])
    c = b5.prep(c2, ci)
    c["row_filter"] = resolve_features(cfg.get("row_filter"))
    return c


def load_union(cfgN: dict, log, check_git: bool = True) -> list[dict]:
    out = []
    for i, ent in enumerate(cfgN["union"]):
        p = ROOT / ent["path"]
        local = git_blob_sha1(p)
        at_head = harness.git("rev-parse", f"HEAD:{ent['path']}") if check_git else ent["git_blob"]
        if not (local == ent["git_blob"] == at_head):
            raise RuntimeError(f"{ent['id']} : blob local {local} / HEAD {at_head} != déclaré {ent['git_blob']}")
        c = json.loads(p.read_text(encoding="utf-8"))
        assert c["id"] == ent["id"] and c.get("procedure", "single") == "single"
        out.append(prep(c, i))
    log(f"[union] {len(out)} configurations vérifiées (blob local = blob HEAD = déclaré) : "
        + ", ".join(f"{c['id']}({c['label']['kind']} H={c['H']}, {len(c['features'])} features, "
                    f"{len(c['hps'])}x{len(c['maps'])})" for c in out))
    return out


def apply_variant(var: str, configs: list[dict]) -> None:
    if var.startswith("tbscale"):
        f = float(var[7:])
        hit = 0
        for c in configs:
            lab = c["label"]
            if lab["kind"] != "tb":
                continue
            for side in ("up", "down"):
                b = lab[side]
                if b["type"] == "vol":
                    b["k"] = round(float(b["k"]) * f, 6)
                elif b["type"] == "fixed_log":
                    b["v"] = round(float(b["v"]) * f, 6)
                else:
                    raise ValueError(b)
            hit += 1
        assert hit > 0, "tbscale : aucun label triple barrière"
        return
    if var.startswith("fplac"):
        assert any(set(FU) & set(c["features"]) for c in configs), "fplac : aucune feature de funding au modèle"
        return
    b5.apply_variant(var, configs)


# ------------------------------------------------------------------ registre

def feat_txt(cfg: dict) -> str:
    f = cfg["features"]
    names = {"f4": "f4 (14)", "f4fu": "f4+fu (18 : f4 + fr_cur,fr_mean3,fr_mean21,fr_z90)",
             "fu": "fu seul (4 : fr_cur,fr_mean3,fr_mean21,fr_z90)"}
    return names.get(f, str(f)) + " ; lignes où fu est défini (row_filter fu, à partir de 2020-01-31)"


def label_txt(lab: dict) -> str:
    if lab["kind"] == "net":
        return f"net de coûts H={lab['H']} bougies 4 h"
    return (f"triple barrière H={lab['H']} bougies 4 h haut={json.dumps(lab['up'], separators=(',', ':'))} "
            f"bas={json.dumps(lab['down'], separators=(',', ':'))}" + (" expiration=NaN" if lab.get("expire_nan") else ""))


def registry_line(out: dict, date: str) -> str:
    cfg = out["config"]
    fs = ";".join(f"{float(r['model']['sharpe']):.3f}" for r in out["folds"])  # "nan" (fold à plat) -> nan
    ft = ";".join(str(r["model"]["trades"]) for r in out["folds"])
    m = out["concatenated"]["model"]
    eligible = isinstance(m["sharpe"], float) and m["sharpe"] > out["concatenated"]["baseline"]["sharpe"]
    statut = ("éligible (> baseline ; candidat désigné dans le rapport P3-R7)" if eligible
              else "abandonné (<= baseline)")
    run_cmd = f".venv\\Scripts\\python experiments\\b7.py experiments\\configs\\{cfg['id']}.json"
    reg_cmd = f".venv\\Scripts\\python experiments\\b7.py --register {cfg['id']}"
    pair = f" ; paire {cfg['pair']}" if cfg.get("pair") else ""
    if cfg.get("procedure") == "nested":
        ch = ";".join(r["chosen_config"] for r in out["folds"])
        desc = (f"emboîté lgbm bougies 4 h (union {','.join(out['union_ids'])}) | features=f4+fu (celles de la config "
                f"choisie) ; lignes où fu est défini | label/H=par fold ({ch}) ; labels de l'union repris de P3-B5 "
                f"et choisis au vu de son OOS (déclaré) | grille hp et mapping = union des configs (blobs git "
                f"vérifiés, experiments/configs/{cfg['id']}.json) ; validation commune H=30 bougies 4 h | grille "
                f"interne {out['grid_size_total']} couples par fold | folds 1-9 (make_folds horizon 123 h pour la "
                f"validation, horizon 4H+3 h de la config choisie pour le train final)")
    else:
        g = len(harness.hp_grid(cfg))
        mp = json.dumps(cfg["mapping"], separators=(",", ":"))
        desc = (f"{cfg['model']} bougies 4 h | features={feat_txt(cfg)}{pair} | label {label_txt(cfg['label'])} ; "
                f"{BRIEF_DECL} | grille hp={json.dumps(cfg['grid'], separators=(',', ':'))} mapping quantiles={mp} | "
                f"grille interne {g}x{out['grid_size_total'] // g}={out['grid_size_total']} | folds 1-9 (make_folds "
                f"horizon {agg4h.horizon_hours(cfg['label']['H'])} h)")
    line = (f"| {cfg['id']} | {date} | {out['commit'][:7]} | seed={out['seed']} | {desc} | Sharpe/fold {fs} | "
            f"trades/fold {ft} | Sharpe concat {m['sharpe']:.4f} | trades concat {m['trades']} | "
            f"écart baseline {out['delta_sharpe_vs_baseline']:+.4f} | écart B&H {out['delta_sharpe_vs_bh']:+.4f} | "
            f"{statut} | `{run_cmd}` puis `{reg_cmd}` |")
    return line


def register(tid: str) -> int:
    out = json.loads((RESULTS / f"{tid}.json").read_text(encoding="utf-8"))
    if out["code_dirty"] or out["variant"] != "base" or out["shuffle_seed"] is not None:
        raise RuntimeError("JSON non éligible au registre (code non commité ou variante)")
    lines = REGISTRE.read_text(encoding="utf-8").splitlines()
    if any(ln.split("|")[1].strip() == tid for ln in lines):
        raise RuntimeError(f"{tid} déjà au registre")
    if len(lines) >= REG_MAX:
        raise RuntimeError(f"budget P3-B7 : registre limité à {REG_MAX} lignes")
    line = registry_line(out, pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d"))
    n_pipes = line.count("|")
    if n_pipes != 19:
        raise RuntimeError(f"ligne de registre mal formée : {n_pipes} « | » (attendu 19)")
    with open(REGISTRE, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(line + "\n")
    print(f"registre : ligne {tid} ajoutée ({len(lines) + 1} lignes)")
    return 0


# ------------------------------------------------------------------ autotest synthétique

def synthetic_funding(seed: int = 0, start="2020-01-01 00:00", end="2025-09-30 16:00") -> pd.DataFrame:
    t = pd.date_range(pd.Timestamp(start, tz="UTC"), pd.Timestamp(end, tz="UTC"), freq="8h")
    rng = np.random.default_rng(seed)
    jit = pd.to_timedelta(rng.integers(0, 48, len(t)), unit="ms")  # gigue 0..47 ms comme les archives réelles
    jit = jit.where(rng.random(len(t)) < 0.5, pd.Timedelta(0))
    r = 1e-4 + np.cumsum(rng.normal(0, 2e-5, len(t))) * 0.1 + rng.normal(0, 5e-5, len(t))
    return pd.DataFrame({"fundingTime": (t + jit).as_unit("ns"), "fundingRate": np.round(r, 8)})


def funding_selftest(log, seed: int = 0) -> None:
    from nested import synthetic_df
    rng = np.random.default_rng(seed)
    df1 = synthetic_df()
    df4 = agg4h.aggregate_4h(df1)
    fund = synthetic_funding(seed)
    G = funding_4h(fund, df1.index, df4.index)
    ft = fund["fundingTime"]
    # 0. règle : la valeur à T est celle du dernier règlement fundingTime <= T + 4 h
    i0 = int(df4.index.searchsorted(pd.Timestamp("2020-02-15", tz="UTC")))  # après le 1er règlement synthétique
    pts = np.sort(rng.choice(np.arange(i0, len(df4) - 50), 300, replace=False))
    bad = 0
    for i in pts:
        T = df4.index[i]
        j = int(np.searchsorted(pd.DatetimeIndex(ft).asi8, (T + 4 * H1).as_unit("ns").value, side="right")) - 1
        bad += int(G["fr_cur"].iloc[i] != fund["fundingRate"].iloc[j])
    log(f"[funding 4 h règle] 300 bougies : fr_cur(T) = taux du dernier règlement fundingTime <= T+4h : "
        f"{300 - bad}/300")
    assert bad == 0
    # bornes : règlement exactement à T+4h utilisé ; à T+4h+1 ms, non
    T = df4.index[3000]
    f2 = pd.concat([fund[ft < T], pd.DataFrame({"fundingTime": [T + 4 * H1], "fundingRate": [0.0123]})],
                   ignore_index=True)
    f3 = pd.concat([fund[ft < T], pd.DataFrame({"fundingTime": [T + 4 * H1 + pd.Timedelta(milliseconds=1)],
                                                "fundingRate": [0.0123]})], ignore_index=True)
    g2 = funding_4h(f2, df1.index, df4.index[2900:3002]).loc[T, "fr_cur"]
    g3 = funding_4h(f3, df1.index, df4.index[2900:3002]).loc[T, "fr_cur"]
    log(f"[funding 4 h bornes] règlement à T+4h -> utilisé à T : {g2 == 0.0123} ; à T+4h+1 ms -> non utilisé à T : "
        f"{g3 != 0.0123}")
    assert g2 == 0.0123 and g3 != 0.0123
    # 1. anti-fuite : règlements postérieurs à la clôture de T modifiés -> rien ne change pour les bougies <= T
    nleak, sens = 0, 0
    for i in pts[:40]:
        T = df4.index[i]
        fut = (ft > T + 4 * H1).to_numpy()
        f2 = fund.copy()
        f2.loc[fut, "fundingRate"] = rng.normal(0, 0.01, int(fut.sum()))
        f2 = f2[~(fut & (rng.random(len(f2)) < 0.2))].reset_index(drop=True)  # et suppression de 20 % des futurs
        G2 = funding_4h(f2, df1.index, df4.index)
        nleak += int(not np.array_equal(G.iloc[: i + 1].to_numpy(), G2.iloc[: i + 1].to_numpy(), equal_nan=True))
        f3 = fund.copy()  # sensibilité : le dernier règlement <= T+4h est bien lu
        j = int(np.flatnonzero(~fut)[-1])
        f3.loc[j, "fundingRate"] += 0.001
        sens += int(not np.array_equal(funding_4h(f3, df1.index, df4.index[i: i + 1]).to_numpy(),
                                       G.iloc[i: i + 1].to_numpy(), equal_nan=True))
    log(f"[funding 4 h anti-fuite] 40 bougies T : règlements fundingTime > T+4h modifiés / supprimés -> features <= T "
        f"identiques : {40 - nleak}/40 ; sensibilité (dernier règlement <= T+4h +0,001) : {sens}/40")
    assert nleak == 0 and sens == 40
    # 2. direct = lot
    neq, neq_pol = 0, 0
    for i in pts[:200]:
        T = df4.index[i]
        known = fund[ft <= T + 4 * H1].tail(bf.FUNDING_LOOKBACK)
        lv = funding_4h_live(known, T).to_numpy()
        neq += int(np.array_equal(lv, G.iloc[i].to_numpy(), equal_nan=True))
        fut = fund[(ft > T + 4 * H1) & (ft <= T + 4 * H1 + pd.Timedelta(hours=24))]
        pol = pd.concat([known, fut], ignore_index=True)  # 90 connus + règlements futurs : ignorés
        neq_pol += int(np.array_equal(funding_4h_live(pol, T).to_numpy(), G.iloc[i].to_numpy(), equal_nan=True))
    log(f"[funding 4 h direct = lot] 200 bougies : direct (90 derniers règlements connus à T+4h) bit à bit = lot "
        f"{neq}/200 ; direct avec règlements futurs fournis (ignorés) = lot {neq_pol}/200")
    assert neq == 200 and neq_pol == 200
    # 3. placebo
    p = placebo_blocks(fund, 1)
    same_mult = np.array_equal(np.sort(p["fundingRate"].to_numpy()), np.sort(fund["fundingRate"].to_numpy()))
    moved = float(np.mean(p["fundingRate"].to_numpy() != fund["fundingRate"].to_numpy()))
    log(f"[placebo] permutation par blocs de 30 jours : mêmes taux (multiensemble) {same_mult} ; part déplacée "
        f"{moved:.3f} ; horodatages inchangés {p['fundingTime'].equals(fund['fundingTime'])}")
    assert same_mult and moved > 0.5 and p["fundingTime"].equals(fund["fundingTime"])


def selftest(log) -> int:
    global _FUNDING_OVERRIDE
    from nested import synthetic_df
    log("AUTO-TEST sur données SYNTHÉTIQUES (aucune lecture du dataset ni du funding)")
    funding_selftest(log)
    _FUNDING_OVERRIDE = synthetic_funding(0)
    df1 = synthetic_df()
    ids = [f"E{i}" for i in range(40, 47)]
    cfgs = {i: json.loads((CONFIGS / f"{i}.json").read_text(encoding="utf-8")) for i in ids}
    for tid in ("E40", "E41", "E46"):
        c = prep(cfgs[tid], 0)
        c["hps"], c["maps"] = c["hps"][:1], c["maps"][:2]
        out = b5.run({"id": "selftest", "seed": 42}, [c], False, None, log, df1=df1, skip_baseline=True)
        assert out["position_changes_off_4h_grid"] == 0
        n1 = [r["n_train"] for r in out["folds"]]
        log(f"[selftest] {tid} réduit : n_train/fold {n1} ; train_start fold 1 {out['folds'][0]['train_start']}")
        fu4 = funding_4h(_FUNDING_OVERRIDE, df1.index, agg4h.aggregate_4h(df1).index)[FU].notna().all(axis=1)
        assert out["folds"][0]["train_start"] >= fu4.index[fu4.to_numpy()][0]
    # paire : mêmes lignes de train
    ca, cb = prep(cfgs["E40"], 0), prep(cfgs["E41"], 0)
    df4, F, X, Y = build_inputs(df1, [ca, cb], "base", log)
    for k in (1, 5, 9):
        _, xa, _ = config_train(df4, X, Y, ca, k, None)
        _, xb, _ = config_train(df4, X, Y, cb, k, None)
        assert xa.index.equals(xb.index)
    log("[selftest] paire E40/E41 : mêmes lignes de train (folds 1, 5, 9) : OK")
    for var in ("fplac1", "tbscale0.8", "tbscale1.2", "shuf1", "featlag1"):
        cc = [prep(cfgs["E41"], 0)]
        cc[0]["hps"], cc[0]["maps"] = cc[0]["hps"][:1], cc[0]["maps"][:2]
        apply_variant(var, cc)
        sh = int(var[4:]) if var.startswith("shuf") else None
        out = b5.run({"id": "selftest", "seed": 42}, cc, False, sh, log, var=var, df1=df1, skip_baseline=True)
        log(f"[selftest variante {var}] OK : label {out['effective_configs'][0]['label']}")
    small = []
    for tid in ("E41", "E43", "E45"):
        c = prep(cfgs[tid], len(small))
        c["hps"], c["maps"] = c["hps"][:1], c["maps"][:2]
        small.append(c)
    b5.run({"id": "selftest", "seed": 42}, small, True, None, log, df1=df1, skip_baseline=True)
    _FUNDING_OVERRIDE = None
    log("auto-test b7 : OK")
    return 0


# ------------------------------------------------------------------ main

def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("config", type=Path, nargs="?")
    ap.add_argument("--variant", default="base")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--register", default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    log = harness.Tee()
    if args.register:
        return register(args.register)
    commit = harness.git("rev-parse", "HEAD")
    dirty = harness.git("status", "--porcelain", "--", *CODE_PATHS) != ""
    log(f"commit {commit} ; code modifié non commité ({', '.join(CODE_PATHS)}) : {dirty}")
    if args.selftest:
        return selftest(log)
    cfg = json.loads(args.config.read_text(encoding="utf-8"))
    nested = cfg.get("procedure") == "nested"
    configs = load_union(cfg, log) if nested else [prep(cfg, 0)]
    var = args.variant
    shuffle_seed = int(var[4:]) if var.startswith("shuf") else None
    apply_variant(var, configs)
    log(f"variante : {var}")
    for c in configs:
        log(f"  {c['id']} : features modèle {len(c['features'])} {c['features']} ; row_filter {c['row_filter']}")
    out = b5.run(cfg, configs, nested, shuffle_seed, log, var=var)
    sig = out.pop("_signal")
    proba = out.pop("_proba")
    for c, e in zip(configs, out["effective_configs"]):
        e["row_filter"] = c["row_filter"]
    out = {"commit": commit, "code_dirty": dirty, "variant": var, **out}
    path = args.out or (RESULTS / (f"{cfg['id']}.json" if var == "base" else f"{cfg['id']}_{var}.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    if var == "base" and args.out is None:
        d = RESULTS / f"adv_{cfg['id']}"
        d.mkdir(parents=True, exist_ok=True)
        sig.to_frame("signal").to_parquet(d / "signal_1h.parquet", engine="pyarrow")
        proba.to_parquet(d / "proba_4h.parquet", engine="pyarrow")
        log(f"signal 1 h et probas 4 h : {d}")
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        json.dump(harness.jsonable(out), fh, indent=1, ensure_ascii=False)
        fh.write("\n")
    with open(path.with_suffix(".txt"), "w", encoding="utf-8", newline="\n") as fh:
        fh.write("\n".join(log.lines) + "\n")
    print(f"écrit : {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
