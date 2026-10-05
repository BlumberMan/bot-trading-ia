r"""Avocat du diable P3-A5, T9 : antériorité du pré-enregistrement P3-B7 (db7f4d5), portée de 5620099, intégrité.
Usage : .venv\Scripts\python tests\adversarial\anteriorite_e40.py CHEMIN_JSON_DB7
(CHEMIN_JSON_DB7 = sortie de run_b7_db7f4d5.py, b7.py de db7f4d5 exécuté hors arbre ; git + JSON, aucune donnée lue)
"""

from __future__ import annotations

import ast
import json
import subprocess
import sys
from pathlib import Path

PRE, FIX, RESC = "db7f4d5", "5620099", "a32fb19"
ROOT = Path(__file__).resolve().parents[2]
IDS = [f"E{i}" for i in range(40, 48)]


def git(*a, check=True):
    r = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, encoding="utf-8")
    if check and r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout.strip(), r.returncode


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    bad = 0
    print("HEAD", git("rev-parse", "HEAD")[0])
    for c in (PRE, FIX, RESC):
        print(git("log", "-1", "--format=%H %ad %s", "--date=iso", c)[0][:160])
    for a, b in ((PRE, FIX), (FIX, RESC), (PRE, RESC)):
        _, rc = git("merge-base", "--is-ancestor", a, b, check=False)
        print(f"{a} ancêtre de {b} : {rc == 0}")
        bad += rc != 0
    print(f"parent de {FIX} : {git('rev-parse', '--short', FIX + '^')[0]} ; parent de {RESC} : "
          f"{git('rev-parse', '--short', RESC + '^')[0]}")
    files = [f"experiments/configs/{i}.json" for i in IDS] + [
        "experiments/agg4h.py", "experiments/b5.py", "experiments/b4.py", "experiments/harness.py"]
    for f in files:
        a, _ = git("rev-parse", f"{PRE}:{f}")
        h, _ = git("rev-parse", f"HEAD:{f}")
        nmod, _ = git("log", "--format=%h", f"{PRE}..HEAD", "--", f)
        ok = a == h and not nmod
        bad += not ok
        print(f"{f} : blob {PRE} {a[:10]} HEAD {h[:10]} ; commits après {PRE} : {nmod.split() if nmod else []} -> "
              f"{'OK' if ok else 'ÉCART'}")
    d, _ = git("diff", "--stat", PRE, "HEAD", "--", "src")
    print(f"diff {PRE}..HEAD src : {d or 'aucun'}")
    bad += bool(d)
    nmod, _ = git("log", "--format=%h", f"{PRE}..HEAD", "--", "experiments/b7.py")
    print(f"commits touchant b7.py après {PRE} : {nmod.split()}")
    bad += nmod.split() != [FIX]
    st, _ = git("show", "--numstat", "--format=", FIX)
    print(f"{FIX} numstat : {st}")
    bad += st.split() != ["1", "1", "experiments/b7.py"]
    diff, _ = git("diff", "-U0", f"{FIX}^", FIX, "--", "experiments/b7.py")
    hunks = [ln for ln in diff.splitlines() if ln.startswith("@@")]
    print(f"{FIX} hunks : {hunks}")
    print("\n".join(ln for ln in diff.splitlines() if ln[:1] in "+-" and not ln.startswith(("+++", "---"))))
    src = (ROOT / "experiments" / "b7.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn_of_line = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            for ln in range(node.lineno, node.end_lineno + 1):
                fn_of_line[ln] = node.name
    new_line = int(hunks[0].split("+")[1].split(",")[0].split()[0])
    print(f"ligne modifiée {new_line} dans la fonction : {fn_of_line.get(new_line)}")
    bad += fn_of_line.get(new_line) != "registry_line"
    callers = sorted({fn_of_line.get(n.lineno, "<module>") for n in ast.walk(tree)
                      if isinstance(n, ast.Call) and getattr(n.func, "id", None) == "registry_line"})
    print(f"appels de registry_line dans b7.py : {callers}")
    bad += callers != ["register"]
    other = []
    for p in list((ROOT / "experiments").glob("*.py")) + list((ROOT / "src").rglob("*.py")):
        if p.name != "b7.py" and "b7.registry_line" in p.read_text(encoding="utf-8"):
            other.append(str(p.relative_to(ROOT)))
    print(f"autres fichiers appelant b7.registry_line : {other}")
    bad += bool(other)
    revs, _ = git("rev-list", PRE)
    hits = []
    for r in revs.splitlines():
        t, _ = git("ls-tree", "-r", "--name-only", r, "experiments/results")
        if any(f"/{i}." in x or f"/{i}_" in x or f"adv_{i}" in x for x in t.splitlines() for i in IDS):
            hits.append(r[:7])
    print(f"commits <= {PRE} ({len(revs.splitlines())}) contenant un résultat E40..E47 : {hits}")
    bad += bool(hits)
    j = json.loads((ROOT / "experiments" / "results" / "E40.json").read_text(encoding="utf-8"))
    ok = j["commit"].startswith(git("rev-parse", PRE)[0]) and j["code_dirty"] is False and j["variant"] == "base"
    print(f"E40.json : commit {j['commit']} code_dirty {j['code_dirty']} variante {j['variant']} -> {'OK' if ok else 'ÉCART'}")
    bad += not ok
    first, _ = git("log", "--diff-filter=A", "--format=%h %ad", "--date=iso", "--", "experiments/results/E40.json")
    print(f"E40.json ajouté en : {first}")
    # b7.py de db7f4d5 exécuté hors arbre vs MA relance à HEAD vs E40.json
    old = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
    mine = json.loads((ROOT / "reports" / "adversarial" / "E40_adv_base.json").read_text(encoding="utf-8"))
    so, sm, sj = (float(x["concatenated"]["model"]["sharpe"]) for x in (old, mine, j))
    to, tm = old["concatenated"]["model"]["trades"], mine["concatenated"]["model"]["trades"]
    same = all((a["best_hp"], a["best_map"]) == (b["best_hp"], b["best_map"]) for a, b in zip(old["folds"], mine["folds"]))
    print(f"b7.py@{PRE} (hors arbre, commit déclaré {old['commit'][:7]}, code_dirty {old['code_dirty']}) : Sharpe "
          f"{so:.9f} trades {to} ; b7.py@HEAD (ma base) : Sharpe {sm:.9f} trades {tm} ; E40.json : {sj:.9f} ; "
          f"|écart| {abs(so - sm):.1e} ; choix par fold identiques : {same}")
    bad += abs(so - sm) > 1e-9 or to != tm or not same
    d, _ = git("diff", "--stat", "gonogo-v2", "HEAD", "--", "GONOGO.md")
    print(f"GONOGO.md vs tag gonogo-v2 : {d or 'identique'}")
    reg = [ln for ln in (ROOT / "experiments" / "REGISTRE.md").read_text(encoding="utf-8").splitlines() if ln.strip()]
    print(f"REGISTRE.md : {len(reg)} lignes ; ids {reg[0].split('|')[1].strip()}..{reg[-1].split('|')[1].strip()}")
    print(f"T9 RÉSULTAT : écarts {int(bad)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
