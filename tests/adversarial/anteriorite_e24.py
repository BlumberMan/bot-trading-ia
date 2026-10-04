"""Avocat du diable P3-A3, T9 : antériorité du pré-enregistrement P3-B4 (e0249a3) et intégrité.
Usage : .venv\\Scripts\\python tests\\adversarial\\anteriorite_e24.py   (git uniquement, aucune donnée lue)
"""

from __future__ import annotations

import subprocess
import sys

PRE, RESC = "e0249a3", "8faaf2a"


def git(*a, check=True):
    r = subprocess.run(["git", *a], capture_output=True, text=True, encoding="utf-8")
    if check and r.returncode:
        raise RuntimeError(r.stderr)
    return r.stdout.strip(), r.returncode


def main() -> int:
    sys.stdout.reconfigure(encoding="utf-8")
    bad = 0
    print("HEAD", git("rev-parse", "HEAD")[0])
    print("pré-enregistrement", git("log", "-1", "--format=%H %ad %s", "--date=iso", PRE)[0])
    print("résultats", git("log", "-1", "--format=%H %ad %s", "--date=iso", RESC)[0])
    _, rc = git("merge-base", "--is-ancestor", PRE, RESC, check=False)
    print(f"{PRE} ancêtre de {RESC} : {rc == 0}")
    bad += rc != 0
    files = [f"experiments/configs/E{i}.json" for i in range(18, 29)] + ["experiments/b4.py", "experiments/labels_b4.py"]
    for f in files:
        a, _ = git("rev-parse", f"{PRE}:{f}")
        h, _ = git("rev-parse", f"HEAD:{f}")
        first, _ = git("log", "--diff-filter=A", "--format=%h", "--", f)
        nmod, _ = git("log", "--format=%h", f"{PRE}..HEAD", "--", f)
        ok = a == h and not nmod
        bad += not ok
        print(f"{f} : blob {PRE} {a[:10]} HEAD {h[:10]} ; ajouté en {first} ; commits après {PRE} : "
              f"{nmod.split() if nmod else []} -> {'OK' if ok else 'ÉCART'}")
    tree, _ = git("ls-tree", "-r", "--name-only", PRE, "experiments/results")
    pre_res = [x for x in tree.splitlines() if any(f"/E{i}" in x for i in range(18, 29)) or "adv_E2" in x]
    print(f"fichiers de résultat E18..E28 présents dans l'arbre de {PRE} : {pre_res}")
    bad += bool(pre_res)
    revs, _ = git("rev-list", PRE)
    hits = []
    for r in revs.splitlines():
        t, _ = git("ls-tree", "-r", "--name-only", r, "experiments/results")
        if any(f"/E{i}" in x for x in t.splitlines() for i in range(18, 29)):
            hits.append(r[:7])
    print(f"commits <= {PRE} contenant un résultat E18..E28 : {hits}")
    bad += bool(hits)
    for p in ("src", "experiments/harness.py", "experiments/extra_features.py"):
        d, _ = git("diff", "--stat", PRE, "HEAD", "--", p)
        print(f"diff {PRE}..HEAD {p} : {d or 'aucun'}")
        bad += bool(d)
    d, _ = git("diff", "--stat", "gonogo-v2", "HEAD", "--", "GONOGO.md")
    print(f"GONOGO.md vs tag gonogo-v2 : {d or 'identique'}")
    touched, _ = git("diff", "--name-only", f"{PRE}..HEAD")
    print(f"fichiers modifiés {PRE}..HEAD hors experiments/results, reports/adversarial, tests/adversarial : "
          f"{[x for x in touched.splitlines() if not x.startswith(('experiments/results', 'reports/adversarial', 'tests/adversarial'))]}")
    print(f"T9 RÉSULTAT : écarts {int(bad)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
