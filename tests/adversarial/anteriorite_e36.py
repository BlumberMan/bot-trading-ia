"""Avocat du diable P3-A4, T9 : antériorité du pré-enregistrement P3-B5 (a0ddea6) et intégrité.
Usage : .venv\\Scripts\\python tests\\adversarial\\anteriorite_e36.py   (git + JSON de résultats, aucune donnée de prix lue)
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PRE, RESC = "a0ddea6", "2db2f81"
ROOT = Path(__file__).resolve().parents[2]
IDS = [f"E{i}" for i in range(29, 40)]


def git(*a, check=True):
    r = subprocess.run(["git", "-C", str(ROOT), *a], capture_output=True, text=True, encoding="utf-8")
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
    files = [f"experiments/configs/{i}.json" for i in IDS] + [
        "experiments/agg4h.py", "experiments/b5.py", "experiments/b4.py", "experiments/harness.py"]
    for f in files:
        a, _ = git("rev-parse", f"{PRE}:{f}")
        h, _ = git("rev-parse", f"HEAD:{f}")
        first, _ = git("log", "--diff-filter=A", "--format=%h", "--", f)
        nmod, _ = git("log", "--format=%h", f"{PRE}..HEAD", "--", f)
        ok = a == h and not nmod
        bad += not ok
        print(f"{f} : blob {PRE} {a[:10]} HEAD {h[:10]} ; ajouté en {first} ; commits après {PRE} : "
              f"{nmod.split() if nmod else []} -> {'OK' if ok else 'ÉCART'}")
    d, _ = git("diff", "--stat", PRE, "HEAD", "--", "src")
    print(f"diff {PRE}..HEAD src : {d or 'aucun'}")
    bad += bool(d)
    revs, _ = git("rev-list", PRE)
    hits = []
    for r in revs.splitlines():
        t, _ = git("ls-tree", "-r", "--name-only", r, "experiments/results")
        if any(f"/{i}" in x for x in t.splitlines() for i in IDS) or any("adv_E29" in x or "adv_E3" in x
                                                                       for x in t.splitlines()):
            hits.append(r[:7])
    print(f"commits <= {PRE} ({len(revs.splitlines())} commits) contenant un résultat E29..E39 : {hits}")
    bad += bool(hits)
    pre_ts = int(git("log", "-1", "--format=%ct", PRE)[0])
    print(f"date de commit {PRE} : {datetime.fromtimestamp(pre_ts, timezone.utc)}")
    for i in IDS:
        p = ROOT / "experiments" / "results" / f"{i}.json"
        j = json.loads(p.read_text(encoding="utf-8"))
        ok = j["commit"].startswith("a0ddea6") and j["code_dirty"] is False and j["variant"] == "base"
        bad += not ok
        mt = datetime.fromtimestamp(os.path.getmtime(p), timezone.utc)
        first, _ = git("log", "--diff-filter=A", "--format=%h", "--", f"experiments/results/{i}.json")
        print(f"{i}.json : commit déclaré {j['commit'][:7]} code_dirty {j['code_dirty']} variante {j['variant']} ; "
              f"ajouté en {first} ; mtime {mt} (info : >= commit pré-enregistrement {mt.timestamp() >= pre_ts}) "
              f"-> {'OK' if ok else 'ÉCART'}")
    d, _ = git("diff", "--stat", "gonogo-v2", "HEAD", "--", "GONOGO.md")
    print(f"GONOGO.md vs tag gonogo-v2 : {d or 'identique'}")
    touched, _ = git("diff", "--name-only", f"{PRE}..HEAD")
    print(f"fichiers modifiés {PRE}..HEAD hors experiments/results, reports/adversarial, tests/adversarial : "
          f"{[x for x in touched.splitlines() if not x.startswith(('experiments/results', 'reports/adversarial', 'tests/adversarial'))]}")
    print(f"T9 RÉSULTAT : écarts {int(bad)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
