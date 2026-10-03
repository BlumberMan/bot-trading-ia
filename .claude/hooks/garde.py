import json, os, re, subprocess, sys

data = json.load(sys.stdin)
tool = data.get("tool_name", "")
inp = data.get("tool_input", {}) or {}
root = os.environ.get("CLAUDE_PROJECT_DIR", os.getcwd())

def block(msg):
    print("BLOQUE PAR LE GARDE : " + msg + " Seul Iyad peut faire cette action a la main.", file=sys.stderr)
    sys.exit(2)

def tag_exists():
    try:
        out = subprocess.run(["git", "tag", "-l", "gonogo-v1"], cwd=root, capture_output=True, text=True).stdout.strip()
        return out == "gonogo-v1"
    except Exception:
        return True

PROTEGES = ("CLAUDE.md",)
ECRITURE = r"((^|[^0-9&>-])>|\btee\b|sed\s+-i|\brm\b|\bmv\b|\bcp\b|\btouch\b|git\s+(checkout|restore|reset|rm|mv|stash))"

if tool in ("Edit", "Write", "MultiEdit", "NotebookEdit"):
    path = (inp.get("file_path") or inp.get("notebook_path") or "").replace("\\", "/")
    name = os.path.basename(path)
    if "/.claude/" in "/" + path.lstrip("./"):
        block("Le dossier .claude/ est protege.")
    if name in PROTEGES:
        block(name + " est protege.")
    if name == ".env":
        block("Le fichier .env est reserve a Iyad.")
    if name == "GONOGO.md" and tag_exists():
        block("GONOGO.md est fige (tag gonogo-v1).")

if tool == "Bash":
    cmd = inp.get("command", "")
    if "gonogo-v1" in cmd and re.search(r"git\s+tag", cmd) and not re.search(r"git\s+tag\s+-l", cmd):
        block("Le tag gonogo-v1 est reserve a Iyad.")
    if "gonogo-v1" in cmd and re.search(r"git\s+push", cmd) and re.search(r"(--delete|--force|-f\b|:refs)", cmd):
        block("Le tag gonogo-v1 est reserve a Iyad.")
    if "GONOGO" in cmd and tag_exists() and re.search(ECRITURE, cmd):
        block("GONOGO.md est fige.")
    if (".claude" in cmd or "CLAUDE.md" in cmd) and re.search(ECRITURE, cmd):
        block("Les fichiers de regles sont proteges.")
    if re.search(r"(^|[\s/])\.env\b", cmd) and not re.search(r"(check-ignore|--exclude=\.env)", cmd):
        block("Le fichier .env est reserve a Iyad.")

sys.exit(0)
