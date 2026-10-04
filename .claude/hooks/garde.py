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
ECRITURE = re.compile(r"(>|\btee\b|sed\s+-i|\brm\b|\bmv\b|\bcp\b|\btouch\b|\btruncate\b|git\s+(checkout|restore|reset|rm|mv|stash|apply))")
INOFFENSIF = re.compile(r"\d?>&\d|\d?>\s*/dev/null")

def segments(cmd):
    return [s for s in re.split(r"&&|\|\||;|\||\n", cmd) if s.strip()]

def ecriture(seg):
    m = ECRITURE.search(INOFFENSIF.sub("", seg))
    return m.group(0) if m else None

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
    for seg in segments(cmd):
        if "gonogo-v" in seg and re.search(r"git\s+tag", seg) and not re.search(r"git\s+tag\s+(-l|--list)", seg):
            block("Le tag gonogo-v1 est reserve a Iyad.")
        if "gonogo-v" in seg and re.search(r"git\s+push", seg) and re.search(r"(--delete|--force|-f\b|:refs)", seg):
            block("Le tag gonogo-v1 est reserve a Iyad.")
        if "GONOGO" in seg and tag_exists():
            op = ecriture(seg)
            if op:
                block("GONOGO.md est fige (operation detectee : '" + op + "').")
        if ".claude" in seg or "CLAUDE.md" in seg:
            op = ecriture(seg)
            if op:
                block("Les fichiers de regles sont proteges (operation detectee : '" + op + "').")
        if re.search(r"(^|[\s/])\.env\b", seg) and not re.search(r"(check-ignore|--exclude=\.env)", seg):
            block("Le fichier .env est reserve a Iyad.")

sys.exit(0)