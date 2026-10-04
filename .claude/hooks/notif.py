import json, sys, urllib.request

TOPIC = "bot-iyad-aba04a32a945"
MARQUEUR = "ATTENTE IYAD"

def send(title, msg, prio="default"):
    req = urllib.request.Request(
        "https://ntfy.sh/" + TOPIC,
        data=msg[:400].encode("utf-8"),
        headers={"Title": title, "Priority": prio},
    )
    try:
        urllib.request.urlopen(req, timeout=10)
    except Exception:
        pass

def dernier_texte(path):
    texte = ""
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                try:
                    e = json.loads(line)
                except Exception:
                    continue
                if e.get("type") != "assistant":
                    continue
                for c in (e.get("message", {}) or {}).get("content", []) or []:
                    if isinstance(c, dict) and c.get("type") == "text" and c.get("text"):
                        texte = c["text"]
    except Exception:
        pass
    return texte

data = json.load(sys.stdin)
event = data.get("hook_event_name", "")

if event == "Notification":
    send("Bot: Claude attend", data.get("message", "Permission ou reponse requise"), "high")
elif event == "Stop":
    t = dernier_texte(data.get("transcript_path", ""))
    if MARQUEUR in t:
        prio = "urgent" if "ALERTE" in t else "high"
        send("Bot: a toi de jouer", t.replace(MARQUEUR, "").strip()[-400:], prio)

sys.exit(0)