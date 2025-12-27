import re

_SAFE_NAME_RE = re.compile(r"[^a-zA-Z0-9._-]+")

def safe_filename(name: str) -> str:
    name = name.strip().replace(" ", "_")
    name = _SAFE_NAME_RE.sub("", name)
    if not name:
        return "upload.csv"
    return name[:120]
