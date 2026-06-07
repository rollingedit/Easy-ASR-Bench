from __future__ import annotations

import json


def json_for_script_tag(value: object) -> str:
    """Return JSON safe to embed as text inside an application/json script tag."""
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    return (
        payload.replace("&", "\\u0026")
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("\u2028", "\\u2028")
        .replace("\u2029", "\\u2029")
    )
