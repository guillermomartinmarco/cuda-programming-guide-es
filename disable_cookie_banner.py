#!/usr/bin/env python3
"""Comenta el script de cookies (OneTrust/cookielaw) en las páginas HTML descargadas.

Sin esto, al abrir la copia local (p. ej. con Live Preview de VS Code) aparece un
cartel de cookies que no se puede cerrar. Es idempotente: si ya está comentado, no lo toca.

Uso:
    python disable_cookie_banner.py [carpeta]    (por defecto: docs.nvidia.com)
"""

import re
import sys
from pathlib import Path

SCRIPT = re.compile(r"<script\b[^>]*cookielaw\.org[^>]*>\s*</script>", re.S)


def is_commented(text, pos):
    before = text[:pos]
    return before.rfind("<!--") > before.rfind("-->")


def disable(text):
    def repl(m):
        if is_commented(m.string, m.start()):
            return m.group(0)
        return f"<!--\n{m.group(0)}\n-->"

    return SCRIPT.sub(repl, text)


def main():
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("docs.nvidia.com")
    if not root.exists():
        sys.exit(f"Error: '{root}' no existe.")

    files = [root] if root.is_file() else sorted(root.rglob("*.html"))
    changed = 0
    for f in files:
        # Bytes en lugar de read_text: así se conservan los saltos de línea originales
        text = f.read_bytes().decode("utf-8")
        new = disable(text)
        if new != text:
            f.write_bytes(new.encode("utf-8"))
            changed += 1
            print(f"comentado: {f}")
    print(f"{changed} de {len(files)} archivos modificados")


if __name__ == "__main__":
    main()
