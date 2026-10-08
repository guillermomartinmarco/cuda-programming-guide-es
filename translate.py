#!/usr/bin/env python3
"""Traduce archivos Markdown de la CUDA Programming Guide con Ollama, sección por sección.

Cada encabezado (#, ##, ###, ...) inicia una sección que se traduce en una llamada
independiente. Los bloques de código, tablas HTML y fórmulas en bloque no se envían
al modelo: se reemplazan por marcadores y se restauran tal cual.

Uso:
    python translate.py <archivo.md> [-o salida.md]
    python translate.py <carpeta> [--force]
"""

import argparse
import hashlib
import json
import re
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

DEFAULT_HOST = "http://192.168.0.100:11434"
DEFAULT_MODEL = "cuda-translator"
SUFFIX = "[es]"
CACHE_DIR = Path(__file__).parent / ".translate_cache"
RETRIES = 4

HEADING = re.compile(r"^#{1,6}\s")
FENCE = re.compile(r"^\s*(```|~~~)")
# Con marcadores tipo "@@KEEP0@@" el modelo inventa código en su lugar; un comentario HTML lo respeta
MARKER = "<!-- keep:{} -->"
MARKER_RE = re.compile(r"<!--\s*keep:(\d+)\s*-->")
IMAGE_RE = re.compile(r"!\[([^\]]*)\]\(([^)\s]+)\)")
IMAGE_TOKEN_RE = re.compile(r"!?\[([^\]]*)\]\(\s*I(\d+)\s*\)")  # el "!" es opcional: el modelo a veces lo borra
LINK_RE = re.compile(r"\]\((?!I\d+\))([^)\s]+)\)")  # no captura las fichas de imagen ya puestas
LINK_TOKEN_RE = re.compile(r"\]\(\s*L(\d+)\s*\)")
FOOTNOTE_RE = re.compile(r"\[\^\w+\]")
HEADING_PREFIX_RE = re.compile(r"^#{1,6}[ \t]+[\d.]*", re.M)
# Cambiar este valor invalida la caché (p. ej. cuando cambia la forma de proteger el texto)
CACHE_VERSION = "3"


def split_sections(md):
    """Corta el documento en cada encabezado, ignorando los '#' dentro de bloques de código."""
    sections, current, fence = [], [], None
    for line in md.splitlines(keepends=True):
        m = FENCE.match(line)
        if m:
            if fence is None:
                fence = m.group(1)
            elif m.group(1) == fence:
                fence = None
        elif fence is None and HEADING.match(line) and current:
            sections.append("".join(current))
            current = []
        current.append(line)
    if current:
        sections.append("".join(current))
    return sections


def split_blocks(section):
    """Separa una sección en bloques ('text', ...) y ('keep', ...) que no se traducen."""
    blocks, buf, keep, closer = [], [], [], None

    def flush_text():
        if buf:
            blocks.append(("text", "".join(buf)))
            buf.clear()

    for line in section.splitlines(keepends=True):
        stripped = line.strip()
        if closer is None:
            m = FENCE.match(line)
            if m:
                closer = lambda l, f=m.group(1): l.strip().startswith(f)
            elif stripped.startswith("<table"):
                closer = lambda l: "</table>" in l
            elif stripped == "$$":
                closer = lambda l: l.strip() == "$$"
            else:
                buf.append(line)
                continue
            flush_text()
            keep.append(line)
            if stripped.startswith("<table") and "</table>" in line:
                blocks.append(("keep", "".join(keep)))
                keep.clear()
                closer = None
        else:
            keep.append(line)
            if closer(line):
                blocks.append(("keep", "".join(keep)))
                keep.clear()
                closer = None
    if keep:  # bloque sin cerrar: se conserva igual
        blocks.append(("keep", "".join(keep)))
    flush_text()
    return blocks


class Translator:
    def __init__(self, host, model, timeout):
        self.url = host.rstrip("/") + "/api/generate"
        self.model = model
        self.timeout = timeout

    def __call__(self, text):
        payload = json.dumps({"model": self.model, "prompt": text, "stream": False}).encode()
        for attempt in range(1, RETRIES + 1):
            req = urllib.request.Request(self.url, data=payload, headers={"Content-Type": "application/json"})
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    data = json.load(resp)
                break
            except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
                # Típicamente el runner de Ollama se cayó por falta de memoria; al reintentar lo vuelve a cargar
                if attempt == RETRIES:
                    raise
                wait = 30 * attempt
                print(f"    error de conexión ({e}); reintento {attempt}/{RETRIES - 1} en {wait}s", file=sys.stderr)
                time.sleep(wait)
        if data.get("done_reason") == "length":
            print("    aviso: la respuesta se cortó por límite de contexto (num_ctx)", file=sys.stderr)
        return clean_response(data.get("response", ""))


# El modelo a veces traduce las etiquetas de los avisos de GitHub, que deben quedar en inglés
ALERT_FIXES = {
    "NOTE": "NOTE",
    "TIP": "TIP",
    "WARNING": "WARNING",
    "CAUTION": "CAUTION",
    "NOTA": "NOTE",
    "CONSEJO": "TIP",
    "SUGERENCIA": "TIP",
    "PISTA": "TIP",
    "IMPORTANTE": "IMPORTANT",
    "ADVERTENCIA": "WARNING",
    "AVISO": "WARNING",
    "PRECAUCIÓN": "CAUTION",
    "PRECAUCION": "CAUTION",
}


def clean_response(text):
    text = text.strip("\n")
    # A veces el modelo envuelve todo en ```markdown ... ```
    m = re.fullmatch(r"```(?:markdown|md)?\n(.*)\n```", text.strip(), flags=re.S)
    if m:
        text = m.group(1)
    # Variantes vistas: [!NOTA], [¡NOTA!], [NOTA]
    def fix_alert(m):
        tag = ALERT_FIXES.get(m.group(2).upper())
        return f"{m.group(1)}[!{tag}]" if tag else m.group(0)

    return re.sub(r"^(\s*>\s*)\[[!¡]?\s*(\w+)\s*!?\]\s*$", fix_alert, text, flags=re.M)


def keep_edges(original, translated):
    """Conserva los saltos de línea del principio y del final del texto original."""
    lead = original[: len(original) - len(original.lstrip("\n"))]
    trail = original[len(original.rstrip("\n")):]
    return lead + translated.strip("\n") + trail


def protect_links(text):
    """Reemplaza las rutas de imágenes y enlaces por fichas: el modelo traduce rutas y nombres de archivo."""
    images, links = [], []

    def image(m):
        images.append(m.group(2))
        return f"![{m.group(1)}](I{len(images) - 1})"

    def link(m):
        links.append(m.group(1))
        return f"](L{len(links) - 1})"

    text = IMAGE_RE.sub(image, text)
    return LINK_RE.sub(link, text), images, links


def translate_checked(text, translate):
    """Traduce y valida que no falten marcadores, enlaces, imágenes, notas ni encabezados.

    Devuelve None si algo no cierra.
    """
    protected, images, links = protect_links(text)
    result = translate(protected.strip("\n"))

    image_ids = sorted(int(m.group(2)) for m in IMAGE_TOKEN_RE.finditer(result))
    link_ids = sorted(int(n) for n in LINK_TOKEN_RE.findall(result))
    if (
        image_ids != list(range(len(images)))
        or link_ids != list(range(len(links)))
        or MARKER_RE.findall(result) != MARKER_RE.findall(protected)
        or sorted(FOOTNOTE_RE.findall(result)) != sorted(FOOTNOTE_RE.findall(text))
        or [h.strip() for h in HEADING_PREFIX_RE.findall(result)]
        != [h.strip() for h in HEADING_PREFIX_RE.findall(text)]
    ):
        return None
    result = IMAGE_TOKEN_RE.sub(lambda m: f"![{m.group(1)}]({images[int(m.group(2))]})", result)
    return LINK_TOKEN_RE.sub(lambda m: f"]({links[int(m.group(1))]})", result)


def translate_text_block(body, translate):
    """Plan B: un bloque de texto entero y, si falla, párrafo por párrafo."""
    result = translate_checked(body, translate)
    if result is not None:
        return keep_edges(body, result)

    out = []
    for part in re.split(r"(\n[ \t]*\n)", body):
        if not part.strip():
            out.append(part)
            continue
        result = translate_checked(part, translate)
        if result is None:
            first_line = part.strip().splitlines()[0][:60]
            print(f"    !! párrafo dejado en inglés (el modelo alteraba enlaces o notas): {first_line}",
                  file=sys.stderr)
            out.append(part)
        else:
            out.append(keep_edges(part, result))
    return "".join(out)


def translate_section(section, translate):
    blocks = split_blocks(section)
    text_idx = [i for i, (kind, body) in enumerate(blocks) if kind == "text" and body.strip()]
    if not text_idx:
        return section

    # Los bloques al principio o al final no se envían: el modelo tiende a perder esos marcadores
    first, last = text_idx[0], text_idx[-1]
    if first > 0 or last < len(blocks) - 1:
        head = "".join(body for _, body in blocks[:first])
        tail = "".join(body for _, body in blocks[last + 1:])
        middle = "".join(body for _, body in blocks[first:last + 1])
        return head + translate_section(middle, translate) + tail

    kept = [body for kind, body in blocks if kind == "keep"]
    parts, i = [], 0
    for kind, body in blocks:
        if kind == "keep":
            parts.append(MARKER.format(i) + "\n")
            i += 1
        else:
            parts.append(body)

    result = translate_checked("".join(parts), translate)

    if result is None:
        print("    la traducción perdió marcadores, enlaces o notas; traduciendo por partes", file=sys.stderr)
        return "".join(
            body if kind == "keep" or not body.strip() else translate_text_block(body, translate)
            for kind, body in blocks
        )

    if kept:
        # Se reemplaza la línea completa: el bloque guardado ya trae su propia sangría
        result = re.sub(r"^[ \t]*" + MARKER_RE.pattern + r"[ \t]*$",
                        lambda m: kept[int(m.group(1))].rstrip("\n"), result, flags=re.M)
        result = MARKER_RE.sub(lambda m: kept[int(m.group(1))].rstrip("\n"), result)

    return keep_edges(section, result)


def load_cache(path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def translate_file(src, dst, translate, use_cache, model):
    sections = split_sections(src.read_text(encoding="utf-8"))
    cache_path = CACHE_DIR / (hashlib.sha1(str(src.resolve()).encode()).hexdigest() + ".json")
    cache = load_cache(cache_path) if use_cache else {}
    CACHE_DIR.mkdir(exist_ok=True)

    print(f"{src} -> {dst} ({len(sections)} secciones)")
    out = []
    for n, section in enumerate(sections, 1):
        key = hashlib.sha1((CACHE_VERSION + "\0" + model + "\0" + section).encode()).hexdigest()
        title = section.lstrip().splitlines()[0][:70] if section.strip() else ""
        if key in cache:
            print(f"  [{n}/{len(sections)}] (caché) {title}")
            out.append(cache[key])
            continue
        print(f"  [{n}/{len(sections)}] {title}", flush=True)
        start = time.time()
        translated = translate_section(section, translate)
        print(f"    {time.time() - start:.1f}s")
        cache[key] = translated
        out.append(translated)
        cache_path.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")

    dst.write_text("".join(out), encoding="utf-8", newline="\n")


def output_for(src):
    return src.with_name(f"{src.stem}{SUFFIX}{src.suffix}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="archivo .md o carpeta (se recorre recursivamente)")
    parser.add_argument("-o", "--output", type=Path, help=f"archivo de salida (por defecto: nombre{SUFFIX}.md)")
    parser.add_argument("-f", "--force", action="store_true", help="sobrescribir traducciones existentes")
    parser.add_argument("--no-cache", action="store_true", help="ignorar las secciones ya traducidas en caché")
    parser.add_argument("--host", default=DEFAULT_HOST, help=f"servidor Ollama (por defecto: {DEFAULT_HOST})")
    parser.add_argument("--model", default=DEFAULT_MODEL, help=f"modelo (por defecto: {DEFAULT_MODEL})")
    parser.add_argument("--timeout", type=int, default=900, help="segundos máximos por llamada")
    args = parser.parse_args()

    translate = Translator(args.host, args.model, args.timeout)

    if args.input.is_dir():
        if args.output:
            parser.error("-o solo se puede usar con un archivo de entrada")
        jobs = [(f, output_for(f)) for f in sorted(args.input.rglob("*.md")) if not f.stem.endswith(SUFFIX)]
    elif args.input.is_file():
        jobs = [(args.input, args.output or output_for(args.input))]
    else:
        sys.exit(f"Error: '{args.input}' no existe.")

    for src, dst in jobs:
        if dst.exists() and dst.stat().st_size > 1 and not args.force:
            print(f"omitido (ya existe, usá --force): {dst}")
            continue
        try:
            translate_file(src, dst, translate, not args.no_cache, args.model)
        except (urllib.error.URLError, ConnectionError, TimeoutError) as e:
            sys.exit(f"Error al conectar con Ollama en {args.host}: {e}\n"
                     "Las secciones ya traducidas quedaron en caché: volvé a ejecutar el mismo comando para seguir.")


if __name__ == "__main__":
    main()
