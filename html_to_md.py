#!/usr/bin/env python3
"""Convierte páginas HTML de la CUDA Programming Guide (Sphinx) a Markdown.

Uso:
    python html_to_md.py <archivo.html> [-o salida.md]
    python html_to_md.py <carpeta> [--force]
"""

import argparse
import re
import sys
from pathlib import Path

from bs4 import BeautifulSoup, NavigableString
from markdownify import MarkdownConverter

LANG_ALIASES = {"c++": "cpp"}

# Tipos de admonition de Sphinx -> alertas de GitHub
ALERTS = {
    "note": "NOTE",
    "hint": "TIP",
    "tip": "TIP",
    "important": "IMPORTANT",
    "warning": "WARNING",
    "attention": "WARNING",
    "caution": "CAUTION",
    "danger": "CAUTION",
    "error": "CAUTION",
}


class Placeholders:
    """Guarda fragmentos que markdownify no debe tocar (fórmulas, tablas HTML)."""

    def __init__(self):
        self.items = []

    def add(self, text):
        self.items.append(text)
        return f"XPLACEHOLDER{len(self.items) - 1}X"

    def restore(self, md):
        # Varias pasadas: una tabla HTML guardada puede contener fórmulas guardadas
        pattern = re.compile(r"XPLACEHOLDER(\d+)X")
        while pattern.search(md):
            md = pattern.sub(lambda m: self.items[int(m.group(1))], md)
        return md


def rewrite_links(article):
    """Los enlaces relativos a otras páginas apuntan al .md en lugar del .html."""
    for a in article.find_all("a", href=True):
        href = a["href"]
        if re.match(r"^[a-z]+:", href):
            continue
        a["href"] = re.sub(r"\.html(?=#|$)", ".md", href)


def convert_code_blocks(soup, article):
    for div in article.select('div[class*="highlight-"]'):
        lang = next(
            (c[len("highlight-"):] for c in div.get("class", []) if c.startswith("highlight-")),
            "",
        )
        pre = div.find("pre")
        if pre is None:
            continue
        new_pre = soup.new_tag("pre")
        new_pre["data-lang"] = LANG_ALIASES.get(lang, lang)
        code = soup.new_tag("code")
        code.string = pre.get_text().rstrip("\n")
        new_pre.append(code)
        div.replace_with(new_pre)

    # <code><span class="pre">a</span> <span class="pre">b</span></code> -> <code>a b</code>
    for code in article.select("code.literal"):
        code.string = code.get_text()


def convert_math(article, ph):
    for el in article.select(".math"):
        tex = el.get_text().strip()
        if el.name == "div":
            tex = re.sub(r"^\\\[|\\\]$", "", tex).strip()
            el.replace_with(NavigableString("\n\n" + ph.add(f"$$\n{tex}\n$$") + "\n\n"))
        else:
            tex = re.sub(r"^\\\(|\\\)$", "", tex).strip()
            el.replace_with(NavigableString(ph.add(f"${tex}$")))


def convert_footnotes(article, ph):
    for ref in article.select("a.footnote-reference"):
        label = ref.get_text(strip=True).strip("[]")
        ref.replace_with(NavigableString(ph.add(f"[^{label}]")))

    for aside in article.select("aside.footnote"):
        span = aside.select_one("span.label")
        label = span.get_text(strip=True).strip("[]") if span else "?"
        if span:
            span.decompose()
        first_p = aside.find("p")
        if first_p:
            first_p.insert(0, NavigableString(ph.add(f"[^{label}]:") + " "))
        aside.unwrap()


def convert_figures(soup, article):
    for fig in article.find_all("figure"):
        img = fig.find("img")
        caption = fig.find("figcaption")
        parts = []
        if img:
            new_img = soup.new_tag("img", src=img.get("src", ""), alt=img.get("alt", ""))
            p = soup.new_tag("p")
            p.append(new_img)
            parts.append(p)
        if caption:
            number = caption.select_one("span.caption-number")
            text = caption.select_one("span.caption-text") or caption
            p = soup.new_tag("p")
            if number:
                strong = soup.new_tag("strong")
                strong.string = number.get_text(strip=True) + "."
                p.append(strong)
                p.append(NavigableString(" "))
            em = soup.new_tag("em")
            for child in list(text.contents):
                em.append(child.extract())
            p.append(em)
            parts.append(p)
        for part in reversed(parts):
            fig.insert_after(part)
        fig.decompose()


def convert_admonitions(soup, article):
    for div in article.select("div.admonition"):
        kind = next((c for c in div.get("class", []) if c in ALERTS), "note")
        title = div.select_one("p.admonition-title")
        if title:
            title.decompose()
        bq = soup.new_tag("blockquote")
        marker = soup.new_tag("p")
        marker.string = f"[!{ALERTS[kind]}]"
        bq.append(marker)
        for child in list(div.contents):
            bq.append(child.extract())
        div.replace_with(bq)


def convert_tabs(soup, article):
    """sphinx-design: cada pestaña pasa a ser un párrafo en negrita + su contenido."""
    for inp in article.select("div.sd-tab-set > input"):
        inp.decompose()
    for label in article.select("label.sd-tab-label"):
        p = soup.new_tag("p")
        strong = soup.new_tag("strong")
        strong.string = " ".join(label.get_text(" ", strip=True).split())
        p.append(strong)
        label.replace_with(p)


def is_complex_table(table):
    if table.find(attrs={"rowspan": True}) or table.find(attrs={"colspan": True}):
        return True
    for cell in table.find_all(["td", "th"]):
        if cell.find(["pre", "ul", "ol", "dl", "table"]) or len(cell.find_all("p")) > 1:
            return True
    return False


def table_to_html(table):
    """Tabla HTML compacta, sin líneas en blanco (cortarían el bloque HTML en GFM)."""
    for el in [table, *table.find_all(True)]:
        el.attrs = {k: v for k, v in el.attrs.items() if k in ("rowspan", "colspan", "href", "src", "alt")}
    for cell in table.find_all(["td", "th"]):
        ps = cell.find_all("p", recursive=False)
        if len(ps) == 1:
            ps[0].unwrap()
    for el in table.find_all(["colgroup"]):
        el.decompose()
    html = str(table)
    lines = []
    for line in html.splitlines():
        if line.strip():
            lines.append(line.rstrip())
        else:
            lines.append("&nbsp;")  # línea vacía dentro de un <pre>
    return "\n".join(lines)


def convert_tables(article, ph):
    for table in article.find_all("table"):
        if table.find_parent("table"):
            continue
        cells = table.find_all(["td", "th"])
        if len(cells) == 1:
            # Tabla usada solo como contenedor (p. ej. de un bloque de código)
            cell = cells[0]
            cell.name = "div"
            table.replace_with(cell.extract())
        elif is_complex_table(table):
            table.replace_with(NavigableString("\n\n" + ph.add(table_to_html(table)) + "\n\n"))
        else:
            for cell in table.find_all(["td", "th"]):
                for p in cell.find_all("p"):
                    p.unwrap()


def html_to_markdown(html):
    soup = BeautifulSoup(html, "html.parser")
    article = soup.select_one("article.bd-article") or soup.find("main") or soup.body
    if article is None:
        raise ValueError("no se encontró el contenido principal")

    for el in article.select("a.headerlink, script, style, button"):
        el.decompose()

    ph = Placeholders()
    rewrite_links(article)
    convert_code_blocks(soup, article)
    convert_math(article, ph)
    convert_footnotes(article, ph)
    convert_figures(soup, article)
    convert_admonitions(soup, article)
    convert_tabs(soup, article)
    convert_tables(article, ph)

    md = MarkdownConverter(
        heading_style="ATX",
        bullets="-",
        code_language_callback=lambda el: el.get("data-lang") or None,
        table_infer_header=True,
        sub_symbol="<sub>",
        sup_symbol="<sup>",
    ).convert_soup(article)

    md = ph.restore(md)
    md = re.sub(r"(\[\^\w+\]:)\s+", r"\1 ", md)
    md = re.sub(r"\s+(\[\^\w+\])(?!:)\s*([.,;:)])", r"\1\2", md)
    md = re.sub(r"[ \t]+$", "", md, flags=re.M)
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip() + "\n"


def convert_file(src, dst, force):
    if dst.exists() and dst.stat().st_size > 1 and not force:
        print(f"omitido (ya existe, usá --force): {dst}")
        return
    md = html_to_markdown(src.read_text(encoding="utf-8"))
    dst.write_text(md, encoding="utf-8", newline="\n")
    print(f"{src} -> {dst}")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("input", type=Path, help="archivo .html o carpeta (se recorre recursivamente)")
    parser.add_argument("-o", "--output", type=Path, help="archivo de salida (solo con un archivo de entrada)")
    parser.add_argument("-f", "--force", action="store_true", help="sobrescribir .md existentes")
    args = parser.parse_args()

    if args.input.is_dir():
        if args.output:
            parser.error("-o solo se puede usar con un archivo de entrada")
        files = sorted(args.input.rglob("*.html"))
        for src in files:
            convert_file(src, src.with_suffix(".md"), args.force)
    elif args.input.is_file():
        convert_file(args.input, args.output or args.input.with_suffix(".md"), args.force or bool(args.output))
    else:
        sys.exit(f"Error: '{args.input}' no existe.")


if __name__ == "__main__":
    main()
