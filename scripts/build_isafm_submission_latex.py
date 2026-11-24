from __future__ import annotations

import re
import sys
from pathlib import Path

from lxml import html
from PIL import Image


ROOT = Path(sys.argv[1]).resolve()
PACKAGE = (
    Path(sys.argv[2]).resolve()
    if len(sys.argv) > 2
    else ROOT / "submission" / "isafm_major_revision_2026-11-21" / "04_latex_source"
)
SOURCE_FIGURES = ROOT / "submission" / "isafm_major_revision_2026-11-21" / "03_figure_files"
REVIEW_FIGURES = PACKAGE / "figures_review"

FIG_MAP = {
    "fig-ga": "graphical_abstract_800dpi.png",
    "fig1": "class_imbalance_distribution_800dpi.png",
    "fig2": "prespecified_contrasts_forest_800dpi.png",
    "fig3": "frequency_tier_f1_recall_800dpi.png",
    "fig4": "accuracy_efficiency_tradeoff_800dpi.png",
    "fig5": "exploratory_lexical_masking_800dpi.png",
}

FIG_LABELS = {
    "fig-ga": "fig:graphical-abstract",
    "fig1": "fig:class-imbalance",
    "fig2": "fig:prespecified-contrasts",
    "fig3": "fig:frequency-tier",
    "fig4": "fig:accuracy-efficiency",
    "fig5": "fig:lexical-masking",
}

WIDE_TABLE_PANELS = {
    4: [
        ("Panel A: condition definition and primary evaluation means", [0, 1, 2, 3, 4, 5]),
        ("Panel B: seed stability and post-hoc sensitivity", [0, 6, 7, 8, 9]),
    ],
    7: [
        ("Panel A: Macro-F1 by training-frequency tier", [0, 1, 3, 5]),
        ("Panel B: Macro-Recall by training-frequency tier", [0, 2, 4, 6]),
    ],
    9: [
        ("Panel A: parameter and memory profile", [0, 1, 2, 4]),
        ("Panel B: training and inference profile", [0, 3, 5, 6]),
    ],
    14: [
        ("Official run matrix (full Run IDs in the frozen supplementary manifest)", [0, 1, 2, 3, 4, 5, 6]),
    ],
    15: [
        ("Panel A: phase inputs and test status", [0, 1, 4]),
        ("Panel B: execution source and expected outputs", [0, 2, 3]),
    ],
}

APPENDIX_TABLE_TITLES = {
    10: "Taxonomy definitions",
    11: "Frozen model and training settings",
    14: "Official 24-run matrix",
    15: "Evaluation provenance and rerun map",
}

REPLACEMENTS = {
    "−": r"$-$",
    "–": "--",
    "—": "---",
    "“": "``",
    "”": "''",
    "‘": "`",
    "’": "'",
    "…": r"\ldots{}",
    "≥": r"$\geq$",
    "≤": r"$\leq$",
    "≈": r"$\approx$",
    "±": r"$\pm$",
    "×": r"$\times$",
    "→": r"$\rightarrow$",
    "Σ": r"$\Sigma$",
    "Δ": r"$\Delta$",
    "·": r"\textperiodcentered{}",
    " ": " ",
}


def latex_escape(value: str) -> str:
    value = re.sub(r"\s+", " ", value or "")
    for old, new in REPLACEMENTS.items():
        value = value.replace(old, new)
    escaped = []
    for char in value:
        escaped.append(
            {
                "\\": r"\textbackslash{}",
                "&": r"\&",
                "%": r"\%",
                "$": r"\$",
                "#": r"\#",
                "_": r"\_",
                "{": r"\{",
                "}": r"\}",
                "~": r"\textasciitilde{}",
                "^": r"\textasciicircum{}",
            }.get(char, char)
        )
    result = "".join(escaped)
    # Restore macros inserted above after escaping their control characters.
    result = result.replace(r"\textbackslash{}ldots\{\}", r"\ldots{}")
    result = result.replace(r"\textbackslash{}textperiodcentered\{\}", r"\textperiodcentered{}")
    result = result.replace(r"\$-\$", r"$-$")
    for macro in (r"\geq", r"\leq", r"\approx", r"\pm", r"\times", r"\rightarrow", r"\Sigma", r"\Delta"):
        mangled = r"\$\textbackslash{}" + macro[1:] + r"\$"
        result = result.replace(mangled, "$" + macro + "$")
    return result


def inline(element) -> str:
    pieces = []
    if element.text:
        pieces.append(latex_escape(element.text))
    for child in element:
        tag = child.tag.lower() if isinstance(child.tag, str) else ""
        body = inline(child)
        if tag in {"strong", "b"}:
            body = r"\textbf{" + body + "}"
        elif tag in {"em", "i"}:
            body = r"\emph{" + body + "}"
        elif tag == "code":
            body = body.replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}").replace("|", r"|\allowbreak{}").replace("=", r"=\allowbreak{}")
            body = r"\texttt{" + body + "}"
        elif tag == "sup":
            body = r"\textsuperscript{" + body + "}"
        elif tag == "sub":
            body = r"\textsubscript{" + body + "}"
        elif tag == "a":
            href = child.get("href") or ""
            if href:
                if href.startswith(("http://", "https://")):
                    body = r"\url{" + href.replace("%", r"\%") + "}"
                else:
                    body = r"\href{" + latex_escape(href) + "}{" + body + "}"
        elif tag == "br":
            body = r"\\"
        # Make spaces at font/style boundaries explicit.  Besides avoiding
        # TeX command-boundary surprises, this preserves word separation when
        # readers copy text from the compiled PDF.
        if pieces and pieces[-1].endswith(" ") and tag in {"strong", "b", "em", "i", "code"}:
            pieces[-1] = pieces[-1][:-1] + r"\ "
        pieces.append(body)
        if child.tail:
            tail = latex_escape(child.tail)
            if tail.startswith(" ") and tag in {"strong", "b", "em", "i", "code"}:
                tail = r"\ " + tail[1:]
            pieces.append(tail)
    return "".join(pieces).strip()


def table_latex(table, index: int) -> str:
    rows = table.xpath("./thead/tr|./tbody/tr|./tr")
    if not rows:
        return ""
    ncols = max(sum(int(c.get("colspan") or 1) for c in row.xpath("./th|./td")) for row in rows)
    caption_el = table.find("caption")
    caption_plain = (
        re.sub(r"\s+", " ", " ".join(caption_el.itertext())).strip()
        if caption_el is not None
        else APPENDIX_TABLE_TITLES.get(index, f"Appendix table {index}")
    )
    label_match = re.search(r"Table\s+([A-Z]?\d+)", caption_plain)
    label = f"tab:{label_match.group(1).lower()}" if label_match else f"tab:appendix-{index}"
    caption_plain = re.sub(r"^Table\s+[A-Z]?\d+[.:]\s*", "", caption_plain)
    caption = latex_escape(caption_plain)
    if index == 1:
        widths = [0.15, 0.28, 0.28, 0.21]
        spec = " ".join(
            [rf">{{\raggedright\arraybackslash}}p{{{column_width:.2f}\linewidth}}" for column_width in widths]
        )

        def compact_row(row):
            cells = row.xpath("./th|./td")
            rendered = [
                inline(cell).replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
                for cell in cells
            ]
            return " & ".join(rendered) + r" \\"

        body = [
            r"\begin{table}[H]",
            r"\centering",
            r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            r"\footnotesize",
            rf"\caption{{{caption}}}\label{{{label}}}",
            rf"\begin{{tabular}}{{{spec}}}",
            r"\toprule",
            compact_row(rows[0]),
            r"\midrule",
        ]
        body.extend(compact_row(row) for row in rows[1:])
        body.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(body)

    if index in WIDE_TABLE_PANELS:
        panel_blocks = []
        for panel_index, (panel_title, selected) in enumerate(WIDE_TABLE_PANELS[index]):
            panel_ncols = len(selected)
            panel_width = {2: 0.47, 3: 0.31, 4: 0.23, 5: 0.18, 6: 0.145}.get(
                panel_ncols, max(0.110, 0.89 / panel_ncols)
            )
            panel_spec = " ".join(
                [rf">{{\raggedright\arraybackslash}}p{{{panel_width:.3f}\linewidth}}" for _ in selected]
            )

            def selected_row(row):
                cells = row.xpath("./th|./td")
                rendered = []
                for position in selected:
                    text = inline(cells[position])
                    rendered.append(text.replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}"))
                return " & ".join(rendered) + r" \\"

            block = [
                r"\begingroup",
                r"\setlength{\tabcolsep}{3pt}",
                r"\renewcommand{\arraystretch}{1.12}",
                r"\footnotesize",
            ]
            if panel_index > 0:
                # Give a captionless continuation panel a locally unique PDF
                # destination. The surrounding group restores the visible
                # table counter, so subsequent table numbering is unchanged.
                block.append(r"\addtocounter{table}{100}")
            block.append(rf"\begin{{longtable}}{{{panel_spec}}}")
            if panel_index == 0:
                block.append(rf"\caption{{{caption}}}\label{{{label}}}\\")
            else:
                block.append(rf"\multicolumn{{{panel_ncols}}}{{l}}{{\textit{{Table~\ref{{{label}}} continued}}}}\\")
            block.extend([
                rf"\multicolumn{{{panel_ncols}}}{{l}}{{\textbf{{{latex_escape(panel_title)}}}}}\\",
                r"\toprule",
                selected_row(rows[0]),
                r"\midrule",
                r"\endfirsthead",
                rf"\multicolumn{{{panel_ncols}}}{{l}}{{\textit{{Table~\ref{{{label}}} continued---{latex_escape(panel_title)}}}}}\\",
                r"\toprule",
                selected_row(rows[0]),
                r"\midrule",
                r"\endhead",
            ])
            block.extend(selected_row(row) for row in rows[1:])
            block.extend([r"\bottomrule", r"\end{longtable}"])
            if panel_index > 0:
                # Undo the +100 offset and longtable's automatic +1.
                block.append(r"\addtocounter{table}{-101}")
            block.append(r"\endgroup")
            panel_blocks.append("\n".join(block))
        return "\n\n".join(panel_blocks)

    width = {2: 0.47, 3: 0.31, 4: 0.23, 5: 0.18, 6: 0.145}.get(ncols, max(0.075, 0.92 / ncols))
    spec = " ".join([rf">{{\raggedright\arraybackslash}}p{{{width:.3f}\linewidth}}" for _ in range(ncols)])
    landscape = ncols >= 7
    out = []
    if landscape:
        out.append(r"\begin{landscape}")
    out.extend([
        r"\begingroup",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\renewcommand{\arraystretch}{1.12}",
        r"\footnotesize",
        rf"\begin{{longtable}}{{{spec}}}",
        rf"\caption{{{caption}}}\label{{{label}}}\\",
        r"\toprule",
    ])
    for row_index, row in enumerate(rows):
        rendered = []
        for cell in row.xpath("./th|./td"):
            text = inline(cell)
            text = text.replace("/", r"/\allowbreak{}").replace(r"\_", r"\_\allowbreak{}")
            span = int(cell.get("colspan") or 1)
            if span > 1:
                rendered.append(rf"\multicolumn{{{span}}}{{l}}{{{text}}}")
            else:
                rendered.append(text)
        out.append(" & ".join(rendered) + r" \\")
        if row_index == 0:
            out.extend([r"\midrule", r"\endfirsthead", rf"\multicolumn{{{ncols}}}{{l}}{{\textit{{{caption} (continued)}}}}\\", r"\toprule", " & ".join(inline(c) for c in row.xpath("./th|./td")) + r" \\", r"\midrule", r"\endhead"])
    out.extend([r"\bottomrule", r"\end{longtable}", r"\endgroup"])
    if landscape:
        out.append(r"\end{landscape}")
    return "\n".join(out)


def render_element(element, table_counter: list[int]) -> str:
    tag = element.tag.lower() if isinstance(element.tag, str) else ""
    classes = set((element.get("class") or "").split())
    if tag in {"style", "script", "nav", "svg", "title", "desc", "defs"} or "toc" in classes:
        return ""
    if tag == "h1" or "review-blind" in classes:
        return ""
    if tag == "h2":
        return "\n" + r"\section*{\raggedright " + inline(element) + "}\n"
    if tag == "h3":
        heading = inline(element)
        if heading.startswith(("Appendix B.", "Appendix E.")):
            keep_with_next = r"\clearpage" + "\n"
        else:
            keep_with_next = r"\filbreak" + "\n" if heading.startswith("Appendix ") else ""
        return "\n" + keep_with_next + r"\subsection*{" + heading + "}\n"
    if tag == "h4":
        return "\n" + r"\subsubsection*{" + inline(element) + "}\n"
    if tag == "p":
        if "short-title" in classes:
            return ""
        if "eq" in classes:
            return r"\[\mathrm{Macro}\mbox{-}\mathrm{F1}=\frac{1}{12}\sum_{k=1}^{12}F1_k.\]" + "\n"
        text = inline(element)
        if "caption" in classes:
            return ""
        if "note" in classes:
            return r"\begin{quote}\small " + text + r"\end{quote}" + "\n"
        if "ref" in classes:
            return r"\noindent\hangindent=1.5em\hangafter=1 " + text + r"\par" + "\n"
        return text + "\n\n"
    if tag == "ul":
        items = [r"\item " + inline(li) for li in element.xpath("./li")]
        return r"\begin{itemize}" + "\n" + "\n".join(items) + "\n" + r"\end{itemize}" + "\n"
    if tag == "table":
        table_counter[0] += 1
        return table_latex(element, table_counter[0]) + "\n"
    if tag == "div" and "figure" in classes:
        fid = element.get("id")
        if fid == "fig-ga":
            # The graphical abstract is delivered as a separate submission figure.
            return ""
        filename = FIG_MAP.get(fid)
        caption_el = next((c for c in element if isinstance(c.tag, str) and c.tag.lower() == "p"), None)
        caption_plain = re.sub(r"\s+", " ", " ".join(caption_el.itertext())).strip() if caption_el is not None else ""
        if fid != "fig-ga":
            caption_plain = re.sub(r"^Figure\s+\d+[.:]\s*", "", caption_plain)
        caption = latex_escape(caption_plain)
        if not filename:
            return ""
        label = FIG_LABELS[fid]
        review_filename = filename.replace("_800dpi.png", "_review.jpg")
        placement = "H"
        width = "0.90"
        max_height = "0.38"
        return "\n".join([
            rf"\begin{{figure}}[{placement}]",
            r"\centering",
            rf"\includegraphics[width={width}\textwidth,height={max_height}\textheight,keepaspectratio]{{figures_review/{review_filename}}}",
            (r"\par\small\textit{" if fid == "fig-ga" else r"\caption{") + caption + "}",
            rf"\label{{{label}}}",
            r"\end{figure}",
            "",
        ])
    if tag == "section" and "summary" in classes:
        paragraph = element.find("p")
        return r"\begin{abstract}" + "\n" + inline(paragraph) + "\n" + r"\end{abstract}" + "\n"
    if tag == "section" and "pagebreak" in classes:
        children = "\n".join(render_element(c, table_counter) for c in element)
        return r"\clearpage" + "\n" + children
    if tag == "div" and "first-page-note" in classes:
        body = "\n".join(render_element(c, table_counter) for c in element)
        return r"\begin{quote}\small" + "\n" + body + r"\end{quote}" + "\n"
    if tag == "div" and ("takeaway" in classes or "author-placeholder" in classes):
        body = "\n".join(render_element(c, table_counter) for c in element)
        return r"\begin{quote}" + "\n" + body + r"\end{quote}" + "\n"
    return "\n".join(render_element(child, table_counter) for child in element)


PREAMBLE = r"""\documentclass[12pt]{article}
\PassOptionsToPackage{hyphens}{url}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{longtable}
\usepackage{array}
\usepackage{float}
\usepackage{hyperref}
\usepackage{xcolor}
\hypersetup{hidelinks,pdfauthor={},pdftitle={Evaluating Adaptation for Imbalanced PCAOB Audit-Finding Classification}}
\linespread{1.3}\selectfont
\setlength{\parindent}{1.5em}
\setlength{\parskip}{0pt}
\setlength{\emergencystretch}{3em}
\pagestyle{plain}
\title{Evaluating Adaptation for Imbalanced PCAOB Audit-Finding Classification:\\Accuracy, Efficiency, and Exploratory Lexical Masking}
\author{Anonymous manuscript prepared for double-blind review}
\date{}
\begin{document}
\maketitle
"""

TITLE_PAGE = r"""\documentclass[12pt]{article}
\PassOptionsToPackage{hyphens}{url}
\usepackage[a4paper,margin=1in]{geometry}
\usepackage[utf8]{inputenc}
\usepackage{hyperref}
\hypersetup{hidelinks,pdfauthor={}}
\linespread{1.3}\selectfont
\title{Evaluating Adaptation for Imbalanced PCAOB Audit-Finding Classification:\\Accuracy, Efficiency, and Exploratory Lexical Masking}
\author{}
\date{}
\begin{document}
\maketitle
\noindent\textbf{Short title:} Domain Adaptation for PCAOB Finding Classification\\
\textbf{Authors:} [Full names in publication order]\\
\textbf{Affiliations:} [Department, institution, city, country for each author]\\
\textbf{Corresponding author:} [Name, postal address, email, telephone, and ORCID]\\
\textbf{Co-author contacts:} [Email address and ORCID for every co-author]

\section*{Declarations}
\noindent\textbf{Funding:} [Sponsor names and grant numbers, or an author-approved no-funding statement]

\noindent\textbf{Conflict of interest:} [Author-approved disclosure]

\noindent\textbf{Data availability:} [Repository URL, archival DOI, access conditions, and license]

\noindent\textbf{AI use:} [Tool name, version, dates, exact uses, and author verification statement]

\noindent\textbf{Acknowledgements:} [Editor-only author-approved text; omit if none]

\noindent\textbf{Permissions:} Original figures were generated from project results; confirm that no third-party permissions are required.
\end{document}
"""

README = """# LaTeX submission source

This folder is the LaTeX counterpart of the revised anonymous DOCX manuscript.

Files:

- `main_anonymous.tex`: anonymous manuscript, including tables, appendices, and references.
- `title_page.tex`: separate editor-only title-page template; complete all bracketed fields.
- `figures_review/`: compact JPEG copies embedded in the review PDF.
- `../03_figure_files/`: the canonical opaque-white 800 dpi RGB TIFF and editable white-background SVG figures for separate journal upload. Keep this sibling directory with the LaTeX folder.
- `main_anonymous.pdf`: compiled review PDF when compilation succeeds.
- `supplementary_tables.tex`: generated seed-level and frequency-tier supplementary tables.
- `supplementary_tables.pdf`: compiled supplementary-table PDF when compilation succeeds.
- `title_page.pdf`: compiled title-page preview when compilation succeeds.

Compile with:

```text
pdflatex -interaction=nonstopmode -halt-on-error main_anonymous.tex
pdflatex -interaction=nonstopmode -halt-on-error main_anonymous.tex
pdflatex -interaction=nonstopmode -halt-on-error title_page.tex
```

The source intentionally uses the standard `article` class because ISAFM accepts an editable free-format initial manuscript and does not publish a dedicated public LaTeX class for this submission stage. The manuscript is anonymous; the title page must be uploaded separately for double-blind review.

Before submission, visually inspect every compiled page, complete the title-page declarations, and upload the separate figure files if requested by the portal.
"""

GRAPHICAL_OVERVIEW = r"""
\begin{center}
\includegraphics[width=0.82\textwidth,height=0.24\textheight,keepaspectratio]{figures_review/graphical_abstract_review.jpg}
\par\small\textit{Study workflow and evidential sequence. FT = full fine-tuning. Source: Phase 1 protocol, frozen Phase 5 matrix, Phase 6 retrospective evaluation, and post-hoc integrity audit.}
\end{center}
"""


def build():
    PACKAGE.mkdir(parents=True, exist_ok=True)
    REVIEW_FIGURES.mkdir(parents=True, exist_ok=True)
    for filename in FIG_MAP.values():
        source = (SOURCE_FIGURES / filename).with_suffix(".tiff")
        target = REVIEW_FIGURES / filename.replace("_800dpi.png", "_review.jpg")
        with Image.open(source) as image:
            image.load()
            if image.mode in {"RGBA", "LA"} or "transparency" in image.info:
                rgba = image.convert("RGBA")
                flattened = Image.new("RGB", rgba.size, "white")
                flattened.paste(rgba, mask=rgba.getchannel("A"))
                image = flattened
            else:
                image = image.convert("RGB")
            image.thumbnail((2200, 2200))
            image.save(target, "JPEG", quality=88, optimize=True, progressive=True)
    tree = html.parse(str(ROOT / "PAPER.html"))
    main = tree.getroot().xpath("//main")[0]
    table_counter = [0]
    body = "\n".join(render_element(child, table_counter) for child in main)
    literature_marker = r"\section*{\raggedright 2. Literature Review and Research Positioning}"
    if literature_marker not in body:
        raise RuntimeError("Could not locate the Literature Review heading for graphical-overview insertion")
    body = body.replace(literature_marker, GRAPHICAL_OVERVIEW + "\n" + literature_marker, 1)
    source = PREAMBLE + body + "\n\\end{document}\n"
    (PACKAGE / "main_anonymous.tex").write_text(source, encoding="utf-8")
    (PACKAGE / "title_page.tex").write_text(TITLE_PAGE, encoding="utf-8")
    (PACKAGE / "README.md").write_text(README, encoding="utf-8")
    print(PACKAGE / "main_anonymous.tex")
    print(PACKAGE / "title_page.tex")


if __name__ == "__main__":
    build()

