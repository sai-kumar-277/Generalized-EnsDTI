"""
Build publication-ready paper documents from PAPER_DRAFT.md:
1. PAPER_SUBMISSION_READY.html (Full standalone scholarly HTML with KaTeX and CSS)
2. PAPER_SUBMISSION_READY.pdf (Full multi-page conference-style PDF via ReportLab)
"""

import os
import re
import markdown
from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, HRFlowable, KeepTogether, PageBreak
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY, TA_LEFT, TA_RIGHT
from reportlab.pdfgen import canvas

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DRAFT_MD_PATH = os.path.join(BASE_DIR, "PAPER_DRAFT.md")
HTML_OUT_PATH = os.path.join(BASE_DIR, "PAPER_SUBMISSION_READY.html")
PDF_OUT_PATH = os.path.join(BASE_DIR, "PAPER_SUBMISSION_READY.pdf")


# -------------------------------------------------------------
# 1. Numbered Canvas for PDF (Running Header & Page X of Y)
# -------------------------------------------------------------
class NumberedCanvas(canvas.Canvas):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#4a5568"))

        # Running header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(
                54, 750,
                "EnsDTI: Novelty-Aware MoE Gating & Conformal Calibration Under Entity Shift"
            )
            self.setStrokeColor(colors.HexColor("#cbd5e0"))
            self.setLineWidth(0.5)
            self.line(54, 744, 558, 744)

        # Running footer (all pages)
        footer_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(558, 36, footer_text)
        self.drawString(54, 36, "NeurIPS / ICML / ICLR Main Track Standard — UROP Research Submission")
        self.setStrokeColor(colors.HexColor("#cbd5e0"))
        self.setLineWidth(0.5)
        self.line(54, 46, 558, 46)

        self.restoreState()


# -------------------------------------------------------------
# 2. HTML Generation
# -------------------------------------------------------------
def generate_html(md_text):
    html_body = markdown.markdown(md_text, extensions=['tables', 'fenced_code', 'attr_list', 'def_list'])
    
    # Custom CSS for scholarly presentation
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Novelty-Aware MoE Gating and Conformal Calibration for DTI Prediction Under Entity Shift</title>
    <!-- MathJax for rendering LaTeX -->
    <script>
    MathJax = {{
      tex: {{
        inlineMath: [['$', '$'], ['\\\\(', '\\\\)']],
        displayMath: [['$$', '$$'], ['\\\\[', '\\\\]']]
      }}
    }};
    </script>
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js" id="MathJax-script" async></script>
    <style>
        @page {{
            size: letter;
            margin: 20mm 20mm 20mm 20mm;
        }}
        body {{
            font-family: 'Times New Roman', Times, Georgia, serif;
            font-size: 11pt;
            line-height: 1.5;
            color: #1a202c;
            max-width: 900px;
            margin: 0 auto;
            padding: 30px;
            background-color: #ffffff;
        }}
        h1 {{
            font-size: 18pt;
            font-weight: bold;
            text-align: center;
            margin-top: 10px;
            margin-bottom: 12px;
            color: #0f172a;
            line-height: 1.25;
        }}
        .author-block {{
            text-align: center;
            font-size: 10pt;
            color: #475569;
            margin-bottom: 25px;
            border-bottom: 1px solid #e2e8f0;
            padding-bottom: 15px;
        }}
        .abstract-box {{
            background-color: #f8fafc;
            border-left: 4px solid #2563eb;
            padding: 15px 20px;
            margin: 20px 0;
            border-radius: 4px;
            font-size: 10pt;
            text-align: justify;
        }}
        h2 {{
            font-size: 13pt;
            color: #1e3a8a;
            border-bottom: 1.5px solid #cbd5e1;
            padding-bottom: 4px;
            margin-top: 28px;
            margin-bottom: 10px;
            font-weight: bold;
        }}
        h3 {{
            font-size: 11pt;
            color: #1e40af;
            margin-top: 18px;
            margin-bottom: 6px;
            font-weight: bold;
        }}
        p {{
            margin-top: 6px;
            margin-bottom: 10px;
            text-align: justify;
        }}
        table {{
            width: 100%;
            border-collapse: collapse;
            margin: 20px 0;
            font-size: 9.5pt;
        }}
        th, td {{
            padding: 8px 10px;
            text-align: left;
            border: 1px solid #cbd5e1;
        }}
        th {{
            background-color: #1e3a8a;
            color: #ffffff;
            font-weight: 600;
            text-align: center;
        }}
        tr:nth-child(even) {{
            background-color: #f8fafc;
        }}
        blockquote {{
            border-left: 3px solid #3b82f6;
            margin: 15px 0;
            padding: 8px 15px;
            background-color: #eff6ff;
            color: #1e293b;
        }}
        code {{
            font-family: 'Consolas', 'Courier New', monospace;
            background-color: #f1f5f9;
            padding: 2px 4px;
            border-radius: 3px;
            font-size: 9pt;
        }}
        img {{
            max-width: 100%;
            height: auto;
            display: block;
            margin: 18px auto;
            border: 1px solid #e2e8f0;
            border-radius: 4px;
            box-shadow: 0 1px 3px rgba(0,0,0,0.1);
        }}
        .figure-caption {{
            text-align: center;
            font-size: 9pt;
            color: #64748b;
            font-style: italic;
            margin-top: 6px;
            margin-bottom: 20px;
        }}
        ol, ul {{
            margin-top: 6px;
            margin-bottom: 12px;
            padding-left: 24px;
        }}
        li {{
            margin-bottom: 4px;
        }}
    </style>
</head>
<body>
{html_body}
</body>
</html>
"""
    with open(HTML_OUT_PATH, "w", encoding="utf-8") as f:
        f.write(html_content)
    print(f"[HTML] Successfully generated {HTML_OUT_PATH} ({len(html_content)} bytes)")


# -------------------------------------------------------------
# 3. PDF Generation via ReportLab
# -------------------------------------------------------------
def format_latex_to_reportlab(text):
    """Convert common LaTeX math symbols and markdown bold/italic to ReportLab XML tags."""
    # Replace math terms
    text = text.replace(r"\text{AUROC}", "AUROC")
    text = text.replace(r"\text{AUPRC}", "AUPRC")
    text = text.replace(r"\text{ens}", "ens")
    text = text.replace(r"\text{train}", "train")
    text = text.replace(r"\text{test}", "test")
    text = text.replace(r"\text{reg}", "reg")
    text = text.replace(r"\text{L2}", "L2")
    text = text.replace(r"\sigma", "&sigma;")
    text = text.replace(r"\rho", "&rho;")
    text = text.replace(r"\alpha", "&alpha;")
    text = text.replace(r"\lambda", "&lambda;")
    text = text.replace(r"\Theta", "&Theta;")
    text = text.replace(r"\kappa", "&kappa;")
    text = text.replace(r"\pm", "&plusmn;")
    text = text.replace(r"\approx", "&asymp;")
    text = text.replace(r"\le", "&le;")
    text = text.replace(r"\ge", "&ge;")
    text = text.replace(r"\ll", "&lt;&lt;")
    text = text.replace(r"\gg", "&gt;&gt;")
    text = text.replace(r"\in", "&isin;")
    text = text.replace(r"\notin", "&notin;")
    text = text.replace(r"\cup", "&cup;")
    text = text.replace(r"\cap", "&cap;")
    text = text.replace(r"\emptyset", "&empty;")
    text = text.replace(r"\times", "&times;")
    text = text.replace(r"\to", "&rarr;")
    text = text.replace(r"\blacksquare", "&#9632;")

    # Strip remaining $ delimiters
    text = re.sub(r"\$([^\$]+)\$", r"<i>\1</i>", text)
    text = text.replace("$$", "")

    # Bold & Italic markdown conversion
    text = re.sub(r"\*\*([^\*]+)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"\*([^\*]+)\*", r"<i>\1</i>", text)
    text = re.sub(r"`([^`]+)`", r"<font face='Courier' color='#1a365d'>\1</font>", text)

    return text


def build_pdf_document(md_text):
    doc = SimpleDocTemplate(
        PDF_OUT_PATH,
        pagesize=letter,
        leftMargin=40,
        rightMargin=40,
        topMargin=46,
        bottomMargin=46
    )

    styles = getSampleStyleSheet()

    # Custom styles
    title_style = ParagraphStyle(
        'DocTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=18,
        textColor=colors.HexColor('#0f172a'),
        alignment=TA_CENTER,
        spaceAfter=8
    )

    meta_style = ParagraphStyle(
        'DocMeta',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=8.5,
        leading=12,
        textColor=colors.HexColor('#475569'),
        alignment=TA_CENTER,
        spaceAfter=14
    )

    h1_style = ParagraphStyle(
        'DocH1',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=11.5,
        leading=14,
        textColor=colors.HexColor('#1e3a8a'),
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'DocH2',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=9.5,
        leading=12,
        textColor=colors.HexColor('#2563eb'),
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    h3_style = ParagraphStyle(
        'DocH3',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=8.5,
        leading=11,
        textColor=colors.HexColor('#1e40af'),
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'DocBody',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11.5,
        textColor=colors.HexColor('#1e293b'),
        alignment=TA_JUSTIFY,
        spaceAfter=5
    )

    abstract_style = ParagraphStyle(
        'DocAbstract',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.0,
        leading=10.8,
        textColor=colors.HexColor('#0f172a'),
        alignment=TA_JUSTIFY
    )

    callout_style = ParagraphStyle(
        'DocCallout',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=8.0,
        leading=10.5,
        textColor=colors.HexColor('#1e3a8a')
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.HexColor('#0f172a')
    )

    table_hdr_style = ParagraphStyle(
        'TableHdr',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=7.2,
        leading=9.2,
        textColor=colors.white,
        alignment=TA_CENTER
    )

    ref_style = ParagraphStyle(
        'DocRef',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=7.5,
        leading=9.8,
        textColor=colors.HexColor('#334155'),
        leftIndent=15,
        firstLineIndent=-15,
        spaceAfter=3
    )

    story = []

    lines = md_text.splitlines()
    in_table = False
    table_lines = []

    for line in lines:
        stripped = line.strip()

        # Handle Table Accumulation
        if stripped.startswith("|") and stripped.endswith("|"):
            in_table = True
            table_lines.append(stripped)
            continue
        elif in_table:
            # End of table, process accumulated table lines
            in_table = False
            # Parse table
            raw_rows = []
            for tl in table_lines:
                # Skip divider lines
                if re.match(r"^\|(\s*:?-+:?\s*\|)+$", tl):
                    continue
                cells = [c.strip() for c in tl.strip("|").split("|")]
                raw_rows.append(cells)

            if raw_rows:
                num_cols = max(len(r) for r in raw_rows)
                # Pad rows to uniform col count
                for r in raw_rows:
                    while len(r) < num_cols:
                        r.append("")

                table_data = []
                for row_idx, r in enumerate(raw_rows):
                    row_data = []
                    for c in r:
                        c_formatted = format_latex_to_reportlab(c)
                        if row_idx == 0:
                            p = Paragraph(c_formatted, table_hdr_style)
                        else:
                            p = Paragraph(c_formatted, table_cell_style)
                        row_data.append(p)
                    table_data.append(row_data)

                # Col widths
                usable_width = 532
                col_w = usable_width / num_cols
                t = Table(table_data, colWidths=[col_w] * num_cols)
                t.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#1e3a8a')),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                    ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#cbd5e1')),
                    ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor('#f8fafc')]),
                    ('TOPPADDING', (0, 0), (-1, -1), 3),
                    ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ]))
                story.append(Spacer(1, 4))
                story.append(t)
                story.append(Spacer(1, 6))

            table_lines = []

        if not stripped:
            continue

        # Title
        if stripped.startswith("# "):
            title_text = stripped[2:].strip()
            story.append(Paragraph(title_text, title_style))
            continue

        # Metadata
        if stripped.startswith("**Authors**:") or stripped.startswith("**Target Venue**:") or stripped.startswith("**Keywords**:"):
            meta_text = format_latex_to_reportlab(stripped)
            story.append(Paragraph(meta_text, meta_style))
            continue

        # Horizontal rule
        if stripped == "---":
            story.append(Spacer(1, 4))
            story.append(HRFlowable(width="100%", thickness=0.8, color=colors.HexColor('#cbd5e1'), spaceAfter=8, spaceBefore=4))
            continue

        # Heading 1
        if stripped.startswith("## "):
            h1_text = stripped[3:].strip()
            story.append(Paragraph(h1_text, h1_style))
            continue

        # Heading 2
        if stripped.startswith("### "):
            h2_text = stripped[4:].strip()
            story.append(Paragraph(h2_text, h2_style))
            continue

        # Heading 3
        if stripped.startswith("#### "):
            h3_text = stripped[5:].strip()
            story.append(Paragraph(h3_text, h3_style))
            continue

        # Images
        img_match = re.search(r"\((output/[^\)]+\.png)\)", stripped)
        if img_match or stripped.startswith("*(Reference Figure:"):
            # Check for embedded figure
            figure_map = {
                "gating_mechanistic_analysis": os.path.join(BASE_DIR, "output", "davis", "gating_mechanistic_analysis.png"),
                "gating_10seed_per_seed": os.path.join(BASE_DIR, "output", "davis", "gating_10seed_per_seed.png"),
                "cold_aware_icp_coverage": os.path.join(BASE_DIR, "output", "davis", "cold_aware_icp_coverage.png"),
                "bindingdb_gating_comparison": os.path.join(BASE_DIR, "output", "bindingdb", "bindingdb_gating_comparison.png")
            }
            matched_img = None
            for key, path in figure_map.items():
                if key in stripped:
                    matched_img = path
                    break
            
            if matched_img and os.path.exists(matched_img):
                story.append(Spacer(1, 4))
                # 532 width, aspect ratio 2.2
                img_flowable = Image(matched_img, width=520, height=210)
                caption = Paragraph(f"<b>Figure</b>: Mechanistic and empirical evaluation ({os.path.basename(matched_img)})", table_cell_style)
                story.append(KeepTogether([img_flowable, Spacer(1, 2), caption]))
                story.append(Spacer(1, 6))
                continue

        # Abstract Box
        if "Abstract" in lines[max(0, lines.index(line)-2):lines.index(line)+1]:
            formatted_p = format_latex_to_reportlab(stripped)
            abstract_table = Table([[Paragraph(f"<b>Abstract</b>—{formatted_p}", abstract_style)]], colWidths=[532])
            abstract_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
                ('BOX', (0, 0), (-1, -1), 0.8, colors.HexColor('#2563eb')),
                ('LEFTPADDING', (0, 0), (-1, -1), 10),
                ('RIGHTPADDING', (0, 0), (-1, -1), 10),
                ('TOPPADDING', (0, 0), (-1, -1), 8),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
            ]))
            story.append(abstract_table)
            story.append(Spacer(1, 6))
            continue

        # Propositions / Theorems Callout Box
        if stripped.startswith("**Proposition") or stripped.startswith("**Theorem"):
            formatted_callout = format_latex_to_reportlab(stripped)
            callout_table = Table([[Paragraph(formatted_callout, callout_style)]], colWidths=[532])
            callout_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f1f5f9')),
                ('LINEBEFORE', (0, 0), (0, -1), 3.0, colors.HexColor('#1e3a8a')),
                ('LEFTPADDING', (0, 0), (-1, -1), 8),
                ('RIGHTPADDING', (0, 0), (-1, -1), 8),
                ('TOPPADDING', (0, 0), (-1, -1), 6),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ]))
            story.append(Spacer(1, 3))
            story.append(callout_table)
            story.append(Spacer(1, 4))
            continue

        # References
        if re.match(r"^\d+\.\s+", stripped):
            formatted_ref = format_latex_to_reportlab(stripped)
            story.append(Paragraph(formatted_ref, ref_style))
            continue

        # Regular Paragraph
        formatted_text = format_latex_to_reportlab(stripped)
        story.append(Paragraph(formatted_text, body_style))

    # Build the document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[PDF] Successfully built {PDF_OUT_PATH} ({os.path.getsize(PDF_OUT_PATH)} bytes)")


def main():
    print("Reading PAPER_DRAFT.md...")
    with open(DRAFT_MD_PATH, "r", encoding="utf-8") as f:
        md_text = f.read()

    print("Generating HTML document...")
    generate_html(md_text)

    print("Generating PDF document...")
    build_pdf_document(md_text)
    print("All documents generated successfully!")


if __name__ == "__main__":
    main()
