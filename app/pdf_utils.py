import re
from datetime import datetime
from typing import Dict, Optional

from fpdf import FPDF


_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.*)$")
_BULLET_RE = re.compile(r"^\s*[-*+]\s+(.*)$")
_NUMBERED_RE = re.compile(r"^\s*(\d+)[.)]\s+(.*)$")
_LINK_RE = re.compile(r"\[([^\]]+)\]\(([^)]+)\)")


def _latin1_safe(text: str) -> str:
    return str(text or "").encode("latin-1", "replace").decode("latin-1")


def _clean_inline_markdown(text: str) -> str:
    cleaned = str(text or "")
    cleaned = _LINK_RE.sub(r"\1 (\2)", cleaned)
    cleaned = cleaned.replace("**", "").replace("__", "")
    cleaned = cleaned.replace("`", "")
    cleaned = re.sub(r"(?<!\w)[*_](.+?)[*_](?!\w)", r"\1", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return _latin1_safe(cleaned)


def _is_list_or_heading(line: str) -> bool:
    return bool(_HEADING_RE.match(line) or _BULLET_RE.match(line) or _NUMBERED_RE.match(line))

def _normalize_heading_text(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(text or "").lower())

def _clean_heading_candidate(line: str) -> str:
    cleaned = str(line or "").strip()
    cleaned = re.sub(r"^#+\s*", "", cleaned)
    cleaned = re.sub(r"^\d+[.)]\s*", "", cleaned)
    cleaned = cleaned.strip("`*_ ").rstrip(":").strip()
    return cleaned

def _strip_redundant_section_heading(section_title: str, section_body: str) -> str:
    text = str(section_body or "").replace("\r\n", "\n").replace("\r", "\n")
    lines = text.split("\n")
    first_idx = None
    for idx, line in enumerate(lines):
        if line.strip():
            first_idx = idx
            break
    if first_idx is None:
        return ""

    title_norm = _normalize_heading_text(section_title)
    first_line_clean = _clean_heading_candidate(lines[first_idx])
    first_line_norm = _normalize_heading_text(first_line_clean)
    heading_with_optional_body = re.match(
        rf"^\s*(?:#+\s*)?(?:\d+[.)]\s*)?{re.escape(section_title)}\s*[:\-]?\s*(.*)$",
        lines[first_idx],
        flags=re.IGNORECASE,
    )

    if heading_with_optional_body:
        remainder = heading_with_optional_body.group(1).strip("`*_ ").strip()
        if remainder:
            lines[first_idx] = remainder
            return "\n".join(lines).strip()
        lines.pop(first_idx)
        while first_idx < len(lines) and not lines[first_idx].strip():
            lines.pop(first_idx)
        return "\n".join(lines).strip()

    if first_line_norm and first_line_norm == title_norm:
        lines.pop(first_idx)
        while first_idx < len(lines) and not lines[first_idx].strip():
            lines.pop(first_idx)

    return "\n".join(lines).strip()


class ProposalPDF(FPDF):
    def __init__(self, title: str):
        super().__init__()
        self.doc_title = title
        self.set_margins(15, 15, 15)
        self.set_auto_page_break(auto=True, margin=15)

    def header(self):
        if self.page_no() == 1:
            return
        self.set_font("Times", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, _latin1_safe(self.doc_title), 0, 1, "R")
        self.ln(2)

    def footer(self):
        self.set_y(-12)
        self.set_font("Times", "I", 8)
        self.set_text_color(120, 120, 120)
        self.cell(0, 6, f"Page {self.page_no()}", 0, 0, "C")


def _write_title_block(pdf: ProposalPDF, date_str: str, title: str):
    pdf.set_text_color(20, 20, 20)
    pdf.set_font("Times", "B", 18)
    pdf.multi_cell(0, 10, _latin1_safe(title), align="C")
    pdf.ln(2)
    pdf.set_font("Times", "", 11)
    pdf.multi_cell(0, 7, _latin1_safe(f"Date: {date_str}"), align="C")
    pdf.ln(4)
    y = pdf.get_y()
    pdf.set_draw_color(190, 190, 190)
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(6)


def _write_section_heading(pdf: ProposalPDF, title: str):
    pdf.set_text_color(15, 15, 15)
    pdf.set_font("Times", "B", 13)
    pdf.multi_cell(0, 8, _latin1_safe(title))
    y = pdf.get_y()
    pdf.set_draw_color(210, 210, 210)
    pdf.line(pdf.l_margin, y, pdf.w - pdf.r_margin, y)
    pdf.ln(3)


def _write_subheading(pdf: ProposalPDF, text: str):
    heading = _clean_inline_markdown(text)
    if not heading:
        return
    pdf.set_text_color(25, 25, 25)
    pdf.set_font("Times", "B", 11)
    pdf.multi_cell(0, 7, heading)
    pdf.ln(1)


def _write_paragraph(pdf: ProposalPDF, text: str):
    paragraph = _clean_inline_markdown(text)
    if not paragraph:
        return
    pdf.set_text_color(25, 25, 25)
    pdf.set_font("Times", "", 11)
    pdf.multi_cell(0, 7, paragraph)
    pdf.ln(1)


def _write_list_item(pdf: ProposalPDF, prefix: str, text: str):
    item = _clean_inline_markdown(text)
    if not item:
        return

    pdf.set_text_color(25, 25, 25)
    pdf.set_font("Times", "", 11)
    indent_x = pdf.l_margin + 6
    width = pdf.w - pdf.r_margin - indent_x
    pdf.set_x(indent_x)
    pdf.multi_cell(width, 6, _latin1_safe(f"{prefix} {item}"))
    pdf.ln(1)


def _render_markdown_like_block(pdf: ProposalPDF, block: str):
    lines = [line.rstrip() for line in str(block or "").split("\n") if line.strip()]
    if not lines:
        return

    if len(lines) == 1:
        heading_match = _HEADING_RE.match(lines[0])
        if heading_match:
            _write_subheading(pdf, heading_match.group(1))
            return

    i = 0
    while i < len(lines):
        line = lines[i]

        heading_match = _HEADING_RE.match(line)
        if heading_match:
            _write_subheading(pdf, heading_match.group(1))
            i += 1
            continue

        bullet_match = _BULLET_RE.match(line)
        numbered_match = _NUMBERED_RE.match(line)
        if bullet_match or numbered_match:
            prefix = "-" if bullet_match else f"{numbered_match.group(1)}."
            content = bullet_match.group(1) if bullet_match else numbered_match.group(2)
            j = i + 1
            continuation_parts = []
            while j < len(lines):
                nxt = lines[j]
                if _is_list_or_heading(nxt):
                    break
                continuation_parts.append(nxt.strip())
                j += 1
            item_text = " ".join([content.strip()] + continuation_parts).strip()
            _write_list_item(pdf, prefix, item_text)
            i = j
            continue

        j = i + 1
        paragraph_parts = [line.strip()]
        while j < len(lines):
            nxt = lines[j]
            if _is_list_or_heading(nxt):
                break
            paragraph_parts.append(nxt.strip())
            j += 1
        _write_paragraph(pdf, " ".join(paragraph_parts))
        i = j


def render_proposal_pdf(
    proposal_sections: Dict[str, str],
    output_path: str = "Grant_Proposal_Submission.pdf",
    title: str = "Grant Proposal Submission",
    date_str: Optional[str] = None,
):
    resolved_date = date_str or datetime.now().strftime("%B %d, %Y")
    pdf = ProposalPDF(title=title)
    pdf.set_title(_latin1_safe(title))
    pdf.set_author("Granted Agent")
    pdf.add_page()

    _write_title_block(pdf, resolved_date, title)

    for section_title, section_body in proposal_sections.items():
        _write_section_heading(pdf, section_title)
        cleaned_body = _strip_redundant_section_heading(section_title, str(section_body or ""))
        blocks = re.split(r"\n\s*\n", cleaned_body)
        if not any(block.strip() for block in blocks):
            _write_paragraph(pdf, "No content generated for this section.")
        else:
            for block in blocks:
                if not block.strip():
                    continue
                _render_markdown_like_block(pdf, block)
        pdf.ln(6)

    pdf.output(output_path)


def build_proposal_plain_text(
    proposal_sections: Dict[str, str],
    date_str: Optional[str] = None,
    title: str = "Grant Proposal Submission",
) -> str:
    resolved_date = date_str or datetime.now().strftime("%B %d, %Y")
    lines = [title, f"Date: {resolved_date}", "", "=" * 70]
    for section_title, section_body in proposal_sections.items():
        lines.extend(
            [
                "",
                "",
                section_title.upper(),
                "-" * len(section_title),
                "",
                str(section_body or "").strip(),
                "",
                "=" * 70,
            ]
        )
    return "\n".join(lines).strip()
