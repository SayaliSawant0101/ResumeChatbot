# backend/tech_helper.py
"""
Technical-expertise helper (Technical Skills → all categories/items):
- Reads sections from resume DOCX
- Parses only the "Technical Skills" section
- Uses your own headers/categories exactly as they appear
- Displays *all* items under each header (no 'top 3' cap)
- Falls back to an 'Other' bucket for unlabeled lines
- No evidence lines; no model call needed (deterministic)

Public API/signatures unchanged:
- is_tech_question(q: str) -> bool
- answer_technical_expertise(question, docx_path, chat_model="gpt-4o-mini", temperature=0.2, max_tools=24) -> str
"""

import re
from typing import Dict, List, Tuple, OrderedDict
from collections import OrderedDict as OD

# ---------- DOCX loader ----------
def load_resume_sections_docx(docx_path: str) -> dict:
    from docx import Document
    doc = Document(docx_path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paras)
    sections, current = {}, None
    head_pat = re.compile(
        r'^(Personal Details|About Me|Professional Experience|Technical Skills|Projects|Education)\s*$',
        re.I,
    )
    for line in text.splitlines():
        if head_pat.match(line.strip()):
            current = line.strip()
            sections[current] = []
        elif current:
            sections[current].append(line)
    for k in list(sections.keys()):
        sections[k] = "\n".join(sections[k]).strip()
    return sections

def _lines(text: str) -> List[str]:
    return [l.strip() for l in (text or "").splitlines() if l.strip()]

# ---------- Parse Technical Skills into ordered categories ----------
LABEL_RE = re.compile(r'^\s*([A-Za-z0-9 &/+._\-()]+)\s*:\s*(.+)$')

def _split_items(payload: str) -> List[str]:
    # Split on common separators: comma, slash, semicolon, bullet, pipe
    raw = [p.strip() for p in re.split(r',|/|;|\u2022|\|', payload or '') if p.strip()]
    # Minimal normalization (keep original casing for neat display)
    norm = []
    for t in raw:
        x = re.sub(r'\s+', ' ', t).strip()
        # Light canonicalization of a few common aliases
        x = x.replace('powerbi', 'Power BI').replace('power bi', 'Power BI')
        x = x.replace('scikit learn', 'scikit-learn')
        x = x.replace('xg boost', 'XGBoost')
        x = x.replace('light gbm', 'LightGBM')
        x = x.replace('k means', 'K-Means')
        x = x.replace('gpt 4o mini', 'gpt-4o-mini').replace('GPT 4o mini', 'gpt-4o-mini')
        norm.append(x)
    # de-dup preserving order
    seen, out = set(), []
    for x in norm:
        key = x.lower()
        if key not in seen:
            seen.add(key); out.append(x)
    return out

def parse_technical_skills_structured(sections: dict) -> Tuple["OrderedDict[str, List[str]]", List[str]]:
    """
    Returns:
      ordered_categories: OrderedDict[label -> list of items]
      orphans: items that came from unlabeled lines (if any)
    """
    txt = sections.get("Technical Skills", "") or ""
    if not txt:
        return OD(), []

    ordered_categories: "OrderedDict[str, List[str]]" = OD()
    orphans: List[str] = []

    for line in _lines(txt):
        m = LABEL_RE.match(line)
        if m:
            label = re.sub(r'\s+', ' ', m.group(1).strip())
            payload = m.group(2).strip()
            items = _split_items(payload)
            if not items:
                continue
            if label not in ordered_categories:
                ordered_categories[label] = []
            # append unique items preserving order
            seen = set(x.lower() for x in ordered_categories[label])
            for it in items:
                if it.lower() not in seen:
                    ordered_categories[label].append(it)
                    seen.add(it.lower())
        else:
            # unlabeled line → split into items and treat as orphans
            items = _split_items(line)
            for it in items:
                if it.lower() not in [x.lower() for x in orphans]:
                    orphans.append(it)

    return ordered_categories, orphans

# ---------- Public API (unchanged) ----------
def is_tech_question(q: str) -> bool:
    """
    Detects queries like 'technical expertise', 'tech stack',
    'tools/frameworks/models used', 'AI models', etc.
    """
    return bool(re.search(
        r'\b(tech(nical)?\s*(expertise|skills|stack)|tools?|frameworks?|libraries|models?|algorithms?|ai\s*models?|llms?)\b',
        q or "", re.I
    ))

# We retain these functions for compatibility, but we won't call OpenAI for this task.
def build_system_prompt_tech() -> str:
    return (
        "Output the categories and their items exactly as provided. "
        "No evidence, no definitions, no extra commentary."
    )

def build_user_prompt_tech(question: str, sections: dict, selected_tools: List[str], _unused: Dict[str, List[str]]) -> str:
    # Not used in this deterministic implementation.
    return "N/A"

def answer_technical_expertise(question: str,
                               docx_path: str,
                               chat_model: str = "gpt-4o-mini",
                               temperature: float = 0.2,
                               max_tools: int = 24) -> str:
    """
    Deterministic output:
      - Use categories from 'Technical Skills' headings (Label: items)
      - List ALL items under each label
      - Add an 'Other' section for unlabeled lines, if present
    """
    sections = load_resume_sections_docx(docx_path)
    cats, orphans = parse_technical_skills_structured(sections)

    if not cats and not orphans:
        return "I couldn't find a 'Technical Skills' section."

    # Build plain-text response
    out_lines: List[str] = []
    out_lines.append("Technical Skills:")

    # Categories in original order
    for label, items in cats.items():
        out_lines.append(f"\n{label}")
        for it in items:
            out_lines.append(f"- {it}")

    # Orphans (unlabeled) at the end
    if orphans:
        out_lines.append("\nOther")
        for it in orphans:
            out_lines.append(f"- {it}")

    return "\n".join(out_lines)
