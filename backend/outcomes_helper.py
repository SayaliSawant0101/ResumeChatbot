# backend/outcomes_helper.py
"""
Outcomes helper:
- Pulls impact-bearing lines (metrics) from Experience/Projects
- Asks OpenAI for THREE concise mini case-studies:
  Problem • Technical approach • Data • Impact (numbers if present)

NEW:
- Forces coverage to include:
    1) Best overall outcome (same as before)
    2) Go-to-Market (GTM) improvement (if evidence exists; else closest commercial-strategy item)
    3) NPS & analytics framework (if evidence exists; else closest CX/feedback/quality-metric item)
- Adds topic mining and injects GTM/NPS evidence into the prompt context.
- Adds a numbered title line for each case study in the format:
    "1. Project/Task Title"
"""

import re

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


def _lines(text: str):
    return [l.strip() for l in (text or "").splitlines() if l.strip()]


# ---------- outcomes helpers ----------
def is_outcomes_question(q: str) -> bool:
    return bool(
        re.search(
            r"\b(outcomes?|business\s*impact|benefit(ed)?\s+business|results?|impact)\b",
            q or "",
            re.I,
        )
    )


def gather_metric_evidence(sections: dict, max_lines: int = 12):
    """
    Generic impact-bearing lines (numbers, KPIs, speedups, lifts, etc.).
    """
    exp = _lines(sections.get("Professional Experience", ""))
    prj = _lines(sections.get("Projects", ""))
    src = exp + prj
    kw_pat = re.compile(
        r"(\b\d+(\.\d+)?%\b|F1|precision|recall|MAE|MAPE|AUC|lift|uplift|ROI|ROAS|CVR|CTR|churn|retention|penetration|share|time|faster|reduction|increase|decrease|growth|TAM|SAM|SOM|throughput)",
        re.I,
    )
    hits, seen = [], set()
    for l in src:
        if kw_pat.search(l) and l not in seen:
            hits.append(l)
            seen.add(l)
        if len(hits) >= max_lines:
            break
    return hits


# ---------- NEW: topic mining for GTM and NPS/Framework ----------
def gather_topic_evidence(sections: dict, max_lines: int = 8):
    """
    Mine explicit evidence for:
      - GTM/go-to-market/commercial strategy/launch/positioning
      - NPS/analytics framework/measurement program/voice-of-customer
    Returns: dict with 'gtm' and 'nps' lists (unique lines).
    """
    exp = _lines(sections.get("Professional Experience", ""))
    prj = _lines(sections.get("Projects", ""))
    src = exp + prj

    gtm_pat = re.compile(
        r"\b(GTM|go[\s-]?to[\s-]?market|commercial\s+strategy|launch|rollout|positioning|pricing|pack(aging)?|segmentation|targeting|channel|market\s+entry)\b",
        re.I,
    )
    nps_pat = re.compile(
        r"\b(NPS|net\s+promoter|analytics\s+framework|measurement\s+framework|feedback\s+loop|voice[-\s]of[-\s]customer|VoC|CSAT|customer\s+experience|CX)\b",
        re.I,
    )

    def _collect(pat):
        hits, seen = [], set()
        for l in src:
            if pat.search(l) and l not in seen:
                hits.append(l)
                seen.add(l)
            if len(hits) >= max_lines:
                break
        return hits

    return {"gtm": _collect(gtm_pat), "nps": _collect(nps_pat)}


# ---------- OpenAI chat ----------
def _openai_client():
    try:
        from openai import OpenAI
        return OpenAI()
    except TypeError:
        # Some environments require fallback import path handling
        from openai import OpenAI as _OpenAI
        return _OpenAI()


def chat_answer_openai(
    system_prompt: str, user_prompt: str, model: str = "gpt-4o-mini", temperature: float = 0.2
) -> str:
    client = _openai_client()
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt}, {"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


# ---------- prompts + answer ----------
def build_system_prompt_outcomes(topics_present: dict):
    """
    Stricter system prompt that fixes the 3 required slots and
    tells the model when GTM/NPS evidence exists.
    Also requires a numbered title line for each case.
    """
    want_gtm = "Yes" if topics_present.get("gtm") else "No"
    want_nps = "Yes" if topics_present.get("nps") else "No"
    return (
        "You are a resume Q&A assistant. Use ONLY the provided context.\n"
        "Write EXACTLY THREE mini achievements, each 5–8 short lines.\n"
        "For EACH achievement:\n"
        "  • Start with a numbered title line in this exact format: '1. Project/Task Title' (use a concise title from the context such as the project name, initiative, or employer + function).\n"
        "  • Then provide FOUR labeled lines in this order:\n"
        "    • Business problem: <commercial or product context>\n"
        "    • Technical approach: <algorithms/models/libraries/cloud stack/evaluation>\n"
        "    • Data: <sources/features>\n"
        "    • Impact: <numbers if present; do NOT invent metrics>\n"
        "Use industry terms (uplift, propensity, cohort, ABSA, SHAP, RAG, CAC/LTV, TAT). Keep the three achievements DISTINCT (no duplication).\n"
        "Required coverage:\n"
        "  1) Market Share Improvement (best/clearest impact in the context).\n"
        f"  2) Go-to-Market (GTM) improvement — use GTM evidence if available ({want_gtm}); otherwise pick the closest commercial strategy/launch/positioning item.\n"
        f"  3) NPS & analytics framework — use NPS/framework evidence if available ({want_nps}); otherwise pick the closest CX/feedback/quality-metric item."
    )


def build_user_prompt_outcomes(question: str, sections: dict, topics: dict):
    """
    Inject generic impact evidence + GTM/NPS topic evidence into the context.
    Also remind the model to include the numbered title line.
    """
    ctx_parts = []
    if sections.get("Professional Experience"):
        ctx_parts.append("[Professional Experience]\n" + sections["Professional Experience"])
    if sections.get("Projects"):
        ctx_parts.append("[Projects]\n" + sections["Projects"])

    # Generic impact lines (metrics etc.)
    metrics = gather_metric_evidence(sections, max_lines=12)
    if metrics:
        ctx_parts.append("[Impact Evidence]\n" + "\n".join(metrics))

    # Topic-specific evidence
    if topics.get("gtm"):
        ctx_parts.append("[GTM Evidence]\n" + "\n".join(topics["gtm"]))
    if topics.get("nps"):
        ctx_parts.append("[NPS/Framework Evidence]\n" + "\n".join(topics["nps"]))

    context = "\n\n---\n\n".join(ctx_parts)
    return (
        f"Question: {question}\n\n"
        f"Context:\n{context}\n\n"
        "Return THREE case-studies in this order: (1) overall best, (2) GTM, (3) NPS/framework).\n"
        "For EACH case-study, begin with a numbered title line in the format '1. Project/Task Title', then the four labeled lines (Business problem, Technical approach, Data, Impact)."
    )


def answer_outcomes_detailed(
    question: str, docx_path: str, chat_model: str = "gpt-4o-mini", temperature: float = 0.2
):
    """
    High-level function for 'biggest outcomes / business impact':
    - Loads resume sections
    - Mines generic impact lines + GTM/NPS topic evidence
    - Builds stricter prompts to force the 3 slots (overall, GTM, NPS/framework)
    - Calls OpenAI for the final response
    """
    sections = load_resume_sections_docx(docx_path)
    topics = gather_topic_evidence(sections, max_lines=8)
    sys_prompt = build_system_prompt_outcomes(topics)
    user_prompt = build_user_prompt_outcomes(question, sections, topics)
    return chat_answer_openai(sys_prompt, user_prompt, model=chat_model, temperature=temperature)
