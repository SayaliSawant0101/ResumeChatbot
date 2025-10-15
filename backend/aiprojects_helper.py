# aiprojects_helper.py
# Extracts AI projects from the "AI Projects & Expertise" section in your DOCX
# and answers queries like "AI projects", "AI expertise", "AI project highlights".
#
# UPDATED:
# - Adds an "outcome" field per project (mined from labeled lines & metric cues)
# - Forces response format: Description · Outcome · Technologies used

import re
from typing import List, Dict

# ---------- DOCX loader (same pattern as other helpers) ----------
def load_resume_sections_docx(docx_path: str) -> dict:
    from docx import Document
    doc = Document(docx_path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paras)
    sections, current = {}, None
    # titles you already use across the resume (add more if needed)
    head_pat = re.compile(
        r'^(Personal Details|About Me|Professional Experience|Technical Skills|Projects|Education|AI Projects & Expertise)\s*$',
        re.I
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


# ---------- Simple project parser for the "AI Projects & Expertise" section ----------
TITLE_HINTS = [
    r"AI[-\s]Driven Conversational Resume.*RAG.*Chatbot",
    r"LLM[-\s]Powered Social Listening.*Competitive Intelligence.*Platform",
]

TOOL_PAT = re.compile(
    r'\b('
    r'FastAPI|FAISS|MiniLM|all[-_ ]?MiniLM[-_ ]?L6[-_ ]?v2|sentence[-_ ]?transformers|OpenAI|GPT[- ]?4o[- ]?mini|'
    r'AWS|S3|Glue|Athena|BERTopic|e5[-_ ]?base[-_ ]?v2|bart[-_ ]?large[-_ ]?mnli|'
    r'twitter[-_ ]?roberta[-_ ]?base[-_ ]?sentiment|NLP|RAG|LLM|Transformers'
    r')\b',
    re.I
)

METRIC_PAT = re.compile(
    r'(\b\d+(\.\d+)?%\b|F1|precision|recall|MAE|MAPE|AUC|lift|uplift|ROI|ROAS|CVR|CTR|churn|retention|TAT|time|faster|reduction|increase|decrease|growth)',
    re.I
)

def _extract_tools(text: str) -> List[str]:
    tools = [m.group(0) for m in TOOL_PAT.finditer(text or "")]
    # Normalize a few common variants
    norm = []
    for t in tools:
        tt = re.sub(r'\s+', ' ', t)
        tt = tt.replace('gpt 4o mini', 'gpt-4o-mini').replace('GPT 4o mini', 'gpt-4o-mini')
        tt = tt.replace('all MiniLM L6 v2', 'all-MiniLM-L6-v2')
        tt = tt.replace('miniLM', 'MiniLM')
        norm.append(tt)
    # De-dup case-insensitively, keep stable order
    seen = set(); out = []
    for t in norm:
        key = t.lower()
        if key not in seen:
            seen.add(key); out.append(t)
    return out

def _first_sentence(text: str, fallback_len: int = 200) -> str:
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return (parts[0] if parts and parts[0] else text[:fallback_len]).strip()

def _extract_outcome(text: str) -> str:
    """
    Pull an outcome/result line if present. Priority:
    1) Labeled lines: Outcome/Impact/Results/Benefit
    2) First line containing metric-ish terms
    3) Empty string if none
    """
    # 1) labeled
    m = re.search(r'\b(Outcome|Impact|Results?|Benefit)\s*:\s*(.+)', text, re.I)
    if m:
        return m.group(2).strip()

    # 2) metric-ish
    for ln in [l.strip() for l in text.splitlines() if l.strip()]:
        if METRIC_PAT.search(ln):
            return ln

    return ""

def parse_ai_projects(sections: dict) -> List[Dict]:
    """
    Heuristic parser: splits the AI section into projects using known titles, then
    gathers short description/problem, outcome, and tools.
    """
    ai_text = sections.get("AI Projects & Expertise", "").strip()
    if not ai_text:
        return []

    # Seed candidates by splitting on non-empty lines
    lines = [l for l in ai_text.splitlines() if l.strip() != ""]
    # Identify title lines (either match known titles or look like a Title Case line)
    title_indices = []
    for i, l in enumerate(lines):
        if any(re.search(pat, l, re.I) for pat in TITLE_HINTS):
            title_indices.append(i)
        else:
            # generic fallback: line with many capitals / keywords and no trailing period
            if (len(l) <= 120 and l[-1:] not in ".:;"
                and re.search(r'(AI|LLM|RAG|Platform|Chatbot|Resume|Listening|NLP|GenAI|Framework)', l, re.I)):
                title_indices.append(i)

    title_indices = sorted(set(title_indices))
    if not title_indices:
        # If we can't find distinct titles, treat the entire section as one "project"
        return [{
            "name": "AI Projects & Expertise",
            "desc": _first_sentence(ai_text),
            "problem": "",
            "outcome": _extract_outcome(ai_text),
            "tools": _extract_tools(ai_text),
        }]

    # Slice into blocks per title
    projects = []
    for ix, start in enumerate(title_indices):
        end = title_indices[ix + 1] if ix + 1 < len(title_indices) else len(lines)
        block = lines[start:end]
        title = block[0].strip()
        body = "\n".join(block[1:]).strip()

        # Pull description/problem/outcome
        prob = ""
        desc = ""
        m_goal = re.search(r'\b(Goal|Problem)\s*:\s*(.+)', body, re.I)
        if m_goal:
            prob = m_goal.group(2).strip()
        desc = _first_sentence(body)
        outc = _extract_outcome(body)

        tools = _extract_tools(body + " " + title)

        projects.append({
            "name": title,
            "desc": desc,
            "problem": prob,
            "outcome": outc,
            "tools": tools
        })

    return projects


# ---------- OpenAI chat wrapper (same signature style) ----------
def chat_answer_openai(system_prompt: str, user_prompt: str,
                       model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    from openai import OpenAI
    import os
    client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content


# ---------- Public API ----------
def is_ai_projects_question(q: str) -> bool:
    return bool(re.search(
        r'\b(ai\s*projects?|genai\s*projects?|llm\s*projects?|ai\s*expertise|ai\s*highlights|ai\s*work|rag\s*projects?)\b',
        q or "", re.I
    ))

def build_system_prompt() -> str:
    # Enforce exact format: Description · Outcome · Technologies used
    return (
        "You are a resume Q&A assistant. Use ONLY the provided context.\n"
        "For EACH AI project, output exactly FOUR labeled lines in THIS order:\n"
        "• Project: <project title from context>\n"
        "• Description: <what it is/does>\n"
        "• Outcome: <business/metric impact; use only factual details from context>\n"
        "• Technologies used: <comma-separated tools/models/cloud/services>\n"
        "Separate projects with a blank line. Keep each project compact and recruiter-friendly."
    )

def build_user_prompt(projects: List[Dict], question: str) -> str:
    # Convert parsed projects to a context block
    ctx_lines = []
    for p in projects:
        tools = ", ".join(p.get("tools", [])[:12])
        prob = p.get("problem", "")
        desc = p.get("desc", "")
        outcome = p.get("outcome", "")
        ctx_lines.append(
            "Project: {name}\n"
            "Description: {desc}\n"
            "Problem: {prob}\n"
            "Outcome: {out}\n"
            "Tools: {tools}\n---".format(
                name=p.get("name",""),
                desc=desc,
                prob=prob,
                out=outcome,
                tools=tools
            )
        )
    ctx = "\n".join(ctx_lines)
    # Clear instruction: include the Project title in the returned format
    return (
        f"Question: {question}\n\n"
        f"Context:\n{ctx}\n\n"
        "Return EACH project as exactly:\n"
        "Project: ...\n"
        "Description: ...\n"
        "Outcome: ...\n"
        "Technologies used: ...\n"
        "Use the project titles from the context. Separate projects with one blank line."
    )

def answer_ai_projects(question: str,
                       docx_path: str,
                       chat_model: str = "gpt-4o-mini",
                       temperature: float = 0.2,
                       max_projects: int = 6) -> str:
    sections = load_resume_sections_docx(docx_path)
    projects = parse_ai_projects(sections)[:max_projects]
    if not projects:
        return "Working on AI projects."
    sys = build_system_prompt()
    usr = build_user_prompt(projects, question)
    return chat_answer_openai(sys, usr, model=chat_model, temperature=temperature)
