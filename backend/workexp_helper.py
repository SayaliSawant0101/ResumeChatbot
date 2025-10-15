# backend/workexp_helper.py
"""
Work-experience helper:
- Loads sections from your resume DOCX
- Retrieves most relevant chunks using OpenAI embeddings
- Builds a targeted prompt and gets an answer from OpenAI
(No total-years calc; shows dates per role.)
"""

import re
import numpy as np

# ---------- DOCX → sections ----------
def load_resume_sections_docx(docx_path: str) -> dict:
    from docx import Document
    doc = Document(docx_path)
    paras = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
    text = "\n".join(paras)
    sections, current = {}, None
    head_pat = re.compile(
        r'^(Personal Details|About Me|Professional Experience|Technical Skills|Projects|Education)\s*$',
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

# ---------- section → chunks ----------
def make_chunks(sections: dict):
    return [{
        "id": name.lower().replace(" ", "_"),
        "section": name,
        "text": body,
        "boost": 1.6 if "experience" in name.lower() else 1.0
    } for name, body in sections.items()]

# ---------- OpenAI embeddings (for retrieval) ----------
def embed_texts_openai(texts, model: str = "text-embedding-3-large"):
    # openai>=1.x
    from openai import OpenAI
    client = OpenAI()  # reads OPENAI_API_KEY from env/.env
    resp = client.embeddings.create(model=model, input=texts)
    # return (n, d) float32
    return np.array([d.embedding for d in resp.data], dtype=np.float32)

def build_index(chunks, embed_model="text-embedding-3-large"):
    return embed_texts_openai([c["text"] for c in chunks], model=embed_model)

def search(query: str, chunks, vecs, embed_model="text-embedding-3-large", k: int = 8):
    qv = embed_texts_openai([query], model=embed_model)[0]
    # cosine similarity
    sims = (vecs @ qv) / ((np.linalg.norm(vecs, axis=1) * np.linalg.norm(qv)) + 1e-9)
    sims = np.array([s * c["boost"] for s, c in zip(sims, chunks)])
    idx = sims.argsort()[::-1][:k]
    return [chunks[i] for i in idx]

# ---------- simple extractors ----------
DATE_PAT = r'([A-Za-z]{3,9}\s?\d{4}|\b\d{4}\b)'
RANGE_PAT = re.compile(rf'(?P<start>{DATE_PAT})\s*[–-]\s*(?P<end>{DATE_PAT}|Present)', re.I)

def extract_roles(experience_text: str):
    """
    Parse 'Professional Experience' into roles with:
      - header (often 'Title — Company')
      - dates (line like 'Jun 2021 – Present')
      - bullets (lines starting with '-')
    """
    lines = [l for l in (experience_text or "").splitlines() if l.strip()]
    roles, cur = [], {"header": "", "dates": "", "bullets": []}
    for ln in lines:
        if "—" in ln or "--" in ln:
            if cur["header"]:
                roles.append(cur)
                cur = {"header": "", "dates": "", "bullets": []}
            cur["header"] = ln.strip()
        elif re.search(RANGE_PAT, ln):
            cur["dates"] = ln.strip()
        elif ln.startswith("-"):
            cur["bullets"].append(ln.lstrip("- ").strip())
    if cur["header"]:
        roles.append(cur)
    return roles

def extract_tech(sections: dict):
    """
    Rough technology list from Technical Skills + Experience (deduped).
    """
    tech = []
    tech_text = sections.get("Technical Skills", "")
    tech += re.findall(r':\s*(.*)', tech_text)  # capture lists after labels like "Programming: ..."
    exp_text = sections.get("Professional Experience", "")
    tools = re.findall(
        r'(Python|SQL|Tableau|Power BI|XGBoost|LightGBM|Prophet|AWS|Athena|Glue|pandas|scikit-learn|transformers|NLP|LLM|RoBERTa|GoEmotions|K-?Means|DBSCAN|SHAP)',
        exp_text, flags=re.I
    )
    if tools:
        tech += tools
    import re as _re
    flat = _re.sub(r'\s*,\s*', ', ', ", ".join(tech))
    dedup = sorted(set([t.strip() for t in _re.split(r',|/|;|\u2022', flat) if t.strip()]), key=lambda s: s.lower())
    return dedup

def role_dates_summary(experience_text: str) -> str:
    """
    Compact list of 'Role — Company : Start – End' lines from Experience.
    """
    lines = []
    for r in extract_roles(experience_text):
        header = r.get("header", "").strip()
        dates = r.get("dates", "").strip()
        if header and dates:
            lines.append(f"- {header} : {dates}")
        elif header:
            lines.append(f"- {header}")
    return "\n".join(lines)

# ---------- OpenAI chat ----------
def chat_answer_openai(system_prompt: str, user_prompt: str,
                       model: str = "gpt-4o-mini", temperature: float = 0.2) -> str:
    from openai import OpenAI
    client = OpenAI()  # uses env key
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "system", "content": system_prompt},
                  {"role": "user", "content": user_prompt}],
        temperature=temperature,
    )
    return resp.choices[0].message.content

# ---------- prompts + answer (dates per position, no total years) ----------
def build_system_prompt_workexp(tech_list: list) -> str:
    tools = ", ".join(tech_list[:30]) if tech_list else ""
    return (
        "You are a resume Q&A assistant. Use only the provided context.\n"
        "For any 'work experience' answer:\n"
        "1) Include ALL organizations (e.g., AGR Knowledge Services Pvt Ltd, WhiteHat Jr., MRVC).\n"
        "2) For EACH position, show: Role/Company (Start – End).\n"
        f"3) List key technologies & tools (deduplicated): {tools}.\n"
        "4) For each organization, give 1–2 concrete impact statements (metrics if present).\n"
        "5) Keep to 5–10 lines; crisp, factual tone."
    )

def build_user_prompt_workexp(question: str, retrieved_chunks: list, roles_dates_block: str) -> str:
    ctx = []
    ctx.append("\n\n---\n\n".join([f"[{c['section']}]\n{c['text']}" for c in retrieved_chunks]))
    if roles_dates_block:
        ctx.append("[Role Date Ranges]\n" + roles_dates_block)
    context = "\n\n---\n\n".join([c for c in ctx if c])
    return f"Question: {question}\n\nContext:\n{context}\n\nAnswer:"

def answer_work_experience(question: str,
                           docx_path: str,
                           embed_model: str = "text-embedding-3-large",
                           chat_model: str = "gpt-4o-mini",
                           k: int = 8,
                           temperature: float = 0.2) -> str:
    """
    Returns a concise work-experience summary using:
      - DOCX sections
      - OpenAI embeddings for retrieval
      - OpenAI chat for answer generation
    """
    sections = load_resume_sections_docx(docx_path)
    chunks = make_chunks(sections)
    vecs = build_index(chunks, embed_model=embed_model)
    retrieved = search(question, chunks, vecs, embed_model=embed_model, k=k)

    tech_list = extract_tech(sections)
    roles_dates_block = role_dates_summary(sections.get("Professional Experience", ""))

    sys_prompt = build_system_prompt_workexp(tech_list)
    user_prompt = build_user_prompt_workexp(question, retrieved, roles_dates_block)

    return chat_answer_openai(sys_prompt, user_prompt, model=chat_model, temperature=temperature)

def is_workexp_question(q: str) -> bool:
    return bool(re.search(r'\b(work|professional)\s+experience\b', q or "", re.I))
