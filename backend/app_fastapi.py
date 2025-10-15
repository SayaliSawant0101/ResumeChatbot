# backend/app_fastapi.py
import os, json, pathlib, re, traceback
from typing import Optional, List, Dict

from dotenv import load_dotenv
BASE_DIR = pathlib.Path(__file__).resolve().parent
load_dotenv(dotenv_path=str(BASE_DIR / ".env"))  # load backend/.env explicitly

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI

# ---- Your helpers (unchanged) ----
from workexp_helper import answer_work_experience, is_workexp_question
from tech_helper import answer_technical_expertise, is_tech_question
from outcomes_helper import answer_outcomes_detailed, is_outcomes_question
from aiprojects_helper import is_ai_projects_question, answer_ai_projects


# ================= Config =================
INDEX_DIR   = BASE_DIR / "rag_index"
DATA_DIR    = BASE_DIR / "data" / "docs"

EMBED_ID    = "sentence-transformers/all-MiniLM-L6-v2"  # must match prepare_docs.py
FAISS_FILE  = INDEX_DIR / "faiss.index"
CHUNKS_JSON = INDEX_DIR / "chunks.json"
META_JSON   = INDEX_DIR / "meta.json"

DOCX_PATH   = str(DATA_DIR / "Resume_RAG_Optimized_Sayali_Sawant.docx")

# ================= Load Index =================
print("[RAG] Loading FAISS/chunks…")
chunks = json.loads(CHUNKS_JSON.read_text(encoding="utf-8"))
index = faiss.read_index(str(FAISS_FILE))
meta_obj = json.loads(META_JSON.read_text(encoding="utf-8"))
chunks_meta = meta_obj.get("chunks_meta", [])
sources = [m.get("source", "") for m in chunks_meta] if len(chunks_meta) == len(chunks) else [""] * len(chunks)

# ================= Embedder =================
print("[RAG] Loading embedder:", EMBED_ID)
embedder = SentenceTransformer(EMBED_ID)
emb_dim = embedder.get_sentence_embedding_dimension()
assert emb_dim == index.d, f"Embed dim {emb_dim} != FAISS dim {index.d}. Rebuild index with {EMBED_ID}."

# ================= OpenAI client =================
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY missing. Put it in backend/.env or your hosting env.")
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)   # 1.x
except TypeError:
    client = OpenAI(apiKey=OPENAI_API_KEY)    # 2.x

# ================= Utils =================
GREETING_RE = re.compile(r"^\s*(hi|hello|hey)\b", re.I)

def _postclean(text: str) -> str:
    return re.sub(r"(?is)\b(Sources?|Context|Question|Instructions)\s*:.*", "", text or "").strip()

def generate_answer(question: str, context: str) -> str:
    prompt = f"""
You are a helpful professional chatbot that answers conversationally based on the provided context.

Context:
{context}

Task:
- Give a concise and natural answer (2–4 sentences).
- Do NOT repeat large resume text blocks or list sources.
- If the user only greets (e.g., "hi", "hello"), reply warmly and briefly.
- If the context doesn't contain the answer, say you don't have that info.

User: {question}
Assistant:
""".strip()
    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=250,
        )
        return _postclean(resp.choices[0].message.content.strip())
    except Exception as e:
        return f"Error calling OpenAI API: {e}"

def search(query: str, k: int = 8):
    q = embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    D, I = index.search(q, k)
    results = []
    for rank, idx in enumerate(I[0]):
        if 0 <= idx < len(chunks):
            results.append(
                {
                    "text": chunks[idx],
                    "source": sources[idx] if idx < len(sources) else "",
                    "score": float(D[0][rank]),
                }
            )
    return results

# ================= FastAPI =================
app = FastAPI(title="Sayali RAG API")

# CORS (allow Netlify preview + localhost; restrict later to your domains)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ChatIn(BaseModel):
    query: str
    k: Optional[int] = 5

class ChatOut(BaseModel):
    answer: str
    sources: List[Dict] = []

@app.post("/chat", response_model=ChatOut)
def chat(body: ChatIn):
    q = (body.query or "").strip()
    if not q:
        return ChatOut(answer="Please enter a question.", sources=[])

    try:
        if GREETING_RE.match(q):
            return ChatOut(answer="Hi! How can I help you with my projects, skills, or experience?", sources=[])

        # your three special routes (unchanged logic)
        if is_outcomes_question(q):
            return ChatOut(answer=answer_outcomes_detailed(q, DOCX_PATH), sources=[])
        if is_tech_question(q):
            return ChatOut(answer=answer_technical_expertise(q, DOCX_PATH), sources=[])
        if is_workexp_question(q):
            return ChatOut(answer=answer_work_experience(q, DOCX_PATH), sources=[])
        if is_ai_projects_question(q):
            return ChatOut(answer=answer_ai_projects(q, DOCX_PATH), sources=[])


        # default RAG over FAISS chunks
        hits = search(q, k=min(max(int(body.k or 5), 1), 10))
        if not hits:
            return ChatOut(
                answer="I couldn't find that in my documents, but I’m happy to clarify if you share more details.",
                sources=[],
            )

        context = "\n\n---\n\n".join(h["text"][:600] for h in hits[:3])
        ans = generate_answer(q, context).strip()

        simple_sources = [
            {"title": h.get("source") or "chunk", "url": "", "score": h.get("score", 0.0)}
            for h in hits[:3]
        ]

        return ChatOut(answer=ans or "Sorry, I couldn't produce an answer.", sources=simple_sources)
    except Exception as e:
        traceback.print_exc()
        return ChatOut(answer=f"Sorry—answering failed: {e}", sources=[])

@app.get("/")
def root():
    return {"ok": True, "message": "Sayali RAG API running"}
