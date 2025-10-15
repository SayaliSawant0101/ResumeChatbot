# backend/prepare_docs.py
"""
Build a FAISS index from files in backend/data/docs
Outputs to backend/rag_index/{faiss.index,chunks.json,meta.json}
"""

import json
import pathlib
from datetime import datetime
from typing import List, Dict

# ---------- paths (absolute; safe no matter where you run it) ----------
SCRIPT_DIR = pathlib.Path(__file__).resolve().parent
DOCS_DIR   = (SCRIPT_DIR / "data" / "docs").resolve()
INDEX_DIR  = (SCRIPT_DIR / "rag_index").resolve()
INDEX_DIR.mkdir(parents=True, exist_ok=True)

CHUNKS_JSON = INDEX_DIR / "chunks.json"
META_JSON   = INDEX_DIR / "meta.json"
FAISS_FILE  = INDEX_DIR / "faiss.index"

print(f"🔎 Scanning: {DOCS_DIR}")

# ---------- loaders ----------
try:
    from pypdf import PdfReader
except Exception:
    PdfReader = None

try:
    from docx import Document as DocxDocument
except Exception:
    DocxDocument = None


def load_txt(p: pathlib.Path) -> str:
    return p.read_text(encoding="utf-8", errors="ignore")


def load_pdf(p: pathlib.Path) -> str:
    if PdfReader is None:
        print("⚠️  pypdf not installed; skipping PDFs.")
        return ""
    try:
        r = PdfReader(str(p))
        out = []
        for pg in r.pages:
            t = pg.extract_text() or ""
            if t.strip():
                out.append(t.strip())
        return "\n".join(out)
    except Exception as e:
        print(f"⚠️  PDF read failed for {p.name}: {e}")
        return ""


def load_docx(p: pathlib.Path) -> str:
    if DocxDocument is None:
        print("⚠️  python-docx not installed; skipping DOCX.")
        return ""
    try:
        d = DocxDocument(str(p))
        paras = [x.text.strip() for x in d.paragraphs if x.text and x.text.strip()]
        return "\n".join(paras)
    except Exception as e:
        print(f"⚠️  DOCX read failed for {p.name}: {e}")
        return ""


def load_text(p: pathlib.Path) -> str:
    ext = p.suffix.lower()
    if ext == ".txt":
        return load_txt(p)
    if ext == ".pdf":
        return load_pdf(p)
    if ext == ".docx":
        return load_docx(p)
    print(f"ℹ️  Skipping unsupported type: {p.name}")
    return ""


# ---------- chunking ----------
def split_chunks(text: str, size=1200, overlap=200) -> List[str]:
    """
    Greedy splitter that keeps sentences/words intact when possible.
    """
    text = " ".join(text.split())
    if not text:
        return []
    chunks, start, n = [], 0, len(text)
    while start < n:
        end = min(n, start + size)
        cut = end
        # prefer to cut on sentence / newline / word boundary
        for sep in [". ", "\n", " "]:
            pos = text.rfind(sep, start, end)
            if pos != -1 and pos > start + 200:
                cut = pos + len(sep)
                break
        chunks.append(text[start:cut].strip())
        if cut >= n:
            break
        start = max(0, cut - overlap)
    return [c for c in chunks if c]


# ---------- embeddings + FAISS ----------
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

EMB_MODEL = "sentence-transformers/all-MiniLM-L6-v2"  # 384-dim; keep in sync with app_fastapi.py


def main():
    files = sorted([p for p in DOCS_DIR.glob("*") if p.suffix.lower() in {".txt", ".pdf", ".docx"}])
    print("📁 Files found:", [p.name for p in files])

    all_chunks: List[str] = []
    chunks_meta: List[Dict] = []

    for f in files:
        txt = load_text(f)
        print(f"— {f.name}: extracted {len(txt)} chars")
        if not txt.strip():
            print("   ⚠️  No extractable text.")
            continue
        ch = split_chunks(txt, size=1200, overlap=200)
        print(f"   ➜ {len(ch)} chunks")
        for i, c in enumerate(ch):
            all_chunks.append(c)
            chunks_meta.append({"source": f.name, "chunk_no_in_doc": i, "chars": len(c)})

    if not all_chunks:
        print("❌ No text found in data/docs (PDFs may be scans). Try a .txt/.docx or OCR your PDFs.")
        return

    with open(CHUNKS_JSON, "w", encoding="utf-8") as f:
        json.dump(all_chunks, f, ensure_ascii=False, indent=2)
    with open(META_JSON, "w", encoding="utf-8") as f:
        json.dump(
            {
                "meta": {
                    "created_at": datetime.utcnow().isoformat() + "Z",
                    "num_chunks": len(all_chunks),
                    "model": EMB_MODEL,
                    "files": sorted({m["source"] for m in chunks_meta}),
                },
                "chunks_meta": chunks_meta,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"✅ Saved: {CHUNKS_JSON.name}, {META_JSON.name}")

    print("🧠 Embedding…")
    model = SentenceTransformer(EMB_MODEL)
    X = (
        model.encode(all_chunks, convert_to_numpy=True, normalize_embeddings=True)
        .astype(np.float32)
    )
    print(f"   shape={X.shape} (dim={X.shape[1]})")
    index = faiss.IndexFlatIP(X.shape[1])  # cosine via dot on normalized vectors
    index.add(X)
    faiss.write_index(index, str(FAISS_FILE))
    print(f"✅ FAISS written: {FAISS_FILE}")
    print(f"🎉 Ready. Total chunks: {len(all_chunks)}")


if __name__ == "__main__":
    main()
