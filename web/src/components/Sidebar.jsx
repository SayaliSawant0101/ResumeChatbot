// src/components/Sidebar.jsx
import { Mail, Phone, Github, Linkedin, Globe } from "lucide-react";

export default function Sidebar() {
  return (
    <aside className="w-full lg:w-80 shrink-0 space-y-4">
      {/* Profile card */}
      <div className="card">
  {/* Header: avatar + name/tagline side-by-side */}
  <div className="flex items-center gap-4">
    <div className="w-14 h-14 rounded-full bg-gradient-to-br from-blue-400 to-emerald-500 grid place-items-center text-white text-2xl font-semibold">
      SS
    </div>

    <div className="min-w-0">
      <h2 className="text-xl font-semibold leading-tight truncate text-emerald-400">Sayali Sawant</h2>
      <p className="text-sm text-white/70 truncate">Data Science x AI/LLMs </p>
    </div>
  </div>

  {/* Contact links */}
  <div className="mt-4 space-y-2 text-sm">
    <a className="flex items-center gap-2 text-white/80 hover:text-white" href="mailto:sayalis2024@gmail.com">
      <Mail size={16}/> sayalis2024@gmail.com
    </a>
    <a className="flex items-center gap-2 text-white/80 hover:text-white" href="tel:+19736875648">
      <Phone size={16}/> +1 (973) 687-5648
    </a>
    <a className="flex items-center gap-2 text-white/80 hover:text-white" href="https://github.com/SayaliSawant0101" target="_blank" rel="noreferrer">
      <Github size={16}/> GitHub
    </a>
    <a className="flex items-center gap-2 text-white/80 hover:text-white" href="https://www.linkedin.com/in/sayalisawant01/" target="_blank" rel="noreferrer">
      <Linkedin size={16}/> LinkedIn
    </a>
     <a className="flex items-center gap-2 text-white/80 hover:text-white" href="https://sayalis.netlify.app/" target="_blank" rel="noreferrer">
      <Globe size={16}/> SayaliS.org
    </a>
  </div>
</div>


      {/* About this chatbot card */}


<div className="card">
  <h3 className="text-[16px] font-semibold mb-5 text-emerald-300">
    The Stack: From Frontend to LLM
  </h3>

  <dl className="grid grid-cols-[80px_1fr] gap-x-3 gap-y-1 text-[12px] leading-5 text-white/85">
    <dt className="text-white/60 text-right">Frontend</dt>
    <dd>Vite + React · Tailwind</dd>

    <dt className="text-white/60 text-right">Backend</dt>
    <dd>Python 3.9 · FastAPI + Uvicorn</dd>

    <dt className="text-white/60 text-right">Retrieval</dt>
    <dd>FAISS · all-MiniLM-L6-v2 (384-dim, cosine)</dd>

    <dt className="text-white/60 text-right">LLM</dt>
    <dd>OpenAI gpt-4o-mini </dd>

    <dt className="text-white/60 text-right">Ingestion</dt>
    <dd>PDF/DOCX/TXT via pypdf, python-docx; smart chunking</dd>

    <dt className="text-white/60 text-right">Deps</dt>
    <dd>fastapi, uvicorn, faiss-cpu, sentence-transformers, transformers, torch, accelerate, numpy, openai, python-dotenv</dd>
  </dl>
<p>.</p>

</div>


  
    </aside>
  );
}
