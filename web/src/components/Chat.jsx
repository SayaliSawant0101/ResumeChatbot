  // src/components/Chat.jsx
  import { useEffect, useRef, useState } from "react";
  import { Send, Sparkles } from "lucide-react";
  import { askRAG } from "../api";
  import Message from "./Message";

  const PROMPTS = [
    "Professional experience — quick summary",
    "Top 3 outcomes I delivered",
    "AI projects — highlights",
    "Tools in action — brief",
  ];

  export default function Chat() {
    const [messages, setMessages] = useState([
      { role: "assistant", text: "Hello — feel free to ask about my experience, projects, skills, or business impact." },
    ]);
    const [q, setQ] = useState("");
    const [loading, setLoading] = useState(false);
    const [used, setUsed] = useState(new Set());     // prompt chips “used” state
    const [atBottom, setAtBottom] = useState(true);  // show “jump to bottom” when false

    // --- auto-scroll + track if user is at bottom ---
    const listRef = useRef(null);
    useEffect(() => {
      const el = listRef.current;
      if (!el) return;

      // keep scrolled to bottom when new messages arrive (only if user is already near bottom)
      if (atBottom) el.scrollTop = el.scrollHeight;

      const onScroll = () => {
        const nearBottom = el.scrollHeight - el.scrollTop - el.clientHeight < 40;
        setAtBottom(nearBottom);
      };
      el.addEventListener("scroll", onScroll);
      return () => el.removeEventListener("scroll", onScroll);
    }, [messages, atBottom]);

    const send = async (text) => {
      const query = (text ?? q).trim();
      if (!query || loading) return;

      // mark chip as used if it came from a prompt
      if (text) setUsed(prev => new Set(prev).add(text));

      setMessages(m => [...m, { role: "user", text: query }]);
      setQ("");
      setLoading(true);
      try {
        const { answer } = await askRAG(query, 5);
        setMessages(m => [...m, { role: "assistant", text: answer }]);
      } catch {
        setMessages(m => [...m, { role: "assistant", text: "Sorry, something went wrong." }]);
      } finally {
        setLoading(false);
      }
    };

    return (
      <section className="flex-1 space-y-2">
        <header className="card flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-semibold text-[#22BAEE] drop-shadow-[0_0_14px_#22BAEE]"
  ><b><i>An AI-driven conversational resume</i></b></h1>
            <p className="text-white/70 text-sm">Replaces static docs with fast, context-aware answers while showcasing my end-to-end technical capability.</p>
          </div>
          <Sparkles className="text-emerald-400 animate-pulse" />
        </header>

        {/* FIXED-SIZE messages box */}
        <div
          ref={listRef}
          className="card h-[420px] md:h-[380px] overflow-y-auto space-y-3 relative"
        >
          {messages.map((m, i) => <Message key={i} role={m.role} text={m.text} />)}

          {/* 1) Typing indicator (shows while loading) */}
          {loading && (
            <div className="glass inline-flex items-center gap-1 px-4 py-2 rounded-2xl w-fit">
              <span className="sr-only">Typing…</span>
              <span className="w-2 h-2 bg-white/70 rounded-full animate-bounce [animation-delay:-0.2s]"></span>
              <span className="w-2 h-2 bg-white/70 rounded-full animate-bounce"></span>
              <span className="w-2 h-2 bg-white/70 rounded-full animate-bounce [animation-delay:0.2s]"></span>
            </div>
          )}
        </div>

        {/* 3) Input row */}
        <div className="flex items-center gap-2">
          <input
            className="input flex-1"
            placeholder="Ask about experience, projects, impact, or skills…"
            value={q}
            onChange={(e) => setQ(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && send()}
          />
          <button onClick={() => send()} className="btn-cta flex items-center gap-2" disabled={loading}>
            <Send size={16} /> {loading ? "Thinking…" : "Send"}
          </button>
        </div>

        {/* 4) Quick prompts under the textbox with “used” state */}
        <div className="flex flex-wrap gap-2">
        <div className="text-emerald-100 text-md"><i>Quick prompts: </i></div>
          {PROMPTS.map((p) => {
            const isUsed = used.has(p);
            return (
              <button
                key={p}
                onClick={() => send(p)}
                className={`px-3 py-1 rounded-full border text-sm transition
                  ${isUsed ? "bg-white/5 border-white/10 opacity-60" : "bg-white/5 border-white/10 hover:bg-white/10"}`}
                title={isUsed ? "Already asked" : "Click to ask"}
              >
                {p}
              </button>

              
            );
          })}
          
        </div>



        

        {/* 2) “Jump to bottom” button when scrolled up */}
        {!atBottom && (
          <button
            onClick={() => {
              const el = listRef.current;
              if (el) el.scrollTop = el.scrollHeight;
            }}
            className="fixed right-6 bottom-28 z-20 px-3 py-2 rounded-full bg-emerald-500/90 text-black shadow"
          >
            ↓ New message
          </button>
        )}
      </section>
    );
  }
