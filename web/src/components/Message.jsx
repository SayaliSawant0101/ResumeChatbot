// src/components/Message.jsx
import { useState } from "react";

export default function Message({ role = "assistant", text }) {
  const isUser = role === "user";
  const [copied, setCopied] = useState(false);
  const [feedback, setFeedback] = useState(null); // "up" | "down" | null

  const copy = async () => {
    try {
      await navigator.clipboard.writeText(text || "");
      setCopied(true);
      setTimeout(() => setCopied(false), 1200);
    } catch {}
  };

  return (
    <div className={`flex ${isUser ? "justify-end" : "justify-start"}`}>
      <div
        className={`group relative max-w-[85%] md:max-w-[70%] px-4 py-3 rounded-2xl whitespace-pre-wrap leading-relaxed
          ${isUser ? "bg-emerald-400 text-black" : "glass text-white"}
          animate-[fadeIn_200ms_ease-out] will-change-transform`}
      >
        {text}

        {/* Toolbar only for assistant */}
        {!isUser && (
          <div className="mt-2 flex items-center gap-2 text-xs text-white/80 opacity-0 group-hover:opacity-100 transition">
            <button
              onClick={copy}
              className="px-2 py-1 rounded bg-white/5 border border-white/10 hover:bg-white/10"
              title="Copy message"
              aria-label="Copy message"
            >
              {copied ? "Copied!" : "Copy"}
            </button>
            <button
              onClick={() => setFeedback("up")}
              className={`px-2 py-1 rounded bg-white/5 border border-white/10 hover:bg-white/10 ${feedback === "up" ? "bg-white/10" : ""}`}
              title="Helpful"
              aria-label="Thumbs up"
            >
              👍
            </button>
            <button
              onClick={() => setFeedback("down")}
              className={`px-2 py-1 rounded bg-white/5 border border-white/10 hover:bg-white/10 ${feedback === "down" ? "bg-white/10" : ""}`}
              title="Not helpful"
              aria-label="Thumbs down"
            >
              👎
            </button>
          </div>
        )}
      </div>
    </div>
  );
}
