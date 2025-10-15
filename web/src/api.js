// web/src/api.js

// Prefer FastAPI when provided via .env.local
//   VITE_BACKEND_URL=http://127.0.0.1:8000
// Fallback: Netlify function path (/.netlify/functions/chat)
const BASE = import.meta.env.VITE_BACKEND_URL?.trim() || "";

export async function askRAG(query, k = 5) {
  const url = BASE ? `${BASE}/chat` : "/.netlify/functions/chat";

  const res = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, k }),
  });

  // Helpful error reporting
  if (!res.ok) {
    let details = "";
    try {
      details = await res.text();
    } catch {}
    throw new Error(`API ${res.status}: ${details || "Request failed"}`);
  }

  // Parse JSON (with safety)
  try {
    return await res.json();
  } catch (e) {
    throw new Error(`Invalid JSON from API: ${(e && e.message) || e}`);
  }
}
