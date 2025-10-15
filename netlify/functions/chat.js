export async function handler(event) {
  if (event.httpMethod !== "POST") return { statusCode: 405, body: "Use POST" };
  const { query } = JSON.parse(event.body || "{}");
  return {
    statusCode: 200,
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ answer: `👋 Hi! You asked: ${query || "(empty)"}`, sources: [] })
  };
}
