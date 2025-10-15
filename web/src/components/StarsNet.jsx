// src/components/StarsNet.jsx
import { useEffect, useRef } from "react";

/**
 * Subtle network of moving dots + linking lines.
 * Robust sizing + higher visibility for dark gradients.
 */
export default function StarsNet({
  count = 95,      // number of points
  maxDist = 160,   // link distance in px
  speed = 0.35,    // point speed
}) {
  const canvasRef = useRef(null);
  const rafRef = useRef(0);
  const ptsRef = useRef([]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext?.("2d");
    if (!ctx) return;

    let w = 1, h = 1, dpr = 1, running = true;

    const measure = () => {
      const rect = canvas.getBoundingClientRect();
      w = Math.max(1, rect.width);
      h = Math.max(1, rect.height);
      dpr = window.devicePixelRatio || 1;
      canvas.width = Math.floor(w * dpr);
      canvas.height = Math.floor(h * dpr);
      ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    };

    const initPoints = () => {
      const n = Math.max(1, count);
      ptsRef.current = Array.from({ length: n }, () => ({
        x: Math.random() * w,
        y: Math.random() * h,
        vx: (Math.random() - 0.5) * speed,
        vy: (Math.random() - 0.5) * speed,
      }));
    };

    const step = () => {
      if (!running) return;

      ctx.clearRect(0, 0, w, h);

      const pts = ptsRef.current;

      // move & bounce
      for (const p of pts) {
        p.x += p.vx;
        p.y += p.vy;
        if (p.x < 0 || p.x > w) p.vx *= -1;
        if (p.y < 0 || p.y > h) p.vy *= -1;
      }

      // links (slightly brighter for visibility)
      for (let i = 0; i < pts.length; i++) {
        const p = pts[i];
        for (let j = i + 1; j < pts.length; j++) {
          const q = pts[j];
          const dx = p.x - q.x, dy = p.y - q.y;
          const dist = Math.hypot(dx, dy);
          if (dist < maxDist) {
            const alpha = 1 - dist / maxDist;
            ctx.strokeStyle = `rgba(200, 220, 255, ${0.25 * alpha})`;
            ctx.lineWidth = 1;
            ctx.beginPath();
            ctx.moveTo(p.x, p.y);
            ctx.lineTo(q.x, q.y);
            ctx.stroke();
          }
        }
      }

      // dots (slightly larger & brighter)
      for (const p of pts) {
        ctx.fillStyle = "rgba(230, 240, 255, 0.9)";
        ctx.beginPath();
        ctx.arc(p.x, p.y, 1.8, 0, Math.PI * 2);
        ctx.fill();
      }

      rafRef.current = requestAnimationFrame(step);
    };

    const onResize = () => { measure(); initPoints(); };
    onResize();
    step();
    window.addEventListener("resize", onResize);
    return () => {
      running = false;
      cancelAnimationFrame(rafRef.current);
      window.removeEventListener("resize", onResize);
    };
  }, [count, maxDist, speed]);

  return (
    <canvas
      ref={canvasRef}
      // bump opacity; keep behind content
      className="pointer-events-none fixed inset-0 z-0 opacity-80"
      style={{ width: "100%", height: "100%" }}
    />
  );
}
