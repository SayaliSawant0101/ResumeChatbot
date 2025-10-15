// src/App.jsx
import StarsNet from "./components/StarsNet";
import Sidebar from "./components/Sidebar";
import Chat from "./components/Chat";

export default function App() {
  return (
    <div className="min-h-screen text-white relative">
      {/* background canvas sits at z-0 */}
      <StarsNet count={140} maxDist={160} speed={0.35} />

      {/* content sits above at z-10 */}
      <div className="relative z-10 mx-auto max-w-6xl px-4 py-8 lg:py-12">
        <div className="grid lg:grid-cols-[20rem_1fr] gap-6">
          <Sidebar />
          <Chat />
        </div>
      </div>
    </div>
  );
}

