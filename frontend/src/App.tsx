import { useCallback, useEffect, useState } from "react";
import { api } from "./api";
import Dashboard from "./pages/Dashboard";
import NewInvestigation from "./pages/NewInvestigation";
import LiveInvestigation from "./pages/LiveInvestigation";
import Findings from "./pages/Findings";
import Reports from "./pages/Reports";

type Page = "dashboard" | "new" | "live" | "findings" | "reports";

const NAV: { key: Page; label: string; icon: string }[] = [
  { key: "dashboard", label: "Dashboard", icon: "◉" },
  { key: "new", label: "New Investigation", icon: "＋" },
  { key: "live", label: "Live Investigation", icon: "▸" },
  { key: "findings", label: "Findings", icon: "⚠" },
  { key: "reports", label: "Reports", icon: "▤" },
];

export default function App() {
  const [page, setPage] = useState<Page>("dashboard");
  const [activeId, setActiveId] = useState<string | null>(null);
  const [activeName, setActiveName] = useState<string>("");
  const [backendOnline, setBackendOnline] = useState<boolean | null>(null);
  const [gemini, setGemini] = useState(false);

  // Backend health / Gemini status
  useEffect(() => {
    let alive = true;
    api
      .health()
      .then((h) => {
        if (!alive) return;
        setBackendOnline(h.status === "ok");
        setGemini(h.gemini_configured);
      })
      .catch(() => alive && setBackendOnline(false));
    return () => {
      alive = false;
    };
  }, []);

  const goNew = useCallback((id: string, name: string) => {
    setActiveId(id);
    setActiveName(name);
    setPage("live");
  }, []);

  const selectInvestigation = useCallback((id: string, name: string) => {
    setActiveId(id);
    setActiveName(name);
  }, []);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <span className="brand-mark">L</span>
          <div>
            <div className="brand-name">LeakLens</div>
            <div className="brand-sub">ML Leakage Investigator</div>
          </div>
        </div>
        <nav className="nav">
          {NAV.map((n) => (
            <button
              key={n.key}
              className={`nav-item ${page === n.key ? "active" : ""}`}
              onClick={() => setPage(n.key)}
            >
              <span className="nav-icon">{n.icon}</span>
              {n.label}
            </button>
          ))}
        </nav>
        <div className="sidebar-footer">
          <div className={`pill ${backendOnline ? "ok" : "down"}`}>
            <span className="dot" /> Backend {backendOnline === null ? "…" : backendOnline ? "online" : "offline"}
          </div>
          <div className="pill gemini">
            <span className="dot" /> Gemini {gemini ? "ready" : "off"}
          </div>
        </div>
      </aside>

      <main className="content">
        {page === "dashboard" && (
          <Dashboard
            onSelect={(id, name) => {
              selectInvestigation(id, name);
              setPage("findings");
            }}
            onOpenNew={() => setPage("new")}
          />
        )}
        {page === "new" && <NewInvestigation onStarted={goNew} />}
        {page === "live" && (
          <LiveInvestigation
            investigationId={activeId}
            investigationName={activeName}
            onShowFindings={() => setPage("findings")}
          />
        )}
        {page === "findings" && (
          <Findings investigationId={activeId} onSelect={selectInvestigation} />
        )}
        {page === "reports" && (
          <Reports investigationId={activeId} onSelect={selectInvestigation} />
        )}
      </main>
    </div>
  );
}
