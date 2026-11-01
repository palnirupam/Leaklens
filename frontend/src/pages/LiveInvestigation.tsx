import { useEffect, useState } from "react";
import { api, Investigation, InvestigationDetail } from "../api";

const STAGES: { key: string; label: string }[] = [
  { key: "queued", label: "Queued" },
  { key: "dataset_profiling", label: "Dataset profiling" },
  { key: "static_code_analysis", label: "Static code analysis" },
  { key: "signal_detection", label: "Signal detection" },
  { key: "hypothesis_planning", label: "Hypothesis planning" },
  { key: "adaptive_experiments", label: "Adaptive experiments" },
  { key: "evidence_verification", label: "Evidence verification" },
  { key: "completed", label: "Completed" },
];

function stageIndex(stage: string): number {
  const i = STAGES.findIndex((s) => s.key === stage);
  return i < 0 ? 0 : i;
}

const SEV_CLASS: Record<string, string> = {
  critical: "sev-critical",
  high: "sev-high",
  medium: "sev-medium",
  low: "sev-low",
  info: "sev-info",
};

export default function LiveInvestigation({
  investigationId,
  investigationName,
  onShowFindings,
}: {
  investigationId: string | null;
  investigationName: string;
  onShowFindings: () => void;
}) {
  const [investigation, setInvestigation] = useState<Investigation | null>(null);
  const [detail, setDetail] = useState<InvestigationDetail | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    if (!investigationId) return;
    let alive = true;
    let timer: ReturnType<typeof setTimeout>;

    const poll = async () => {
      try {
        const [inv, det] = await Promise.all([
          api.getInvestigation(investigationId),
          api.detail(investigationId),
        ]);
        if (!alive) return;
        setInvestigation(inv);
        setDetail(det);
        if (inv.status === "completed" || inv.status === "failed") return;
        timer = setTimeout(poll, 1200);
      } catch (e) {
        if (alive) setError(String(e));
      }
    };
    poll();
    return () => {
      alive = false;
      clearTimeout(timer);
    };
  }, [investigationId]);

  if (!investigationId) {
    return (
      <div>
        <h1>Live Investigation</h1>
        <div className="empty">
          <div className="empty-icon">▸</div>
          <p>Start a new investigation or select one from the Dashboard.</p>
        </div>
      </div>
    );
  }

  if (error) return <div className="error-box">{error}</div>;
  if (!investigation) return <div className="muted">Loading investigation…</div>;

  const done = investigation.status === "completed";
  const failed = investigation.status === "failed";
  const pct = Math.round(investigation.progress * 100);
  const current = stageIndex(investigation.stage);
  const showHypotheses = detail && detail.hypotheses.length > 0;
  const showExperiments = detail && detail.experiments.length > 0;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Live Investigation</h1>
          <div className="meta-row">
            <span className="badge">{investigationName || investigation.source_filename}</span>
            <span className="badge">target: {investigation.target_column}</span>
            {investigation.gemini_used && <span className="badge gemini">Gemini</span>}
            <span className={`status-badge status-${investigation.status}`}>{investigation.status}</span>
          </div>
        </div>
      </div>

      {/* Progress */}
      <div className="section">
        <div className="progress-track">
          <div className="progress-fill" style={{ width: `${pct}%` }} />
        </div>
        <div className="progress-row">
          <span>Stage: <b>{STAGES[current].label}</b></span>
          <span className="muted">{pct}%</span>
        </div>
        {investigation.current_agent && (
          <p className="muted">Agent: <code>{investigation.current_agent}</code></p>
        )}
      </div>

      {/* Timeline */}
      <div className="section">
        <h2>Pipeline</h2>
        <div className="timeline">
          {STAGES.map((s, i) => {
            const state = i < current ? "done" : i === current ? "active" : "todo";
            return (
              <div key={s.key} className={`tl-step ${state}`}>
                <span className="tl-dot">{state === "done" ? "✓" : i + 1}</span>
                <span className="tl-label">{s.label}</span>
              </div>
            );
          })}
        </div>
      </div>

      {failed && (
        <div className="error-box">Investigation failed: {investigation.error}</div>
      )}

      {/* Hypotheses */}
      {showHypotheses && (
        <div className="section">
          <div className="section-head"><h2>Hypotheses</h2></div>
          <div className="hyp-list">
            {detail!.hypotheses.map((h) => (
              <div className="hyp" key={h.id}>
                <div className="hyp-top">
                  <span className={`badge cat-${h.category}`}>{h.category.replace(/_/g, " ")}</span>
                  <span className="hyp-status">{h.status}</span>
                  <span className="hyp-source">{h.source === "gemini" ? "✦ Gemini" : "rule"}</span>
                </div>
                <div className="hyp-title">{h.title}</div>
                <div className="conf">
                  <div className="conf-track">
                    <div className="conf-fill" style={{ width: `${Math.round(h.confidence * 100)}%` }} />
                  </div>
                  <span className="conf-num">{Math.round(h.confidence * 100)}%</span>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Experiments */}
      {showExperiments && (
        <div className="section">
          <div className="section-head"><h2>Experiments</h2></div>
          <table className="table">
            <thead>
              <tr><th>Experiment</th><th>Status</th><th>Interpretation</th><th>Cost</th></tr>
            </thead>
            <tbody>
              {detail!.experiments.map((e) => (
                <tr key={e.id}>
                  <td className="mono">{e.name}</td>
                  <td><span className={`status-badge status-${e.status}`}>{e.status}</span></td>
                  <td className="muted">{e.interpretation || "—"}</td>
                  <td>{e.status === "completed" ? `${e.cost_ms.toFixed(0)} ms` : e.status}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Findings on completion */}
      {done && detail && detail.findings.length > 0 && (
        <div className="section success">
          <div className="section-head"><h2>Findings</h2></div>
          {detail.findings.map((f) => (
            <div className="finding" key={f.id}>
              <div className="finding-head">
                <span className={`sev ${SEV_CLASS[f.severity] || "sev-info"}`}>{f.severity}</span>
                <span className={`sev sev-ev-${f.evidence_status}`}>{f.evidence_status}</span>
                <span className="muted">{f.category.replace(/_/g, " ")}</span>
              </div>
              <h3>{f.title}</h3>
            </div>
          ))}
          <button className="btn primary" onClick={onShowFindings}>Open full Findings</button>
        </div>
      )}

      {done && detail && detail.findings.length === 0 && (
        <div className="section success">
          <h2>Investigation complete</h2>
          <p>{detail.summary || "No leakage detected by the supported checks."}</p>
        </div>
      )}
    </div>
  );
}
