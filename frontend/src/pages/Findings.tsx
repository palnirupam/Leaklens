import { useEffect, useMemo, useState } from "react";
import { api, Finding, Investigation } from "../api";

const SEV_CLASS: Record<string, string> = {
  critical: "sev-critical",
  high: "sev-high",
  medium: "sev-medium",
  low: "sev-low",
  info: "sev-info",
};

const SEV_LABEL: Record<string, string> = {
  critical: "Critical",
  high: "High",
  medium: "Medium",
  low: "Low",
  info: "Info",
};

function catLabel(cat: string): string {
  return cat.replace(/_/g, " ");
}

function CopyButton({ text }: { text: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <button
      className="btn mini"
      onClick={() => {
        navigator.clipboard?.writeText(text).then(() => {
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        });
      }}
    >
      {copied ? "✓ Copied" : "⧉ Copy"}
    </button>
  );
}

export default function Findings({
  investigationId,
  onSelect,
}: {
  investigationId: string | null;
  onSelect: (id: string, name: string) => void;
}) {
  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [selected, setSelected] = useState<string | null>(investigationId);
  const [findings, setFindings] = useState<Finding[] | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    api
      .listInvestigations()
      .then(setInvestigations)
      .catch((e) => setError(String(e)));
  }, []);

  useEffect(() => {
    if (!selected) {
      setFindings(null);
      return;
    }
    setFindings(null);
    api
      .findings(selected)
      .then(setFindings)
      .catch((e) => setError(String(e)));
  }, [selected]);

  const counts = useMemo(() => {
    if (!findings) return { total: 0, confirmed: 0, potential: 0 };
    const confirmed = findings.filter(
      (f) => f.evidence_status === "confirmed" || f.evidence_status === "strong"
    ).length;
    return {
      total: findings.length,
      confirmed,
      potential: findings.length - confirmed,
    };
  }, [findings]);

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Findings</h1>
          <p className="subtitle">Evidence-backed leakage findings per investigation.</p>
        </div>
      </div>

      <div className="toolbar">
        <select
          className={selected ? "" : "placeholder"}
          value={selected ?? ""}
          onChange={(e) => {
            const v = e.target.value;
            setSelected(v || null);
            const inv = investigations.find((i) => i.id === v);
            if (v) onSelect(v, inv?.source_filename ?? "");
          }}
        >
          <option value="">Select an investigation…</option>
          {investigations.map((i) => (
            <option key={i.id} value={i.id}>
              {i.source_filename} → {i.target_column}
            </option>
          ))}
        </select>
      </div>

      {error && <div className="error-box">{error}</div>}

      {/* Loading skeleton */}
      {!findings && selected && (
        <div className="skeleton-stack">
          <div className="skel" style={{ height: 96 }} />
          <div className="skel" style={{ height: 150 }} />
          <div className="skel" style={{ height: 120 }} />
        </div>
      )}

      {findings && findings.length === 0 && (
        <div className="empty">
          <div className="empty-icon">✓</div>
          <p>No findings — no leakage detected by the supported checks.</p>
        </div>
      )}

      {findings && findings.length > 0 && (
        <>
          {/* Stats strip */}
          <div className="findings-stats">
            <div className="stat-chip">
              <span className="stat-num">{counts.total}</span>
              <span className="stat-label">Findings</span>
            </div>
            <div className="stat-chip good">
              <span className="stat-num">{counts.confirmed}</span>
              <span className="stat-label">Confirmed / strong</span>
            </div>
            <div className="stat-chip warn">
              <span className="stat-num">{counts.potential}</span>
              <span className="stat-label">Potential</span>
            </div>
          </div>

          <div className="section no-pad">
            {findings.map((f) => (
              <div className={`finding-card ${SEV_CLASS[f.severity] || "sev-info"}`} key={f.id}>
                <div className="finding-card-head">
                  <span className={`sev ${SEV_CLASS[f.severity] || "sev-info"}`}>
                    {SEV_LABEL[f.severity] || f.severity}
                  </span>
                  <span className={`sev sev-ev-${f.evidence_status}`}>{f.evidence_status}</span>
                  <span className="badge">{catLabel(f.category)}</span>
                  <span className="muted verify">
                    {f.verification_status === "supported_by_experiment"
                      ? "⚙ verified by experiment"
                      : f.verification_status}
                  </span>
                  <div className="spacer" />
                  <CopyButton text={f.description} />
                </div>

                <h3 className="finding-title">{f.title}</h3>
                <p className="finding-desc">{f.description}</p>

                <div className="finding-meta">
                  {f.affected_features.length > 0 && (
                    <div className="chips">
                      <span className="chip-label">Affected features</span>
                      {f.affected_features.map((af) => (
                        <span className="chip feature" key={af}>{af}</span>
                      ))}
                    </div>
                  )}

                  {f.code_paths.length > 0 && (
                    <div className="chips">
                      <span className="chip-label">Code</span>
                      {f.code_paths.map((cp) => (
                        <span className="chip code" key={cp}>{cp}</span>
                      ))}
                    </div>
                  )}
                </div>

                {f.correction && (
                  <div className="correction">
                    <span className="correction-icon">💡</span>
                    <div>
                      <b>Suggested correction</b>
                      <p>{f.correction}</p>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>
        </>
      )}
    </div>
  );
}
