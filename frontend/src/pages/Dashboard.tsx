import { useEffect, useState } from "react";
import { api, Stats } from "../api";

function Card({ icon, label, value, hint }: { icon: string; label: string; value: string; hint?: string }) {
  return (
    <div className="card">
      <div className="card-icon">{icon}</div>
      <div className="card-body">
        <div className="card-label">{label}</div>
        <div className="card-value">{value}</div>
        {hint && <div className="card-hint">{hint}</div>}
      </div>
    </div>
  );
}

export default function Dashboard({
  onSelect,
  onOpenNew,
}: {
  onSelect: (id: string, name: string) => void;
  onOpenNew: () => void;
}) {
  const [stats, setStats] = useState<Stats | null>(null);
  const [error, setError] = useState("");

  useEffect(() => {
    let alive = true;
    api
      .stats()
      .then((s) => alive && setStats(s))
      .catch((e) => alive && setError(String(e)));
    return () => {
      alive = false;
    };
  }, []);

  const maxFindings = stats && stats.recent.length > 0
    ? Math.max(1, ...stats.recent.map((i) => i.finding_count))
    : 1;

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Dashboard</h1>
          <p className="subtitle">
            Autonomous ML data-leakage investigation — budget-aware adaptive experiments.
          </p>
        </div>
        <button className="btn primary" onClick={onOpenNew}>＋ New Investigation</button>
      </div>

      {error && <div className="error-box">{error}</div>}

      {stats && (
        <>
          <div className="cards">
            <Card
              icon="◔"
              label="Total Investigations"
              value={String(stats.total_investigations)}
            />
            <Card
              icon="✓"
              label="Confirmed Findings"
              value={String(stats.confirmed_findings)}
              hint="confirmed / strong evidence"
            />
            <Card
              icon="⚠"
              label="Potential Findings"
              value={String(stats.potential_findings)}
              hint="need manual review"
            />
            <Card
              icon="⏱"
              label="Avg Investigation"
              value={`${stats.avg_duration_sec.toFixed(1)}s`}
              hint="wall-clock"
            />
          </div>

          <div className="section">
            <div className="section-head">
              <h2>Recent Analysis History</h2>
              {stats.recent.length > 0 && (
                <span className="muted">showing last {stats.recent.length}</span>
              )}
            </div>

            {stats.recent.length === 0 ? (
              <div className="empty">
                <div className="empty-icon">◌</div>
                <p>No investigations yet.</p>
                <button className="btn primary" onClick={onOpenNew}>Start your first investigation</button>
              </div>
            ) : (
              <>
                <div className="mini-chart">
                  {stats.recent.map((i) => (
                    <div key={i.id} className="bar-col" title={`${i.source_filename}: ${i.finding_count} findings`}>
                      <div
                        className="bar"
                        style={{ height: `${Math.max(8, (i.finding_count / maxFindings) * 100)}%` }}
                      />
                      <span className="bar-label">{i.finding_count}</span>
                    </div>
                  ))}
                </div>

                <table className="table">
                  <thead>
                    <tr>
                      <th>Dataset</th>
                      <th>Target</th>
                      <th>Status</th>
                      <th>Findings</th>
                      <th>Started</th>
                    </tr>
                  </thead>
                  <tbody>
                    {stats.recent.map((i) => (
                      <tr
                        key={i.id}
                        className="clickable"
                        onClick={() => onSelect(i.id, i.source_filename)}
                      >
                        <td className="mono">{i.source_filename}</td>
                        <td>{i.target_column}</td>
                        <td><span className={`status-badge status-${i.status}`}>{i.status}</span></td>
                        <td><span className="count">{i.finding_count}</span></td>
                        <td className="muted">{new Date(i.created_at).toLocaleString()}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </>
            )}
          </div>
        </>
      )}
    </div>
  );
}
