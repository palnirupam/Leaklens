import { useEffect, useState } from "react";
import { api, Investigation, InvestigationDetail } from "../api";

export default function Reports({
  investigationId,
  onSelect,
}: {
  investigationId: string | null;
  onSelect: (id: string, name: string) => void;
}) {
  const [investigations, setInvestigations] = useState<Investigation[]>([]);
  const [selected, setSelected] = useState<string | null>(investigationId);
  const [detail, setDetail] = useState<InvestigationDetail | null>(null);
  const [preview, setPreview] = useState(false);
  const [error, setError] = useState("");
  const [downloading, setDownloading] = useState(false);

  useEffect(() => {
    let alive = true;
    api.listInvestigations()
      .then((items) => alive && setInvestigations(items))
      .catch((err) => alive && setError(String(err)));
    return () => { alive = false; };
  }, []);

  useEffect(() => {
    setDetail(null);
    setError("");
    setPreview(false);
    if (!selected) return;
    let alive = true;
    api
      .detail(selected)
      .then((report) => alive && setDetail(report))
      .catch((err) => alive && setError(String(err)));
    return () => { alive = false; };
  }, [selected]);

  async function download(format: "json" | "html") {
    if (!selected) return;
    setError("");
    setDownloading(true);
    try {
      await api.downloadReport(selected, format);
    } catch (err) {
      setError(String(err));
    } finally {
      setDownloading(false);
    }
  }

  return (
    <div>
      <div className="page-head">
        <div>
          <h1>Reports</h1>
          <p className="subtitle">Download a full investigation report — dataset profile, experiments, findings, recommendations.</p>
        </div>
      </div>

      <div className="toolbar">
        <select
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

      {error && <div className="error-box" role="alert">{error}</div>}

      {selected && detail && (
        <>
          <div className="cards">
            <button type="button" className="card clickable" disabled={downloading} onClick={() => download("json")}>
              <div className="card-icon">｛</div>
              <div className="card-body">
                <div className="card-label">Download JSON report</div>
                <div className="card-value">JSON</div>
              </div>
            </button>
            <button type="button" className="card clickable" disabled={downloading} onClick={() => download("html")}>
              <div className="card-icon">⛶</div>
              <div className="card-body">
                <div className="card-label">Download HTML report</div>
                <div className="card-value">HTML</div>
              </div>
            </button>
            <button type="button" className="card clickable" aria-expanded={preview} aria-controls="report-preview" onClick={() => setPreview((p) => !p)}>
              <div className="card-icon">▤</div>
              <div className="card-body">
                <div className="card-label">Quick preview</div>
                <div className="card-value">…</div>
              </div>
            </button>
          </div>

          <div className="section">
            <h2>Report summary</h2>
            <div className="kv">
              <div><span>Investigation</span><b>{detail.id}</b></div>
              <div><span>Dataset</span><b>{detail.config.source_filename}</b></div>
              <div><span>Target</span><b>{detail.config.target}</b></div>
              <div><span>Status</span><b>{detail.status}</b></div>
              <div><span>Summary</span><b>{detail.summary}</b></div>
              <div><span>Rows / Columns</span><b>{detail.dataset_profile.row_count} / {detail.dataset_profile.column_count}</b></div>
              <div><span>Experiments</span><b>{detail.experiments.length}</b></div>
              <div><span>Findings</span><b>{detail.findings.length}</b></div>
            </div>

            {preview && (
              <details open id="report-preview" className="preview">
                <summary>JSON preview</summary>
                <pre>{JSON.stringify(detail, null, 2)}</pre>
              </details>
            )}
          </div>
        </>
      )}
    </div>
  );
}
