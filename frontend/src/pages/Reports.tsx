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

  useEffect(() => {
    api.listInvestigations().then(setInvestigations).catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!selected) {
      setDetail(null);
      return;
    }
    api
      .detail(selected)
      .then(setDetail)
      .catch(() => undefined);
    setPreview(false);
  }, [selected]);

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

      {selected && detail && (
        <>
          <div className="cards">
            <a className="card clickable" href={api.reportUrl(selected, "json")} download>
              <div className="card-icon">｛</div>
              <div className="card-body">
                <div className="card-label">Download JSON report</div>
                <div className="card-value">JSON</div>
              </div>
            </a>
            <a className="card clickable" href={api.reportUrl(selected, "html")} download>
              <div className="card-icon">⛶</div>
              <div className="card-body">
                <div className="card-label">Download HTML report</div>
                <div className="card-value">HTML</div>
              </div>
            </a>
            <div className="card clickable" onClick={() => setPreview((p) => !p)}>
              <div className="card-icon">▤</div>
              <div className="card-body">
                <div className="card-label">Quick preview</div>
                <div className="card-value">…</div>
              </div>
            </div>
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
              <details open className="preview">
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
