import { FormEvent, ReactNode, useRef, useState } from "react";
import { api } from "../api";

function Field({ label, hint, children }: { label: string; hint?: string; children: ReactNode }) {
  return (
    <label className="field">
      <span className="field-label">{label}</span>
      {children}
      {hint && <span className="field-hint">{hint}</span>}
    </label>
  );
}

export default function NewInvestigation({
  onStarted,
}: {
  onStarted: (id: string, name: string) => void;
}) {
  const datasetRef = useRef<HTMLInputElement>(null);
  const scriptRef = useRef<HTMLInputElement>(null);
  const [target, setTarget] = useState("");
  const [timestamp, setTimestamp] = useState("");
  const [entity, setEntity] = useState("");
  const [unavailable, setUnavailable] = useState("");
  const [gemini, setGemini] = useState(false);
  const [seed, setSeed] = useState("42");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState("");

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setError("");
    if (!datasetRef.current?.files?.length) {
      setError("Please choose a CSV dataset.");
      return;
    }
    if (!target.trim()) {
      setError("Please provide the target column name.");
      return;
    }
    const fd = new FormData();
    const file = datasetRef.current.files[0];
    fd.append("file", file);
    fd.append("target_column", target.trim());
    fd.append("random_seed", seed || "42");
    fd.append("gemini_enabled", String(gemini));
    if (scriptRef.current?.files?.length) fd.append("script", scriptRef.current.files[0]);
    if (timestamp.trim()) fd.append("timestamp_column", timestamp.trim());
    if (entity.trim()) fd.append("entity_column", entity.trim());
    if (unavailable.trim()) {
      fd.append(
        "unavailable_at_prediction",
        JSON.stringify(unavailable.split(",").map((s) => s.trim()).filter(Boolean))
      );
    }

    setBusy(true);
    try {
      const inv = await api.createInvestigation(fd);
      onStarted(inv.id, inv.source_filename);
    } catch (err) {
      setError(String(err));
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="narrow">
      <div className="page-head">
        <div>
          <h1>New Investigation</h1>
          <p className="subtitle">Provide a dataset, target, and optional training script for inspection.</p>
        </div>
      </div>

      {error && <div className="error-box">{error}</div>}

      <form className="form" onSubmit={onSubmit}>
        <div className="form-card">
          <h2>Dataset</h2>
          <Field label="Dataset (CSV) *" hint="CSV only, max 25 MB">
            <input ref={datasetRef} type="file" accept=".csv" />
          </Field>
          <Field label="Python training script (optional)" hint="Static AST inspection only — never executed">
            <input ref={scriptRef} type="file" accept=".py" />
          </Field>
        </div>

        <div className="form-card">
          <h2>Configuration</h2>
          <div className="grid2">
            <Field label="Target column *">
              <input value={target} onChange={(e) => setTarget(e.target.value)} placeholder="e.g. y" />
            </Field>
            <Field label="Random seed">
              <input type="number" value={seed} onChange={(e) => setSeed(e.target.value)} />
            </Field>
          </div>
          <div className="grid2">
            <Field label="Timestamp column" hint="enables temporal-leakage checks">
              <input value={timestamp} onChange={(e) => setTimestamp(e.target.value)} placeholder="e.g. timestamp" />
            </Field>
            <Field label="Entity / group column" hint="enables contamination checks">
              <input value={entity} onChange={(e) => setEntity(e.target.value)} placeholder="e.g. customer_id" />
            </Field>
          </div>
          <Field label="Features unavailable at prediction time (comma-separated)">
            <input value={unavailable} onChange={(e) => setUnavailable(e.target.value)} placeholder="e.g. duration, future_price" />
          </Field>
        </div>

        <div className="form-card">
          <h2>Assistance</h2>
          <label className="check">
            <input type="checkbox" checked={gemini} onChange={(e) => setGemini(e.target.checked)} />
            <span>Enable Gemini hypothesis assistance (optional; requires backend key)</span>
          </label>
        </div>

        <button className="btn primary" disabled={busy}>
          {busy ? "Starting…" : "Start Investigation"}
        </button>
      </form>
    </div>
  );
}
