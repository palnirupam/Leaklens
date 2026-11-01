// Thin typed client for the LeakLens backend.

export interface Investigation {
  id: string;
  status: string;
  stage: string;
  progress: number;
  current_agent: string;
  created_at: string;
  finished_at: string | null;
  error: string | null;
  gemini_used: boolean;
  source_filename: string;
  target_column: string;
  finding_count: number;
  experiment_count: number;
  summary: string;
}

export interface Stats {
  total_investigations: number;
  confirmed_findings: number;
  potential_findings: number;
  avg_duration_sec: number;
  recent: Investigation[];
}

export interface Finding {
  id: string;
  title: string;
  category: string;
  severity: string;
  evidence_status: string;
  description: string;
  affected_features: string[];
  code_paths: string[];
  correction: string;
  verification_status: string;
}

export interface Hypothesis {
  id: string;
  title: string;
  category: string;
  rationale: string;
  affected_features: string[];
  source: string;
  confidence: number;
  status: string;
}

export interface Experiment {
  id: string;
  hypothesis_id: string;
  name: string;
  status: string;
  metrics: Record<string, unknown>;
  evidence: Record<string, unknown>;
  interpretation: string;
  cost_ms: number;
  error: string | null;
}

export interface InvestigationDetail {
  id: string;
  status: string;
  stage: string;
  created_at: string;
  finished_at: string | null;
  gemini_used: boolean;
  summary: string;
  config: {
    target: string;
    timestamp_column: string | null;
    entity_column: string | null;
    unavailable_at_prediction: string[];
    source_filename: string;
    random_seed: number;
  };
  dataset_profile: {
    row_count: number;
    column_count: number;
    duplicate_rows: number;
    target_class_balance: Record<string, number>;
    columns: {
      name: string;
      dtype: string;
      null_pct: number;
      n_unique: number;
      association: number;
      suspicious: boolean;
      suspicious_reason: string;
    }[];
    warnings: string[];
  };
  hypotheses: Hypothesis[];
  experiments: Experiment[];
  findings: Finding[];
}

const API_KEY = (import.meta.env.VITE_API_KEY as string | undefined) || "";

function authHeaders(): Record<string, string> {
  return API_KEY ? { "X-API-Key": API_KEY } : {};
}

async function getJSON<T>(url: string): Promise<T> {
  const res = await fetch(url, { headers: authHeaders() });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`HTTP ${res.status}: ${body}`);
  }
  return res.json() as Promise<T>;
}

export const api = {
  health: () => getJSON<{ status: string; gemini_configured: boolean }>("/api/health"),
  stats: () => getJSON<Stats>("/api/stats"),
  listInvestigations: () => getJSON<Investigation[]>("/api/investigations"),
  getInvestigation: (id: string) => getJSON<Investigation>(`/api/investigations/${id}`),
  detail: (id: string) => getJSON<InvestigationDetail>(`/api/investigations/${id}/detail`),
  findings: (id: string) => getJSON<Finding[]>(`/api/investigations/${id}/findings`),
  reportUrl: (id: string, format: "json" | "html") =>
    `/api/investigations/${id}/report?format=${format}`,
  createInvestigation: async (form: FormData): Promise<Investigation> => {
    const res = await fetch("/api/investigations", {
      method: "POST",
      body: form,
      headers: authHeaders(),
    });
    if (!res.ok) {
      const body = await res.text().catch(() => "");
      throw new Error(`HTTP ${res.status}: ${body}`);
    }
    return res.json() as Promise<Investigation>;
  },
};
