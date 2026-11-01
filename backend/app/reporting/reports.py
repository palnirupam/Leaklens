"""Report generation: JSON and HTML.
"""
from __future__ import annotations

import html
import json
from typing import Any

from app.core.models import Investigation


def investigation_to_dict(inv: Investigation) -> dict[str, Any]:
    def _col(c):
        return {
            "name": c.name,
            "dtype": c.dtype,
            "null_pct": round(c.null_pct, 2),
            "n_unique": c.n_unique,
            "association": round(c.target_association, 3),
            "suspicious": c.suspicious,
            "suspicious_reason": c.suspicious_reason,
        }

    return {
        "id": inv.id,
        "status": inv.status,
        "stage": inv.stage,
        "created_at": inv.created_at,
        "finished_at": inv.finished_at,
        "gemini_used": inv.gemini_used,
        "summary": inv.summary,
        "config": {
            "target": inv.config.target_column,
            "timestamp_column": inv.config.timestamp_column,
            "entity_column": inv.config.entity_column,
            "unavailable_at_prediction": inv.config.unavailable_at_prediction,
            "source_filename": inv.config.source_filename,
            "random_seed": inv.config.random_seed,
        },
        "dataset_profile": {
            "row_count": inv.profile.row_count if inv.profile else 0,
            "column_count": inv.profile.column_count if inv.profile else 0,
            "duplicate_rows": inv.profile.duplicate_rows if inv.profile else 0,
            "target_class_balance": inv.profile.target_class_balance if inv.profile else {},
            "columns": [_col(c) for c in inv.profile.columns] if inv.profile else [],
            "warnings": inv.profile.warnings if inv.profile else [],
        },
        "hypotheses": [
            {
                "id": h.id,
                "title": h.title,
                "category": h.category,
                "rationale": h.rationale,
                "affected_features": h.affected_features,
                "source": h.source,
                "confidence": round(h.confidence, 2),
                "status": h.status,
            }
            for h in inv.hypotheses
        ],
        "experiments": [
            {
                "id": r.experiment_id,
                "hypothesis_id": r.hypothesis_id,
                "name": r.name,
                "status": r.status,
                "metrics": r.metrics,
                "evidence": r.evidence,
                "interpretation": r.interpretation,
                "cost_ms": round(r.cost_ms, 1),
                "error": r.error,
            }
            for r in inv.results
        ],
        "findings": [
            {
                "id": f.id,
                "title": f.title,
                "category": f.category,
                "severity": f.severity,
                "evidence_status": f.evidence_status,
                "description": f.description,
                "affected_features": f.affected_features,
                "code_paths": f.code_paths,
                "correction": f.correction,
                "verification_status": f.verification_status,
            }
            for f in inv.findings
        ],
    }


def render_json(inv: Investigation) -> str:
    return json.dumps(investigation_to_dict(inv), indent=2, default=str)


def render_html(inv: Investigation) -> str:
    d = investigation_to_dict(inv)
    findings_rows = []
    for f in d["findings"]:
        findings_rows.append(
            "<tr>"
            f"<td>{html.escape(f['category'])}</td>"
            f"<td><span class='sev sev-{html.escape(f['severity'])}'>{html.escape(f['severity'])}</span></td>"
            f"<td>{html.escape(f['evidence_status'])}</td>"
            f"<td>{html.escape(f['title'])}</td>"
            f"<td>{html.escape(f['description'][:140])}</td>"
            f"<td>{html.escape(', '.join(f['affected_features']))}</td>"
            "</tr>"
        )
    exp_rows = []
    for e in d["experiments"]:
        exp_rows.append(
            "<tr>"
            f"<td>{html.escape(e['name'])}</td>"
            f"<td>{html.escape(e['status'])}</td>"
            f"<td>{html.escape(json.dumps(e['metrics']))}</td>"
            f"<td>{html.escape(e['interpretation'])}</td>"
            f"<td>{e['cost_ms']} ms</td>"
            "</tr>"
        )
    html_text = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>LeakLens report</title>
<style>
body{{font-family:Segoe UI,system-ui,sans-serif;margin:24px;color:#1f2933;background:#f8fafc}}
h1{{border-bottom:3px solid #2563eb;padding-bottom:8px}}
.section{{background:#fff;border:1px solid #e2e8f0;border-radius:8px;padding:16px;margin:16px 0}}
table{{width:100%;border-collapse:collapse;font-size:14px}}
th,td{{text-align:left;padding:8px;border-bottom:1px solid #e2e8f0}}
th{{background:#f1f5f9}}
.sev{{padding:2px 8px;border-radius:10px;color:#fff;font-size:12px}}
.sev-critical{{background:#dc2626}}.sev-high{{background:#ea580c}}
.sev-medium{{background:#d97706}}.sev-low{{background:#0891b2}}.sev-info{{background:#64748b}}
</style></head><body>
<h1>LeakLens Investigation Report</h1>
<p><b>Investigation:</b> {html.escape(d['id'])}</p>
<p><b>Dataset:</b> {html.escape(d['config']['source_filename'])} &nbsp; <b>Target:</b> {html.escape(d['config']['target'])}</p>
<p><b>Status:</b> {html.escape(d['status'])} &nbsp; <b>Gemini used:</b> {d['gemini_used']}</p>
<p>{html.escape(d['summary'])}</p>

<div class="section"><h2>Dataset Profile</h2>
<p>Rows: {d['dataset_profile']['row_count']} &nbsp; Columns: {d['dataset_profile']['column_count']} &nbsp; Unit rows: {d['dataset_profile']['duplicate_rows']}</p>
<table><tr><th>Column</th><th>Type</th><th>Null %</th><th>Unique</th><th>Assoc</th><th>Suspicious</th></tr>
{''.join('<tr><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td><td>%s</td></tr>' % (html.escape(c['name']), html.escape(c['dtype']), c['null_pct'], c['n_unique'], c['association'], html.escape(c.get('suspicious_reason','') or '')) for c in d['dataset_profile']['columns'])}
</table></div>

<div class="section"><h2>Findings</h2>
<table><tr><th>Category</th><th>Severity</th><th>Evidence</th><th>Title</th><th>Description</th><th>Features</th></tr>
{''.join(findings_rows)}
</table></div>

<div class="section"><h2>Experiments</h2>
<table><tr><th>Experiment</th><th>Status</th><th>Metrics</th><th>Interpretation</th><th>Cost</th></tr>
{''.join(exp_rows)}
</table></div>

<div class="section"><h2>Recommendations</h2>
<ul>
{''.join('<li><b>%s</b>: %s</li>' % (html.escape(f['category']), html.escape(f['correction'])) for f in d['findings'] if f['correction'])}
</ul></div>
</body></html>"""
    return html_text
