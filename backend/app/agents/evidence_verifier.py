"""Agent 4 — Evidence Verification Agent.

Consumes the executed experiments and produces the final findings. Only
experimental or static-analysis evidence may establish a finding. Unsupported
Gemini-generated hypotheses are rejected, and inconclusive hypotheses are
reported as "suspicious / needs manual review" rather than confirmed.
"""
from __future__ import annotations

import uuid
from typing import Optional

from app.core.models import Investigation, Finding, CandidateHypothesis

EVIDENCE_STRENGTH = {
    "confirmed": 3,
    "strong": 2,
    "potential": 1,
    "suspicious": 1,
    "inconclusive": 0,
}

CORRECTIONS = {
    "target_derived_feature": (
        "High feature-target association is NOT proof of leakage on its own. "
        "Only remove this column if you can verify it is genuinely derived from "
        "the target, from future information, or otherwise unavailable at "
        "prediction time; otherwise it may simply be a legitimate strong "
        "predictor — keep it, but document why it is safe."
    ),
    "prediction_time_unavailable": (
        "Exclude this feature from the production model (it is not available at "
        "prediction time). Document the exclusion in the model card."
    ),
    "duplicate_contamination": (
        "Deduplicate records and use an entity-aware split so no entity spans both "
        "train and test. Re-evaluate metrics after cleaning."
    ),
    "temporal_leakage": (
        "Split chronologically (train on the past, validate on the future) so the "
        "model never sees future information during training."
    ),
    "preprocessing_leakage": (
        "Move all preprocessing (scaling, imputation, encoding, oversampling) inside "
        "a pipeline that is fitted on the training split only."
    ),
}


class EvidenceVerifier:
    def verify(self, investigation: Investigation) -> list[Finding]:
        findings: list[Finding] = []
        for h in investigation.hypotheses:
            finding = self._verify_hypothesis(investigation, h)
            if finding is not None:
                findings.append(finding)
        investigation.findings = findings
        return findings

    def _verify_hypothesis(self, investigation: Investigation, h: CandidateHypothesis) -> Optional[Finding]:
        results = [r for r in investigation.results if r.hypothesis_id == h.id and r.status == "completed"]
        supporting = [r for r in results if self._supports(h, r)]

        # Gemini-only claims without any supporting experiment are rejected.
        if h.source == "gemini" and not supporting and h.category not in (
            "prediction_time_unavailable", "preprocessing_leakage"
        ):
            h.status = "rejected"
            return None

        # Static preprocessing suspicion -> potential finding (not proven).
        if h.category == "preprocessing_leakage":
            if supporting:
                return self._build(
                    investigation, h, "potential", "suspicious",
                    "Static analysis flagged preprocessing-before-split patterns; "
                    "runtime leakage is not proven without a pipeline experiment.",
                )
            return None

        # Availability is a deployment fact, reported as info/low (not a leak).
        if h.category == "prediction_time_unavailable":
            return self._build(
                investigation, h, "low", "suspicious",
                "Feature is unavailable at prediction time. This is a deployment "
                "hazard, not evidence of train/test contamination.",
            )

        # Interaction flags are review-only — never auto-confirmed as leakage.
        if h.category == "feature_interaction_leakage":
            return self._build(
                investigation, h, "medium", "suspicious",
                "Two features are individually uninformative but strongly predictive "
                "together. This is NOT proof of leakage: verify whether the "
                "combination is derived from target/future information or is simply "
                "a legitimate interaction before acting on it.",
            )

        if not results:
            h.status = "inconclusive"
            if h.category == "duplicate_contamination" and investigation.config.entity_column:
                return self._build(
                    investigation, h, "medium", "suspicious",
                    "Duplicate/entity pattern detected but not yet tested.",
                )
            return None

        if not supporting:
            # we have results but none support -> reject hypothesis
            h.status = "rejected"
            return None

        # Confidence based on number of corroborating experiments.
        n_support = len(supporting)
        if n_support >= 2:
            status = "confirmed"
            severity = "critical"
        else:
            status = "strong"
            severity = "high"
        h.status = "confirmed" if status == "confirmed" else "investigating"

        # Predictive power alone must not be labelled "confirmed leak". Only a
        # mechanism-level reason (e.g. static analysis showing the feature is
        # computed from target/future) justifies confirmation; otherwise we keep
        # it as a strong signal flagged for manual review.
        if h.category == "target_derived_feature" and not h.code_paths:
            status = "potential"
            severity = "medium"
            h.status = "investigating"

        return self._build(
            investigation, h, severity, status,
            self._describe_evidence(h, supporting),
            verified_by="experiment",
        )

    def _supports(self, h: CandidateHypothesis, r) -> bool:
        interp = r.interpretation or ""
        ev = r.evidence or {}
        if h.category == "target_derived_feature":
            if r.name == "feature_ablation":
                return ev.get("auc_drop", 0.0) > 0.10
            if r.name == "permutation_test":
                return interp == "target-linked"
        if h.category == "duplicate_contamination":
            return interp == "contamination_present"
        if h.category == "temporal_leakage":
            if r.name == "temporal_split_compare":
                return ev.get("gap_auc", 0.0) > 0.10
            if r.name == "feature_ablation":
                return ev.get("auc_drop", 0.0) > 0.10
        if h.category == "prediction_time_unavailable":
            return True
        if h.category == "preprocessing_leakage":
            return True
        return False

    def _describe_evidence(self, h: CandidateHypothesis, supporting) -> str:
        lines = []
        for r in supporting:
            ev = r.evidence or {}
            expl = ev.get("explanation", "")
            lines.append(f"[{r.name}] {expl}")
        return " ".join(lines)

    def _build(
        self,
        investigation: Investigation,
        h: CandidateHypothesis,
        severity: str,
        evidence_status: str,
        description: str,
        verified_by: str = "",
    ) -> Finding:
        return Finding(
            id=f"finding_{uuid.uuid4().hex[:8]}",
            investigation_id=investigation.id,
            category=h.category,
            severity=severity,
            evidence_status=evidence_status,
            title=h.title,
            description=description,
            affected_features=h.affected_features,
            code_paths=h.code_paths,
            supporting_experiments=h.experiments,
            correction=CORRECTIONS.get(h.category, ""),
            verification_status="supported_by_experiment" if verified_by else "unverified",
            verified_by=verified_by,
        )
