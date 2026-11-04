"""Agent 2 — Leakage Hypothesis Planner.

Turns detection signals into structured candidate hypotheses. If Gemini is
enabled it may propose additional hypotheses from sanitized metadata, but the
planner never depends on Gemini and always validates its output (only columns
from the real profile and only known categories are accepted).
"""
from __future__ import annotations

import uuid
from typing import Optional

from app.core.models import CandidateHypothesis
from app.detectors.leakage_detectors import DetectionSignals
from app.services.gemini_client import GeminiClient

CATEGORY_LABELS = {
    "target_derived_feature": "Target-derived / suspiciously predictive feature",
    "prediction_time_unavailable": "Feature unavailable at prediction time",
    "duplicate_contamination": "Train/test contamination via duplicates",
    "preprocessing_leakage": "Preprocessing fitted before split (static)",
    "evaluation_protocol_risk": "Missing or unsafe validation split (static)",
    "temporal_leakage": "Temporal leakage / future information",
    "feature_interaction_leakage": "Strong joint signal with weak individual signals",
}


def _new_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8]}"


def _make_hypothesis(
    investigation_id: str,
    category: str,
    title: str,
    description: str,
    rationale: str,
    affected: list[str],
    code_paths: list[str] | None = None,
    source: str = "rule",
) -> CandidateHypothesis:
    return CandidateHypothesis(
        id=_new_id("hyp"),
        investigation_id=investigation_id,
        category=category,
        title=title,
        description=description,
        rationale=rationale,
        affected_features=affected,
        code_paths=code_paths or [],
        source=source,
        confidence=0.3 if source == "rule" else 0.2,
    )


class HypothesisPlanner:
    def __init__(self, gemini: Optional[GeminiClient] = None) -> None:
        self.gemini = gemini

    def plan(
        self,
        investigation_id: str,
        profile,
        signals: DetectionSignals,
        code_findings: list,
        gemini_enabled: bool = False,
    ) -> tuple[list[CandidateHypothesis], bool]:
        """Return (hypotheses, gemini_used)."""
        hypotheses: list[CandidateHypothesis] = []
        gemini_used = False

        # 1) target-derived / suspicious features
        for feat, assoc in signals.high_association_features:
            hypotheses.append(
                _make_hypothesis(
                    investigation_id,
                    "target_derived_feature",
                    f"Feature `{feat}` is suspiciously predictive",
                    f"Association of `{feat}` with the target is {assoc:.2f}.",
                    (
                        "High feature-target association can indicate a derived feature "
                        "that leaks target information, or a legitimate strong predictor. "
                        "Requires ablation/permutation experiments to interpret."
                    ),
                    affected=[feat],
                )
            )

        # 2) contamination via conflicting duplicates
        if signals.conflicting_duplicates > 0:
            hypotheses.append(
                _make_hypothesis(
                    investigation_id,
                    "duplicate_contamination",
                    "Duplicate observations carry conflicting labels",
                    f"{signals.conflicting_duplicates} feature-identical rows disagree on the target.",
                    (
                        "Rows sharing all feature values but differing in the target indicate "
                        "train/test contamination or data-entry conflicts; can inflate apparent accuracy."
                    ),
                    affected=[signals.entity_column] if signals.entity_column else [],
                )
            )
        elif signals.duplicate_rows > 0:
            hypotheses.append(
                _make_hypothesis(
                    investigation_id,
                    "duplicate_contamination",
                    "Duplicate observations present",
                    f"{signals.duplicate_rows} feature-duplicate rows found.",
                    "Exact duplicates can cause leakage when split across train/test.",
                    affected=[signals.entity_column] if signals.entity_column else [],
                )
            )
        elif signals.repeated_entities > 0 and signals.entity_column:
            hypotheses.append(
                _make_hypothesis(
                    investigation_id,
                    "duplicate_contamination",
                    f"Entity `{signals.entity_column}` repeats across rows",
                    (
                        f"{signals.repeated_entities} entities occur more than once, covering "
                        f"{signals.repeated_entity_rows} rows."
                    ),
                    (
                        "A row-wise train/test split can place the same entity in both sets. "
                        "Use a group-aware split and compare metrics before treating this as "
                        "confirmed contamination."
                    ),
                    affected=[signals.entity_column],
                )
            )

        # 3) temporal leakage
        if signals.temporal and not signals.temporal.get("error"):
            candidates = [f for f, _ in signals.high_association_features]
            hypotheses.append(
                _make_hypothesis(
                    investigation_id,
                    "temporal_leakage",
                    "Time-based split may expose future information",
                    "A timestamp column exists; random splitting is compared against chronological splitting.",
                    (
                        "When data are time-ordered, splitting randomly can place future rows in "
                        "training. Chronological splits prevent this."
                    ),
                    affected=candidates,
                )
            )

        # 3b) interaction signals (review-only, never automatically confirmed)
        for fi, fj, best_single, joint in signals.interaction_features:
            hypotheses.append(
                _make_hypothesis(
                    investigation_id,
                    "feature_interaction_leakage",
                    f"Features `{fi}` and `{fj}` are only predictive together",
                    f"Joint nonlinear signal AUC≈{joint:.2f} vs best single-feature AUC≈{best_single:.2f}.",
                    (
                        "Neither feature is predictive on its own, but their combination "
                        "carries the signal. This can indicate an unusual/derived interaction "
                        "or simply a strong legitimate pattern — verify these columns' origin."
                    ),
                    affected=[fi, fj],
                )
            )

        # 4) prediction-time availability
        for feat in signals.prediction_time_unavailable:
            hypotheses.append(
                _make_hypothesis(
                    investigation_id,
                    "prediction_time_unavailable",
                    f"Feature `{feat}` is unavailable at prediction time",
                    f"`{feat}` is marked as not available when making real predictions.",
                    (
                        "This is a deployment hazard rather than proof of leakage: the feature "
                        "cannot be used in production and should be excluded from the model."
                    ),
                    affected=[feat],
                )
            )

        # 5) static preprocessing patterns
        preprocessing_findings = [
            f for f in code_findings if f.pattern == "preprocessing_before_split"
        ]
        if preprocessing_findings:
            paths = [f"line {f.line}" for f in preprocessing_findings]
            hypotheses.append(
                _make_hypothesis(
                    investigation_id,
                    "preprocessing_leakage",
                    "Training script shows suspected preprocessing-before-split patterns",
                    f"{len(preprocessing_findings)} suspicious static patterns found.",
                    (
                        "Fitting scalers/imputers/oversampling on the whole dataset before "
                        "splitting can leak training information into validation. Static only; "
                        "not treated as proven."
                    ),
                    affected=[],
                    code_paths=paths,
                )
            )

        split_findings = [f for f in code_findings if f.pattern == "missing_split"]
        if split_findings:
            hypotheses.append(
                _make_hypothesis(
                    investigation_id,
                    "evaluation_protocol_risk",
                    "Training script has no detectable validation split",
                    "Static analysis could not find a train/test or cross-validation split.",
                    (
                        "Training and evaluating on the same observations can produce "
                        "misleading metrics. Static analysis cannot see dynamically imported "
                        "or framework-specific split logic, so this remains a review flag."
                    ),
                    affected=[],
                    code_paths=[f"line {f.line}" for f in split_findings],
                )
            )

        # 6) Gemini enrichment (optional)
        if gemini_enabled and self.gemini and self.gemini.is_configured:
            extra = self._gemini_suggest(investigation_id, profile, signals)
            if extra is not None:
                gemini_used = True
                hypotheses.extend(extra)

        return hypotheses, gemini_used

    def _gemini_suggest(self, investigation_id: str, profile, signals: DetectionSignals):
        summary = self.gemini.sanitized_frame_summary(profile, signals) if self.gemini else {}
        prompt = (
            "You are a machine-learning leakage expert. Given sanitized dataset metadata, "
            "propose plausible data-leakage hypotheses.\n"
            "Rules:\n"
            "- Only use column names present in the provided 'name' fields.\n"
            "- Only use these category codes: "
            + ", ".join(CATEGORY_LABELS) + ".\n"
            "- Do NOT invent metrics or experimental results.\n"
            "- Return a JSON object with key 'hypotheses' = list of objects: "
            "{category, title, description, rationale, affected_features:[...]}.\n"
            f"Metadata:\n{json_dumps(summary)}"
        )
        try:
            resp = self.gemini.generate_structured(prompt)
        except Exception:  # noqa: BLE001
            return None
        if not isinstance(resp, dict) or not isinstance(resp.get("hypotheses"), list):
            return None
        out = []
        for item in resp["hypotheses"]:
            if not isinstance(item, dict):
                return None
            category = item.get("category")
            features = item.get("affected_features", [])
            if not isinstance(category, str) or category not in CATEGORY_LABELS:
                return None
            if not isinstance(features, list) or any(
                not isinstance(c, str) or profile.column(c) is None for c in features
            ):
                return None
            if any(not isinstance(item.get(key, ""), str) for key in ("title", "description", "rationale")):
                return None
            if category in ("target_derived_feature", "prediction_time_unavailable", "feature_interaction_leakage") and not features:
                return None
            if category == "prediction_time_unavailable" and not set(features).issubset(signals.prediction_time_unavailable):
                return None
            out.append(
                CandidateHypothesis(
                    id=_new_id("hyp"),
                    investigation_id=investigation_id,
                    category=category,
                    title=item.get("title", ""),
                    description=item.get("description", ""),
                    rationale=item.get("rationale", ""),
                    affected_features=list(dict.fromkeys(features)),
                    source="gemini",
                    confidence=0.2,
                )
            )
        return out


def json_dumps(obj) -> str:
    import json
    return json.dumps(obj, default=str)
