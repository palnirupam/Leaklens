def main() -> None:
    import sys
    sys.path.insert(0, "backend")
    import tempfile, os
    from app.data.synthetic import LEAK_CATEGORIES, generate_case
    from app.core.models import InvestigationConfig
    from app.services.orchestrator import InvestigationOrchestrator

    orch = InvestigationOrchestrator()  # offline, no Gemini

    for cat in LEAK_CATEGORIES:
        case = generate_case(cat, seed=5)
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "data.csv")
            case.df.to_csv(path, index=False)
            cfg = InvestigationConfig(
                dataset_path=path,
                target_column=case.target,
                timestamp_column=case.timestamp_column,
                entity_column=case.entity_column,
                unavailable_at_prediction=case.unavailable_at_prediction,
                random_seed=5,
                source_filename=case.name,
            )
            inv = orch.create(cfg)
            orch.run(inv)
            cats = sorted({f.category for f in inv.findings})
            statuses = sorted({f.evidence_status for f in inv.findings})
            gt = str(case.leaked_features or "none")
            print(
                f"{cat:28s} gt={gt:20s} "
                f"findings={cats} statuses={statuses} exp={len(inv.results)}"
            )
    print("PIPELINE_OK")


if __name__ == "__main__":
    main()
