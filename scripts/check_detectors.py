def main() -> None:
    import sys
    sys.path.insert(0, "backend")
    import pandas as pd
    from app.data.synthetic import generate_case
    from app.detectors.dataset_investigator import investigate
    from app.detectors.leakage_detectors import detect_signals
    from app.detectors.ast_analyzer import analyze_script

    # AST analyzer
    leaky = (
        "import pandas as pd\n"
        "from sklearn.preprocessing import StandardScaler\n"
        "from sklearn.model_selection import train_test_split\n"
        "df = pd.read_csv('d.csv')\n"
        "X = df.drop('y', axis=1)\n"
        "y = df['y']\n"
        "X = StandardScaler().fit_transform(X)\n"   # fitted before split -> suspicious
        "X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)\n"
    )
    analysis = analyze_script(leaky)
    print("AST findings:", len(analysis.findings), "saw_split:", analysis.saw_split)
    for f in analysis.findings:
        print("   -", f.line, f.pattern, f.severity, f.message[:60])

    # DataFrame detectors for each synthetic category
    for cat in ["clean", "target_derived_feature", "temporal_leakage", "duplicate_contamination"]:
        case = generate_case(cat, seed=3)
        df = case.df
        prof = investigate(df, case.target)
        sig = detect_signals(
            df, prof, case.target,
            timestamp_column=case.timestamp_column,
            entity_column=case.entity_column,
            unavailable_at_prediction=case.unavailable_at_prediction,
        )
        print(f"\n[{cat}]")
        print("   high_assoc:", sig.high_association_features[:3])
        print("   dup_rows:", sig.duplicate_rows, "conflicts:", sig.conflicting_duplicates)
        print("   temporal:", sig.temporal)
    print("DETECTORS_OK")


if __name__ == "__main__":
    main()
