def main() -> None:
    import sys
    sys.path.insert(0, "backend")
    from app.data.synthetic import LEAK_CATEGORIES, generate_case, generate_benchmark_cases

    for cat in LEAK_CATEGORIES:
        case = generate_case(cat, seed=1)
        print(
            f"{cat:28s} shape={case.df.shape} "
            f"leaked={case.leaked_features} ts={case.timestamp_column} "
            f"unavail={case.unavailable_at_prediction}"
        )
    cases = generate_benchmark_cases(per_category=2, seed_offset=0)
    print("benchmark cases:", len(cases))
    print("SYNTHETIC_OK")


if __name__ == "__main__":
    main()
