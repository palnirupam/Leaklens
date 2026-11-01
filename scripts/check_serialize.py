def main() -> None:
    import sys, json
    sys.path.insert(0, "backend")
    from app.core.store import from_payload, _json_default
    from app.core.models import Investigation, InvestigationConfig, DatasetProfile, ColumnProfile

    prof = DatasetProfile(
        row_count=10, column_count=2, duplicate_rows=0, duplicate_rows_pct=0.0,
        columns=[ColumnProfile(name="a", dtype="numeric", null_count=0, null_pct=0.0, n_unique=5, is_numeric=True, is_categorical=False, is_datetime=False, target_association=0.4)],
        target_column="y",
    )
    inv = Investigation(id="inv_x", config=InvestigationConfig(dataset_path="d.csv", target_column="y"))
    inv.profile = prof
    s = json.dumps(inv.__dict__, default=_json_default)
    print("serialized columns type:", type(json.loads(s)["profile"]["columns"]))
    print("first col item type:", type(json.loads(s)["profile"]["columns"][0]))
    back = from_payload(json.loads(s))
    print("reconstructed profile rows:", back.profile.row_count, "col:", back.profile.columns[0].name)
    print("SERIALIZE_OK")


if __name__ == "__main__":
    main()
