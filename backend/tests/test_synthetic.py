from app.data.synthetic import LEAK_CATEGORIES, generate_case
from app.core.models import LEAKAGE_CATEGORIES


def test_clean_has_no_leak():
    c = generate_case("clean", seed=0)
    assert c.leaked_features == []
    assert c.category == "clean"
    assert c.df.shape[1] > 4


def test_target_derived_flags_feature():
    c = generate_case("target_derived_feature", seed=1)
    assert c.leaked_features == ["f_derived"]
    assert "f_derived" in c.df.columns


def test_temporal_has_timestamp_and_hint():
    c = generate_case("temporal_leakage", seed=2)
    assert c.timestamp_column == "timestamp"
    assert "future_outcome_hint" in c.leaked_features


def test_duplicate_generates_conflicts():
    c = generate_case("duplicate_contamination", seed=3)
    assert len(c.df) > 600  # duplicates appended
    assert c.category == "duplicate_contamination"


def test_all_categories_generate():
    for cat in LEAK_CATEGORIES:
        c = generate_case(cat, seed=4)
        assert c.df.shape[0] > 100


def test_public_category_names_match_runtime_categories():
    assert "prediction_time_unavailable" in LEAKAGE_CATEGORIES
    assert "duplicate_contamination" in LEAKAGE_CATEGORIES
    assert "evaluation_protocol_risk" in LEAKAGE_CATEGORIES
