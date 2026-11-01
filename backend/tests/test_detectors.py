from app.data.synthetic import generate_case
from app.detectors.dataset_investigator import investigate
from app.detectors.leakage_detectors import detect_signals


def test_target_derived_signal():
    c = generate_case("target_derived_feature", seed=5)
    prof = investigate(c.df, c.target)
    sig = detect_signals(c.df, prof, c.target)
    feats = [f for f, _ in sig.high_association_features]
    assert "f_derived" in feats


def test_duplicate_conflict_signal():
    c = generate_case("duplicate_contamination", seed=5)
    prof = investigate(c.df, c.target)
    sig = detect_signals(c.df, prof, c.target)
    assert sig.conflicting_duplicates > 0


def test_temporal_signal_association():
    c = generate_case("temporal_leakage", seed=5)
    prof = investigate(c.df, c.target)
    sig = detect_signals(c.df, prof, c.target, timestamp_column=c.timestamp_column)
    feats = [f for f, _ in sig.high_association_features]
    assert "future_outcome_hint" in feats


def test_clean_has_no_signals():
    c = generate_case("clean", seed=5)
    prof = investigate(c.df, c.target)
    sig = detect_signals(c.df, prof, c.target)
    assert sig.high_association_features == []
    assert sig.conflicting_duplicates == 0
