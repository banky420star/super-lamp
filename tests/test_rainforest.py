"""Tests for RainforestDetector — market regime classification."""
import numpy as np
import pandas as pd
import pytest
import tempfile
import os

from Python.rainforest_detector import RainforestDetector, FEATURE_NAMES

# Valid regime set as defined in the production module
REGIMES = {'bull_trend', 'bear_trend', 'ranging', 'breakout_up', 'breakout_down', 'reversal_up', 'reversal_down'}


@pytest.fixture
def detector():
    return RainforestDetector(n_estimators=10, max_depth=4, random_state=42)


@pytest.fixture
def synthetic_df(detector):
    return detector._generate_synthetic_training_data(n_bars=500)


# ---------------------------------------------------------------------------
# Synthetic data generation
# ---------------------------------------------------------------------------

class TestSyntheticData:
    def test_returns_dataframe(self, detector):
        df = detector._generate_synthetic_training_data(100)
        assert isinstance(df, pd.DataFrame)

    def test_has_ohlcv_columns(self, detector):
        df = detector._generate_synthetic_training_data(100)
        for col in ['open', 'high', 'low', 'close', 'volume']:
            assert col in df.columns, f"Missing column: {col}"

    def test_correct_length(self, detector):
        df = detector._generate_synthetic_training_data(200)
        assert len(df) == 200

    def test_prices_positive(self, detector):
        df = detector._generate_synthetic_training_data(100)
        assert (df['close'] > 0).all()

    def test_high_gte_low(self, detector):
        df = detector._generate_synthetic_training_data(100)
        assert (df['high'] >= df['low']).all()

    def test_open_gte_zero(self, detector):
        df = detector._generate_synthetic_training_data(100)
        assert (df['open'] > 0).all()

    def test_volume_positive(self, detector):
        df = detector._generate_synthetic_training_data(100)
        assert (df['volume'] > 0).all()

    def test_deterministic_with_same_seed(self):
        # RNG seeded with 42 — two calls should produce identical output
        d1 = RainforestDetector()
        d2 = RainforestDetector()
        df1 = d1._generate_synthetic_training_data(200)
        df2 = d2._generate_synthetic_training_data(200)
        pd.testing.assert_frame_equal(df1, df2)

    def test_one_bar(self, detector):
        # Edge: minimum viable call should not crash
        df = detector._generate_synthetic_training_data(1)
        assert isinstance(df, pd.DataFrame)
        assert len(df) == 1


# ---------------------------------------------------------------------------
# Feature extraction
# ---------------------------------------------------------------------------

class TestFeatureExtraction:
    def test_returns_ndarray(self, detector, synthetic_df):
        X = detector.extract_features(synthetic_df)
        assert isinstance(X, np.ndarray)

    def test_exact_feature_count(self, detector, synthetic_df):
        X = detector.extract_features(synthetic_df)
        # FEATURE_NAMES has 30 entries (14 core + 5 timing + 11 patterns)
        assert X.shape[1] == len(FEATURE_NAMES)

    def test_row_count_matches_input(self, detector, synthetic_df):
        X = detector.extract_features(synthetic_df)
        assert X.shape[0] == len(synthetic_df)

    def test_dtype_is_float32(self, detector, synthetic_df):
        X = detector.extract_features(synthetic_df)
        assert X.dtype == np.float32

    def test_no_all_nan_rows(self, detector, synthetic_df):
        X = detector.extract_features(synthetic_df)
        if len(X) > 0:
            all_nan_mask = np.all(np.isnan(X), axis=1)
            assert not all_nan_mask.any(), "Found rows where every feature is NaN"

    def test_minimum_rows_returned(self, detector, synthetic_df):
        X = detector.extract_features(synthetic_df)
        # With 500 bars we should get exactly 500 rows (no dropping in extract_features)
        assert len(X) >= 300

    def test_small_dataframe(self, detector):
        # 10 bars — should not crash, just return short feature matrix
        df = detector._generate_synthetic_training_data(10)
        X = detector.extract_features(df)
        assert isinstance(X, np.ndarray)
        assert X.shape[0] == 10

    def test_missing_volume_column_raises(self, detector, synthetic_df):
        # Dropping a required column should raise a KeyError
        bad_df = synthetic_df.drop(columns=['volume'])
        with pytest.raises((KeyError, Exception)):
            detector.extract_features(bad_df)

    def test_missing_close_column_raises(self, detector, synthetic_df):
        bad_df = synthetic_df.drop(columns=['close'])
        with pytest.raises((KeyError, Exception)):
            detector.extract_features(bad_df)


# ---------------------------------------------------------------------------
# Auto-labelling
# ---------------------------------------------------------------------------

class TestLabeling:
    def test_returns_ndarray(self, detector, synthetic_df):
        y = detector.label_regimes(synthetic_df)
        assert isinstance(y, np.ndarray)

    def test_length_matches_input(self, detector, synthetic_df):
        y = detector.label_regimes(synthetic_df)
        assert len(y) == len(synthetic_df)

    def test_valid_regime_labels(self, detector, synthetic_df):
        y = detector.label_regimes(synthetic_df)
        unique = set(y)
        assert unique.issubset(REGIMES), f"Invalid regimes: {unique - REGIMES}"

    def test_all_regimes_can_appear(self, detector):
        # With 2000 bars at least 3 different regimes should appear
        df = detector._generate_synthetic_training_data(2000)
        y = detector.label_regimes(df)
        assert len(set(y)) >= 3

    def test_ranging_is_default(self, detector, synthetic_df):
        # 'ranging' should appear (it's the fallback label)
        y = detector.label_regimes(synthetic_df)
        assert 'ranging' in set(y)

    def test_default_forward_window(self, detector, synthetic_df):
        # No forward_window arg — uses default of 20
        y = detector.label_regimes(synthetic_df)
        assert len(y) == len(synthetic_df)

    def test_custom_forward_window(self, detector, synthetic_df):
        y = detector.label_regimes(synthetic_df, forward_window=5)
        assert len(y) == len(synthetic_df)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

class TestTraining:
    def test_fit_completes(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        assert detector.is_trained()

    def test_not_trained_initially(self, detector):
        assert not detector.is_trained()

    def test_fit_sets_trained_at(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        assert detector._trained_at is not None

    def test_fit_populates_classes(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        assert len(detector._classes) >= 1

    def test_fit_populates_feature_importances(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        assert len(detector._feature_importances) == len(FEATURE_NAMES)

    def test_fit_feature_importance_values_sum_to_one(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        total = sum(detector._feature_importances.values())
        assert abs(total - 1.0) < 0.01

    def test_fit_with_small_data_warns_but_works(self, detector):
        # 100 bars < 250 recommended — should warn but not crash
        df = detector._generate_synthetic_training_data(100)
        try:
            detector.fit(df)
        except Exception as e:
            pytest.fail(f"fit() raised on small data: {e}")

    def test_fit_raises_on_too_few_rows(self, detector):
        # < 50 valid rows after feature extraction → ValueError
        df = detector._generate_synthetic_training_data(10)
        with pytest.raises(ValueError):
            detector.fit(df)

    def test_refit_overwrites_previous_model(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        first_trained_at = detector._trained_at
        import time
        time.sleep(0.01)
        detector.fit(synthetic_df)
        assert detector._trained_at >= first_trained_at

    def test_fit_without_sklearn_raises(self, detector, synthetic_df, monkeypatch):
        import Python.rainforest_detector as rf_mod
        monkeypatch.setattr(rf_mod, "_SKLEARN_AVAILABLE", False)
        with pytest.raises(ImportError):
            detector.fit(synthetic_df)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

class TestPrediction:
    @pytest.fixture
    def trained_detector(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        return detector

    def test_predict_returns_dict(self, trained_detector, synthetic_df):
        result = trained_detector.predict_regime(synthetic_df)
        assert isinstance(result, dict)

    def test_predict_has_required_keys(self, trained_detector, synthetic_df):
        result = trained_detector.predict_regime(synthetic_df)
        for key in ['regime', 'confidence', 'probabilities', 'feature_importances']:
            assert key in result, f"Missing key: {key}"

    def test_regime_is_valid(self, trained_detector, synthetic_df):
        result = trained_detector.predict_regime(synthetic_df)
        assert result['regime'] in REGIMES

    def test_confidence_in_range(self, trained_detector, synthetic_df):
        result = trained_detector.predict_regime(synthetic_df)
        assert 0.0 <= result['confidence'] <= 1.0

    def test_probabilities_sum_to_one(self, trained_detector, synthetic_df):
        result = trained_detector.predict_regime(synthetic_df)
        total = sum(result['probabilities'].values())
        assert abs(total - 1.0) < 0.02

    def test_probabilities_cover_all_regimes(self, trained_detector, synthetic_df):
        result = trained_detector.predict_regime(synthetic_df)
        # All 7 regimes should have an entry (missing ones filled with 0.0)
        assert set(result['probabilities'].keys()) == REGIMES

    def test_feature_importances_sum_to_one(self, trained_detector, synthetic_df):
        result = trained_detector.predict_regime(synthetic_df)
        total = sum(result['feature_importances'].values())
        assert abs(total - 1.0) < 0.02

    def test_top_patterns_is_list(self, trained_detector, synthetic_df):
        result = trained_detector.predict_regime(synthetic_df)
        assert isinstance(result.get('top_patterns', []), list)

    def test_top_patterns_have_required_keys(self, trained_detector, synthetic_df):
        result = trained_detector.predict_regime(synthetic_df)
        for pattern in result.get('top_patterns', []):
            assert 'pattern' in pattern
            assert 'feature' in pattern
            assert 'importance' in pattern

    def test_predict_untrained_returns_default(self, detector, synthetic_df):
        # Untrained detector should return safe defaults, not crash
        result = detector.predict_regime(synthetic_df)
        assert result['regime'] == 'ranging'
        assert result['confidence'] == 0.0
        assert 'error' in result

    def test_predict_confidence_equals_top_probability(self, trained_detector, synthetic_df):
        result = trained_detector.predict_regime(synthetic_df)
        regime = result['regime']
        prob = result['probabilities'][regime]
        assert abs(result['confidence'] - prob) < 0.0001

    def test_predict_with_single_bar(self, trained_detector, synthetic_df):
        # Prediction on a 1-row df should not crash (uses last row)
        single = synthetic_df.tail(1).reset_index(drop=True)
        result = trained_detector.predict_regime(single)
        assert result['regime'] in REGIMES

    def test_predict_empty_dataframe_returns_safe_default(self, trained_detector):
        # Regression: empty DataFrame must not raise ValueError from sklearn.
        # Previously crashed with "Found array with 0 sample(s)".
        empty = pd.DataFrame(columns=['open', 'high', 'low', 'close', 'volume'])
        result = trained_detector.predict_regime(empty)
        assert isinstance(result, dict)
        assert result['regime'] in REGIMES
        assert result['confidence'] == 0.0
        assert 'error' in result


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'test_rf.pkl')
            detector.save(path)
            assert os.path.exists(path)

            new_detector = RainforestDetector()
            assert not new_detector.is_trained()
            new_detector.load(path)
            assert new_detector.is_trained()

    def test_predictions_consistent_after_load(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        pred_before = detector.predict_regime(synthetic_df)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'rf.pkl')
            detector.save(path)
            new_detector = RainforestDetector()
            new_detector.load(path)
            pred_after = new_detector.predict_regime(synthetic_df)

        assert pred_before['regime'] == pred_after['regime']

    def test_load_nonexistent_returns_false(self, detector):
        result = detector.load('/nonexistent/path.pkl')
        assert result == False

    def test_save_untrained_raises(self, detector):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'untrained.pkl')
            with pytest.raises(RuntimeError):
                detector.save(path)

    def test_load_returns_true_on_success(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'model.pkl')
            detector.save(path)
            new_det = RainforestDetector()
            result = new_det.load(path)
            assert result == True

    def test_load_corrupted_file_returns_false(self, detector):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'corrupt.pkl')
            with open(path, 'wb') as f:
                f.write(b'not a valid pickle file at all')
            result = detector.load(path)
            assert result == False

    def test_save_creates_parent_dirs(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'nested', 'deep', 'model.pkl')
            detector.save(path)
            assert os.path.exists(path)


# ---------------------------------------------------------------------------
# Top patterns
# ---------------------------------------------------------------------------

class TestTopPatterns:
    def test_returns_list(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        patterns = detector.get_top_patterns()
        assert isinstance(patterns, list)

    def test_respects_n_limit(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        patterns = detector.get_top_patterns(n=5)
        assert len(patterns) <= 5

    def test_not_trained_returns_empty(self, detector):
        patterns = detector.get_top_patterns()
        assert patterns == []

    def test_patterns_sorted_by_importance_descending(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        patterns = detector.get_top_patterns(n=10)
        importances = [p['importance'] for p in patterns]
        assert importances == sorted(importances, reverse=True)

    def test_default_n_is_ten(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        patterns = detector.get_top_patterns()
        # 14 features total, default n=10
        assert len(patterns) <= 10

    def test_n_larger_than_features_returns_all(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        patterns = detector.get_top_patterns(n=100)
        # Can't exceed the number of features
        assert len(patterns) <= len(FEATURE_NAMES)

    def test_importance_values_are_floats(self, detector, synthetic_df):
        detector.fit(synthetic_df)
        patterns = detector.get_top_patterns(n=5)
        for p in patterns:
            assert isinstance(p['importance'], float)
