import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "00-Foundation" / "numpy-from-scratch"))
from loss_function import mse_loss, mae_loss, binary_cross_entropy, categorical_cross_entropy


class TestMSE:
    def test_perfect_predictions_zero(self):
        assert mse_loss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0.0

    def test_unit_error(self):
        # mean((0-1)^2, (0-1)^2) = 1.0
        result = mse_loss(np.array([0.0, 0.0]), np.array([1.0, 1.0]))
        assert abs(result - 1.0) < 1e-7

    def test_asymmetric(self):
        # mean((1-3)^2, (2-2)^2) = mean(4, 0) = 2.0
        result = mse_loss(np.array([1.0, 2.0]), np.array([3.0, 2.0]))
        assert abs(result - 2.0) < 1e-7

    def test_nonnegative(self):
        y_true = np.random.randn(50)
        y_pred = np.random.randn(50)
        assert mse_loss(y_true, y_pred) >= 0.0


class TestMAE:
    def test_perfect_predictions_zero(self):
        assert mae_loss(np.array([1.0, 2.0, 3.0]), np.array([1.0, 2.0, 3.0])) == 0.0

    def test_unit_error(self):
        result = mae_loss(np.array([0.0, 0.0]), np.array([1.0, -1.0]))
        assert abs(result - 1.0) < 1e-7

    def test_nonnegative(self):
        y_true = np.random.randn(50)
        y_pred = np.random.randn(50)
        assert mae_loss(y_true, y_pred) >= 0.0

    def test_mae_less_sensitive_to_outliers_than_mse(self):
        y_true = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
        y_pred = np.array([0.0, 0.0, 0.0, 0.0, 10.0])  # one large outlier
        assert mae_loss(y_true, y_pred) < mse_loss(y_true, y_pred)


class TestBCE:
    def test_near_perfect_predictions_low_loss(self):
        y_true = np.array([1.0, 0.0, 1.0])
        y_pred = np.array([0.999, 0.001, 0.999])
        assert binary_cross_entropy(y_true, y_pred) < 0.01

    def test_uniform_prediction_equals_log2(self):
        # BCE at p=0.5 for all: -[y*log(0.5) + (1-y)*log(0.5)] = log(2) per sample
        y_true = np.array([1.0, 0.0, 1.0, 0.0])
        y_pred = np.array([0.5, 0.5, 0.5, 0.5])
        expected = np.log(2)
        assert abs(binary_cross_entropy(y_true, y_pred) - expected) < 1e-5

    def test_nonnegative(self):
        y_true = np.array([1.0, 0.0, 1.0])
        y_pred = np.array([0.7, 0.3, 0.8])
        assert binary_cross_entropy(y_true, y_pred) >= 0.0

    def test_no_log_zero_crash(self):
        y_true = np.array([1.0, 0.0])
        y_pred = np.array([1.0, 0.0])  # clipped internally to avoid log(0)
        result = binary_cross_entropy(y_true, y_pred)
        assert np.isfinite(result)


class TestCCE:
    def test_perfect_one_hot_low_loss(self):
        y_true = np.array([[1, 0, 0], [0, 1, 0]], dtype=float)
        y_pred = np.array([[0.999, 0.0005, 0.0005], [0.0005, 0.999, 0.0005]])
        assert categorical_cross_entropy(y_true, y_pred) < 0.01

    def test_nonnegative(self):
        y_true = np.array([[1, 0, 0], [0, 0, 1]], dtype=float)
        y_pred = np.array([[0.6, 0.3, 0.1], [0.2, 0.3, 0.5]])
        assert categorical_cross_entropy(y_true, y_pred) >= 0.0

    def test_no_log_zero_crash(self):
        y_true = np.eye(3)
        y_pred = np.eye(3)
        result = categorical_cross_entropy(y_true, y_pred)
        assert np.isfinite(result)
