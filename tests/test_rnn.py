import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "04-Sequence-Models" / "4.1-rnn-from-scratch"))
from rnn import init_params, rnn_forward, rnn_backward, sgd_update


@pytest.fixture
def params():
    return init_params(hidden_size=16)


class TestRNNForward:
    def test_returns_scalar_prediction(self, params):
        x_seq = np.random.randn(10).astype(np.float32)
        y_pred, hs = rnn_forward(x_seq, params)
        assert isinstance(y_pred, float)

    def test_hidden_state_count(self, params):
        # hs should have seq_len + 1 entries (h_0 through h_T)
        x_seq = np.random.randn(7).astype(np.float32)
        _, hs = rnn_forward(x_seq, params)
        assert len(hs) == 8  # 7 steps + initial h_0

    def test_hidden_state_shape(self, params):
        x_seq = np.random.randn(5).astype(np.float32)
        _, hs = rnn_forward(x_seq, params)
        hidden_size = params["W_hh"].shape[0]
        for h in hs:
            assert h.shape == (hidden_size,)

    def test_zero_input_gives_finite_output(self, params):
        x_seq = np.zeros(20, dtype=np.float32)
        y_pred, _ = rnn_forward(x_seq, params)
        assert np.isfinite(y_pred)

    def test_hidden_state_bounded_by_tanh(self, params):
        x_seq = np.random.randn(20).astype(np.float32) * 100  # extreme inputs
        _, hs = rnn_forward(x_seq, params)
        for h in hs:
            assert np.all(np.abs(h) <= 1.0 + 1e-6)  # tanh output in [-1, 1]


class TestRNNBackward:
    def test_gradient_shapes(self, params):
        x_seq = np.random.randn(10).astype(np.float32)
        y_pred, hs = rnn_forward(x_seq, params)
        grads, _ = rnn_backward(x_seq, 0.0, y_pred, hs, params)

        assert grads["W_xh"].shape == params["W_xh"].shape
        assert grads["W_hh"].shape == params["W_hh"].shape
        assert grads["b_h"].shape  == params["b_h"].shape
        assert grads["W_hy"].shape == params["W_hy"].shape
        assert grads["b_y"].shape  == params["b_y"].shape

    def test_gradients_clipped(self, params):
        x_seq = np.random.randn(10).astype(np.float32)
        y_pred, hs = rnn_forward(x_seq, params)
        grads, _ = rnn_backward(x_seq, 0.0, y_pred, hs, params, clip=1.0)

        for g in grads.values():
            assert np.all(np.abs(g) <= 1.0 + 1e-6)

    def test_loss_decreases_after_one_step(self, params):
        np.random.seed(99)
        x_seq  = np.random.randn(15).astype(np.float32)
        y_true = 0.5

        y_pred, hs = rnn_forward(x_seq, params)
        loss_before = (y_pred - y_true) ** 2

        grads, _ = rnn_backward(x_seq, y_true, y_pred, hs, params)
        sgd_update(params, grads, lr=0.1)

        y_after, _ = rnn_forward(x_seq, params)
        loss_after = (y_after - y_true) ** 2

        assert loss_after < loss_before

    def test_zero_loss_gives_zero_gradients(self, params):
        x_seq = np.random.randn(10).astype(np.float32)
        y_pred, hs = rnn_forward(x_seq, params)
        # Use y_true == y_pred so loss = 0 and all grads should be 0
        grads, loss = rnn_backward(x_seq, y_pred, y_pred, hs, params)
        assert loss == 0.0
        for g in grads.values():
            np.testing.assert_array_equal(g, np.zeros_like(g))
