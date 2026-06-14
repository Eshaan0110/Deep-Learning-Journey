import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "03-Convolutional-Networks" / "3.1-conv-from-scratch"))
from conv2d import conv2d, max_pool2d, avg_pool2d, output_size, relu


class TestConv2d:
    def test_output_shape_no_padding(self):
        img    = np.ones((6, 6), dtype=np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)
        out = conv2d(img, kernel)
        assert out.shape == (4, 4)  # (6-3)//1 + 1 = 4

    def test_output_shape_with_padding(self):
        img    = np.ones((8, 8), dtype=np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)
        out = conv2d(img, kernel, padding=1)
        assert out.shape == (8, 8)  # same-size convolution

    def test_output_shape_with_stride(self):
        img    = np.ones((8, 8), dtype=np.float32)
        kernel = np.ones((3, 3), dtype=np.float32)
        out = conv2d(img, kernel, stride=2)
        assert out.shape == (3, 3)  # (8-3)//2 + 1 = 3

    def test_known_value_horizontal_edge(self):
        # 4×4 input, horizontal-edge kernel: output at (0,0) should be -2
        img = np.array([
            [1, 2, 3, 0],
            [0, 1, 2, 3],
            [3, 0, 1, 2],
            [2, 3, 0, 1],
        ], dtype=np.float32)
        kernel = np.array([
            [-1, -1, -1],
            [ 0,  0,  0],
            [ 1,  1,  1],
        ], dtype=np.float32)
        out = conv2d(img, kernel)
        # patch [0:3, 0:3] * kernel: -(1+2+3) + 0 + (3+0+1) = -6+4 = -2
        assert abs(out[0, 0] - (-2.0)) < 1e-5

    def test_identity_kernel_preserves_values(self):
        img = np.random.rand(5, 5).astype(np.float32)
        identity = np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.float32)
        out = conv2d(img, identity)
        # output should equal the center region of img
        np.testing.assert_allclose(out, img[1:4, 1:4], atol=1e-6)

    def test_uniform_kernel_sums_patch(self):
        img = np.ones((4, 4), dtype=np.float32)
        kernel = np.ones((2, 2), dtype=np.float32)
        out = conv2d(img, kernel)
        # each position sums a 2×2 patch of ones → 4
        assert np.all(out == 4.0)

    def test_output_size_helper(self):
        assert output_size(28, 5, padding=2) == 28
        assert output_size(28, 5, stride=2, padding=2) == 14


class TestMaxPool2d:
    def test_output_shape(self):
        fm  = np.ones((4, 4), dtype=np.float32)
        out = max_pool2d(fm)
        assert out.shape == (2, 2)

    def test_takes_maximum(self):
        fm = np.array([
            [1, 5, 2, 3],
            [4, 2, 6, 1],
            [3, 7, 1, 2],
            [0, 1, 4, 8],
        ], dtype=np.float32)
        out = max_pool2d(fm)
        assert out[0, 0] == 5.0  # max of [[1,5],[4,2]]
        assert out[0, 1] == 6.0  # max of [[2,3],[6,1]]
        assert out[1, 0] == 7.0  # max of [[3,7],[0,1]]
        assert out[1, 1] == 8.0  # max of [[1,2],[4,8]]

    def test_output_size_stride2(self):
        fm  = np.ones((8, 8), dtype=np.float32)
        out = max_pool2d(fm, pool_size=2, stride=2)
        assert out.shape == (4, 4)


class TestAvgPool2d:
    def test_output_shape(self):
        fm  = np.ones((4, 4), dtype=np.float32)
        out = avg_pool2d(fm)
        assert out.shape == (2, 2)

    def test_averages_correctly(self):
        fm = np.array([
            [1, 3, 2, 4],
            [5, 7, 6, 8],
            [9, 11, 10, 12],
            [13, 15, 14, 16],
        ], dtype=np.float32)
        out = avg_pool2d(fm)
        assert abs(out[0, 0] - 4.0) < 1e-5   # mean([1,3,5,7]) = 4
        assert abs(out[0, 1] - 5.0) < 1e-5   # mean([2,4,6,8]) = 5
        assert abs(out[1, 0] - 12.0) < 1e-5  # mean([9,11,13,15]) = 12
        assert abs(out[1, 1] - 13.0) < 1e-5  # mean([10,12,14,16]) = 13

    def test_uniform_input(self):
        fm  = np.full((4, 4), 7.0, dtype=np.float32)
        out = avg_pool2d(fm)
        assert np.all(np.abs(out - 7.0) < 1e-5)


class TestRelu:
    def test_negative_becomes_zero(self):
        x   = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        out = relu(x)
        np.testing.assert_array_equal(out, [0, 0, 0, 1, 3])

    def test_positive_unchanged(self):
        x = np.array([0.5, 2.0, 100.0])
        np.testing.assert_array_equal(relu(x), x)
