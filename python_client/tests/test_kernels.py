"""Tests for the vision-specific kernels module (token merging)."""

import pytest
import torch

from oxidizedvision.kernels import TokenMerge, token_merge


class TestTokenMerge:
    def test_output_shape(self):
        x = torch.randn(2, 16, 8)
        out = token_merge(x, r=4)
        assert out.shape == (2, 12, 8)

    def test_r_zero_is_identity(self):
        x = torch.randn(3, 10, 4)
        assert torch.equal(token_merge(x, r=0), x)

    def test_identical_tokens_merge_without_changing_value(self):
        # Duplicate every token so every merge averages two identical
        # vectors — the merged result must equal the original value.
        base = torch.randn(1, 8, 5)
        x = base.repeat_interleave(1, dim=1)
        x[:, 1::2, :] = x[:, 0::2, :]  # make each (even, odd) pair identical
        out = token_merge(x, r=4)
        expected_sorted, _ = base[:, 0::2, :].sort(dim=1)
        got_sorted, _ = out.sort(dim=1)
        assert torch.allclose(got_sorted, expected_sorted, atol=1e-5)

    def test_r_out_of_range_raises(self):
        x = torch.randn(1, 8, 4)
        with pytest.raises(ValueError):
            token_merge(x, r=5)  # n // 2 == 4, so 5 is invalid

    def test_module_matches_functional(self):
        x = torch.randn(2, 12, 6)
        m = TokenMerge(r=3)
        assert torch.equal(m(x), token_merge(x, 3))

    def test_traceable_and_exportable_to_onnx(self):
        import io

        x = torch.randn(1, 16, 8)
        m = TokenMerge(r=4)
        traced = torch.jit.trace(m, x)
        assert torch.allclose(traced(x), m(x))

        buf = io.BytesIO()
        torch.onnx.export(m, x, buf, opset_version=14, input_names=["tokens"], output_names=["out"])
        assert buf.tell() > 0
