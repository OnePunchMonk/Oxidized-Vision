"""
OxidizedVision — Vision-specific inference kernels.

Generic ONNX/graph optimizers (constant folding, Conv+BN fusion — see
`optimize.py`) don't touch the two costs that dominate ViT-family inference:
running full self-attention over every patch token, and carrying every
token through every block even when many are redundant (e.g. large regions
of flat background/sky in an image). This module implements
`TokenMerge`, a `torch.nn.Module` you insert into a ViT-style model
(between transformer blocks) *before* export, exportable to ONNX just like
any other layer — unlike a Rust/graph-level rewrite, it needs no custom
ONNX op.

This is the same algorithm as `kernel_vision::token_merge` in the Rust
runtime (see `rust_runtime/crates/kernel_vision`), reimplemented in
vectorized PyTorch so it can run inside a model's forward pass and survive
`torch.onnx.export`. Use the Rust version instead when merging needs to
happen outside a traced/exported model (e.g. as a serving-side
preprocessing step ahead of a non-ViT runner).

Reference: "Token Merging: Your ViT But Faster" (Bolya et al., ICLR 2023).
"""

import torch
import torch.nn as nn


def token_merge(tokens: torch.Tensor, r: int) -> torch.Tensor:
    """Merge the `r` most-similar token pairs, batched.

    Args:
        tokens: `[batch, n, dim]` sequence of tokens (e.g. ViT patch
            embeddings, with or without a leading class token — if a class
            token is present, slice it out before calling and concatenate
            it back after, since it should never be merged away).
        r: number of tokens to remove via merging. Must satisfy
            `0 <= r <= n // 2`.

    Returns:
        `[batch, n - r, dim]` tensor. Token order is: unmerged even-index
        tokens (in original relative order), then all odd-index tokens
        (unmerged ones unchanged, merged-into ones averaged) — the same
        merge semantics as `kernel_vision::token_merge`, though exact row
        order isn't required to match since both are just token sets fed
        into permutation-invariant attention layers.
    """
    b, n, d = tokens.shape
    if r == 0 or n < 2:
        return tokens
    if not (0 <= r <= n // 2):
        raise ValueError(f"token_merge: r={r} must be in [0, n // 2] for n={n}")

    a, b_tok = tokens[:, 0::2, :], tokens[:, 1::2, :]  # [batch, n//2(+), d]
    a_norm = torch.nn.functional.normalize(a, dim=-1)
    b_norm = torch.nn.functional.normalize(b_tok, dim=-1)

    # Cosine similarity of every A token to every B token: [batch, na, nb].
    sim = a_norm @ b_norm.transpose(-1, -2)
    best_sim, best_b = sim.max(dim=-1)  # [batch, na] each

    # Rank each batch's A tokens by their best-match similarity, merge the
    # top r (many-to-one into a B destination is allowed, matching the
    # reference ToMe algorithm and `kernel_vision::token_merge`).
    _, order = best_sim.sort(dim=-1, descending=True)
    merge_a_idx = order[:, :r]  # [batch, r] — indices into the A axis
    keep_a_idx, _ = order[:, r:].sort(dim=-1)  # kept A tokens, original relative order

    dst_for_merge = torch.gather(best_b, 1, merge_a_idx)  # [batch, r]
    src_vals = torch.gather(a, 1, merge_a_idx.unsqueeze(-1).expand(-1, -1, d))

    nb = b_tok.shape[1]
    add = torch.zeros_like(b_tok).scatter_add(
        1, dst_for_merge.unsqueeze(-1).expand(-1, -1, d), src_vals
    )
    counts = torch.zeros(b, nb, device=tokens.device, dtype=tokens.dtype).scatter_add(
        1, dst_for_merge, torch.ones_like(dst_for_merge, dtype=tokens.dtype)
    )
    merged_b = (b_tok + add) / (counts.unsqueeze(-1) + 1.0)

    kept_a = torch.gather(a, 1, keep_a_idx.unsqueeze(-1).expand(-1, -1, d))
    return torch.cat([kept_a, merged_b], dim=1)


class TokenMerge(nn.Module):
    """Drop-in module: merges `r` tokens out of its input sequence.

    Insert between transformer blocks in a ViT-style model, e.g.:

    ```python
    self.blocks = nn.ModuleList([
        block1, TokenMerge(r=8), block2, TokenMerge(r=8), block3, ...
    ])
    ```

    Reduces the token count seen by every subsequent block (and therefore
    their O(n^2) attention cost and O(n) MLP cost), trading a small amount
    of accuracy for throughput — no retraining required, though fine-tuning
    after insertion typically recovers most of the gap (see the ToMe paper).
    """

    def __init__(self, r: int):
        super().__init__()
        self.r = r

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return token_merge(tokens, self.r)

    def extra_repr(self) -> str:
        return f"r={self.r}"
