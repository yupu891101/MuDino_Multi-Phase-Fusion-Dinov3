import math

import torch
from torch import nn


def mix_scaled_dot_product_attention(
    a_qk: torch.Tensor,
    b_qk: torch.Tensor,
    a_v: torch.Tensor,
    b_v: torch.Tensor,
    attn_mask: torch.Tensor | None = None,
    dropout_p: float = 0.0,
    scale: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_len, b_len = a_qk.size(-2), b_qk.size(-2)
    scale_factor = 1 / math.sqrt(a_qk.size(-1)) if scale is None else scale
    attn_bias = torch.zeros(a_len, b_len, device=a_qk.device, dtype=a_qk.dtype)

    if attn_mask is not None:
        if attn_mask.dtype == torch.bool:
            attn_mask = attn_mask.clone().masked_fill_(attn_mask, float("-inf"))
        else:
            attn_bias += attn_mask

    attn_weight = a_qk @ b_qk.transpose(-2, -1) * scale_factor
    attn_weight += attn_bias
    attn_weight_a = torch.softmax(attn_weight, dim=-1)
    attn_weight_b = torch.softmax(attn_weight, dim=-2).transpose(-2, -1)
    attn_weight_a = torch.dropout(attn_weight_a, dropout_p, train=True)
    attn_weight_b = torch.dropout(attn_weight_b, dropout_p, train=True)
    return attn_weight_a @ b_v, attn_weight_b @ a_v


def mix_attention_froward(
    a_qk: torch.Tensor,
    b_qk: torch.Tensor,
    a_v: torch.Tensor,
    b_v: torch.Tensor,
    num_heads: int,
    head_dim: int,
    embed_dim: int,
    linear_a_out: nn.Linear,
    linear_b_out: nn.Linear,
    attn_mask=None,
    dropout_p=0.0,
    scale=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    a_length, bsz, _ = a_qk.size()
    b_length, bsz, _ = b_qk.size()

    # (L, N, E) -> (N, num_heads, L, head_dim)
    a_qk = a_qk.reshape(a_length, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
    b_qk = b_qk.reshape(b_length, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
    a_v = a_v.reshape(a_length, bsz, num_heads, head_dim).permute(1, 2, 0, 3)
    b_v = b_v.reshape(b_length, bsz, num_heads, head_dim).permute(1, 2, 0, 3)

    a_out, b_out = mix_scaled_dot_product_attention(a_qk, b_qk, a_v, b_v, attn_mask, dropout_p, scale)

    # (N, num_heads, L, head_dim) -> (L, N, E)
    a_out = a_out.permute(2, 0, 1, 3).reshape(a_length * bsz, embed_dim)
    b_out = b_out.permute(2, 0, 1, 3).reshape(b_length * bsz, embed_dim)

    a_out = linear_a_out(a_out).reshape(a_length, bsz, embed_dim)
    b_out = linear_b_out(b_out).reshape(b_length, bsz, embed_dim)

    return a_out, b_out


class MultiHeadMixAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        batch_first: bool = True,
        dropout: float = 0.0,
        bias: bool = True,
        scale: float | None = None,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.bias = bias
        self.scale = scale
        self.batch_first = batch_first
        self.linear_a_qkey = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.linear_a_value = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.linear_b_qkey = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.linear_b_value = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.linear_a_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.linear_b_out = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.head_dim = embed_dim // num_heads
        if self.head_dim * num_heads != embed_dim:
            raise ValueError("embed_dim must be divisible by num_heads")

    def _check_mask_and_merge(
        self,
        attn_mask: torch.Tensor | None,
        a_key_padding_mask: torch.Tensor | None,
        b_key_padding_mask: torch.Tensor | None,
    ) -> torch.Tensor | None:
        merged_mask: torch.Tensor | None = None

        if attn_mask is not None:
            # expand attn_mask to match the atteniton size for SDP -> [n, num_heads, L, S]
            if attn_mask.dim() == 2:  # input mask shape: [L, S]
                attn_mask = attn_mask.view(1, 1, self.a_length, self.b_length).expand(
                    self.bsz, self.num_heads, self.a_length, self.b_length
                )
            elif attn_mask.dim() == 3:  # input mask shape: [n*num_heads, L, S]
                attn_mask = attn_mask.view(self.bsz, self.num_heads, self.a_length, self.b_length)
            merged_mask = attn_mask

        if a_key_padding_mask is not None:  # input mask shape: [n, L]
            # expand a_key_padding_mask to match the atteniton size for SDP -> [n, num_heads, L, S]
            a_key_padding_mask = a_key_padding_mask.view(self.bsz, 1, self.a_length, 1).expand(
                self.bsz, self.num_heads, self.a_length, self.b_length
            )
            if merged_mask is None:
                merged_mask = a_key_padding_mask
            else:
                merged_mask = torch.logical_or(merged_mask, a_key_padding_mask)

        if b_key_padding_mask is not None:  # input mask shape: [n, S]
            # expand b_key_padding_mask to match the atteniton size for SDP -> [n, num_heads, L, S]
            b_key_padding_mask = b_key_padding_mask.view(self.bsz, 1, 1, self.b_length).expand(
                self.bsz, self.num_heads, self.a_length, self.b_length
            )
            if merged_mask is None:
                merged_mask = b_key_padding_mask
            else:
                merged_mask = torch.logical_or(merged_mask, b_key_padding_mask)
        return merged_mask

    def forward(
        self,
        a: torch.Tensor,
        b: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        a_key_padding_mask: torch.Tensor | None = None,
        b_key_padding_mask: torch.Tensor | None = None,
    ):
        if not a.size(-1) == b.size(-1) == self.embed_dim:
            raise ValueError("input tensor.size(-1) must be equal to embed_dim")
        if not a.dim() == b.dim() == 3:
            raise ValueError("input tensor must be batched")
        """
        Args:
            a: Input tensor with shape: [n, L, E] for batch_first=True, otherwise [L, n, E].
            b: Input tensor with shape: [n, S, E] for batch_first=True, otherwise [S, n, E].
            attn_mask: Input bool tensor with shape: [L, S] or [n*num_heads, L, S]
            a_key_padding_mask: Input bool tensor with shape: [n, L]
            b_key_padding_mask: Input bool tensor with shape: [n, S]
        Returns:
            a_out: Output tensor with shape: [n, L, E] for batch_first=True, otherwise [L, n, E].
            b_out: Output tensor with shape: [n, S, E] for batch_first=True, otherwise [S, n, E].

        Note:
            n is batch size, L is the maximum length of a, S is the maximum length of b, E is embedding dimensions.
            mask:
                A bool tensor where "True" values are positions that should be "masked" with float('-inf').
        """
        # get all size parameters
        if self.batch_first:
            a, b = a.transpose(0, 1), b.transpose(0, 1)
            # a: [L, n, E], b: [S, n, E]
        self.a_length, self.bsz, _ = a.size()
        self.b_length, self.bsz, _ = b.size()
        head_dim = self.embed_dim // self.num_heads

        # check mask shape and merge attn_mask, a_key_padding_mask, b_key_padding_mask
        if attn_mask is not None:
            if attn_mask.dim() == 3 and attn_mask.size(0) != self.bsz * self.num_heads:
                raise ValueError("The first dimension of attn_mask must be equal to batch_size * num_heads")
            if attn_mask.size()[-2:] != (self.a_length, self.b_length):
                raise ValueError(
                    "The size of 2 dimensional attn_mask should be"
                    f"[{self.a_length, self.b_length}] got {attn_mask.size()}"
                )
        merged_mask = self._check_mask_and_merge(attn_mask, a_key_padding_mask, b_key_padding_mask)

        a_qk: torch.Tensor = self.linear_a_qkey(a)
        a_v: torch.Tensor = self.linear_a_value(a)
        b_qk: torch.Tensor = self.linear_b_qkey(b)
        b_v: torch.Tensor = self.linear_b_value(b)
        a_out, b_out = mix_attention_froward(
            a_qk,
            b_qk,
            a_v,
            b_v,
            self.num_heads,
            head_dim,
            self.embed_dim,
            self.linear_a_out,
            self.linear_b_out,
            merged_mask,
            self.dropout,
            self.scale,
        )

        if self.batch_first:
            a_out, b_out = a_out.transpose(0, 1), b_out.transpose(0, 1)

        return a_out, b_out
