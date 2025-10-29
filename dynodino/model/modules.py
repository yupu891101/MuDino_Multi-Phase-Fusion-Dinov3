from typing import Literal

from torch import Tensor, nn

from .mhma import MultiHeadMixAttention

type AvaliableActions = Literal["relu", "leakyrelu", "prelu", "sigmoid", "tanh", "silu", "gelu"]


def get_activation(activation: str):
    activation = activation.lower()
    match activation:
        case "relu":
            return nn.ReLU()
        case "leakyrelu":
            return nn.LeakyReLU()
        case "prelu":
            return nn.PReLU()
        case "sigmoid":
            return nn.Sigmoid()
        case "tanh":
            return nn.Tanh()
        case "silu":
            return nn.SiLU()
        case "gelu":
            return nn.GELU()
        case _:
            return nn.ReLU()


class GLUBlock(nn.Module):
    """Gated Linear Unit Block from 'GLU Variants Improve Transformer'"""

    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.w = nn.Linear(d_model, dim_feedforward * 2, bias=False, device=device, dtype=dtype)
        self.v = nn.Linear(dim_feedforward, d_model, bias=False, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.activation = get_activation(activation)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(self.activation(self.w(x)) * self.v(x))


class FFNBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1,
        activation: str = "gelu",
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,
        bias: bool = True,
        use_glu: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.l_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device, dtype=dtype)
        if use_glu:
            self.linear1 = GLUBlock(d_model, dim_feedforward, dropout, activation, device, dtype)
        else:
            self.linear1 = nn.Linear(d_model, dim_feedforward, bias=bias, device=device, dtype=dtype)
        self.linear2 = nn.Linear(dim_feedforward, d_model, bias=bias, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)

        self.activation = get_activation(activation)

    def _ff_block(self, x: Tensor) -> Tensor:
        x = self.linear2(self.dropout(self.activation(self.linear1(x))))
        return self.dropout(x)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_first:
            x = x + self._ff_block(self.l_norm(x))
        else:
            x = self.l_norm(x + self._ff_block(x))
        return x


class MixAttnBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        batch_first: bool = True,
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.a_l_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device, dtype=dtype)
        self.b_l_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.MA = MultiHeadMixAttention(d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first)

    def _ma_block(
        self,
        a: Tensor,
        b: Tensor,
        src_mask: Tensor | None,
        a_key_padding_mask: Tensor | None,
        b_key_padding_mask: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        a, b = self.MA(
            a, b, attn_mask=src_mask, a_key_padding_mask=a_key_padding_mask, b_key_padding_mask=b_key_padding_mask
        )
        return self.dropout(a), self.dropout(b)

    def forward(
        self,
        a: Tensor,
        b: Tensor,
        src_mask: Tensor | None = None,
        a_key_padding_mask: Tensor | None = None,
        b_key_padding_mask: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if self.norm_first:
            new_a, new_b = self._ma_block(
                self.a_l_norm(a), self.b_l_norm(b), src_mask, a_key_padding_mask, b_key_padding_mask
            )
            a, b = a + new_a, b + new_b
        else:
            new_a, new_b = self._ma_block(a, b, src_mask, a_key_padding_mask, b_key_padding_mask)
            a, b = self.a_l_norm(a + new_a), self.b_l_norm(b + new_b)
        return self.dropout(a), self.dropout(b)


class SelfAttnBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        batch_first: bool = True,
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.l_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first, device=device, dtype=dtype
        )

    def _sa_block(
        self, x: Tensor, src_mask: Tensor | None, src_key_padding_mask: Tensor | None, is_causal: bool = False
    ) -> Tensor:
        x = self.mha(
            x, x, x, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False, is_causal=is_causal
        )[0]
        return self.dropout(x)

    def forward(
        self,
        x: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        if self.norm_first:
            x = x + self._sa_block(self.l_norm(x), src_mask, src_key_padding_mask, is_causal)
        else:
            x = self.l_norm(x + self._sa_block(x, src_mask, src_key_padding_mask, is_causal))
        return x


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dropout: float = 0.1,
        batch_first: bool = True,
        norm_first: bool = True,
        layer_norm_eps: float = 1e-5,
        bias: bool = False,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm_first = norm_first
        self.l_norm = nn.LayerNorm(d_model, eps=layer_norm_eps, bias=bias, device=device, dtype=dtype)
        self.dropout = nn.Dropout(dropout)
        self.mha = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, bias=bias, batch_first=batch_first, device=device, dtype=dtype
        )

    def _ca_block(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        src_mask: Tensor | None,
        src_key_padding_mask: Tensor | None,
        is_causal: bool = False,
    ) -> Tensor:
        x, _ = self.mha(
            q, k, v, attn_mask=src_mask, key_padding_mask=src_key_padding_mask, need_weights=False, is_causal=is_causal
        )
        return self.dropout(x)

    def forward(
        self,
        q: Tensor,
        k: Tensor,
        v: Tensor,
        src_mask: Tensor | None = None,
        src_key_padding_mask: Tensor | None = None,
        is_causal: bool = False,
    ) -> Tensor:
        if self.norm_first:
            q = q + self._ca_block(
                self.l_norm(q), self.l_norm(k), self.l_norm(v), src_mask, src_key_padding_mask, is_causal
            )
        else:
            q = self.l_norm(q + self._ca_block(q, k, v, src_mask, src_key_padding_mask, is_causal))
        return self.dropout(q)
