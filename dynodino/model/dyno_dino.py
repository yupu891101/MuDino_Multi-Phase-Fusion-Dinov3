from typing import Literal

import torch
from dinov3.models.vision_transformer import DinoVisionTransformer
from torch import Tensor, nn

from .med_dino_v3 import get_med_dino_v3
from .modules import CrossAttnBlock, FFNBlock, MixAttnBlock


class DynoDinoFusionLayer(nn.Module):
    def __init__(
        self,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 1024,
        activation: str = "gelu",
        first_phase: Literal["arterial", "portal"] = "arterial",
    ):
        super().__init__()
        self.first_phase = first_phase
        self.mix_attn1 = MixAttnBlock(d_model=d_model, nhead=nhead)
        self.mix_attn2 = MixAttnBlock(d_model=d_model, nhead=nhead)
        self.ffn1 = FFNBlock(d_model=d_model, dim_feedforward=dim_feedforward, activation=activation, use_glu=True)
        self.ffn2 = FFNBlock(d_model=d_model, dim_feedforward=dim_feedforward, activation=activation, use_glu=True)

    def foward(self, x: Tensor) -> Tensor:
        """Create a fusion block with the specified attention type."""
        # x shape: (B*3, L, C)
        # transform x to (B, 3, L, C)
        B3, L, C = x.shape
        x = x.view(B3 // 3, 3, L, C)
        a, p, d = x[:, 0], x[:, 1], x[:, 2]  # a: arterial, p: portal, d: delayed
        # Mix Attention
        if self.first_phase == "arterial":
            a, d = self.mix_attn1(a, d)
            p, d = self.mix_attn2(p, d)
        else:
            p, d = self.mix_attn1(a, d)
            a, d = self.mix_attn2(p, d)
        # FFN
        a = self.ffn1(a)
        p = self.ffn2(p)
        # aggregate
        x = torch.stack([a, p, d], dim=1).view(B3, L, C)
        return x


class DynodinoFusionModel(nn.Module):
    def __init__(
        self,
        layer_policy: Literal["a_first", "p_first", "all_a", "all_p"] = "a_first",
        layer_num: int = 4,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 1024,
        activation: str = "gelu",
    ):
        super().__init__()
        self.layer_policy = layer_policy
        self.d_model = d_model
        self.nhead = nhead
        self.dim_feedforward = dim_feedforward
        self.activation = activation

        self.cross_attn_block1 = CrossAttnBlock(d_model=d_model, nhead=nhead)
        self.cross_attn_block2 = CrossAttnBlock(d_model=d_model, nhead=nhead)
        self.ffn = FFNBlock(d_model=d_model, dim_feedforward=dim_feedforward, activation=activation, use_glu=True)
        self.fusion_layers = self._make_layers(layer_num)

    def _make_layers(self, layer_num: int):
        module_list = nn.ModuleList()
        for i in range(layer_num - 1):
            match self.layer_policy:
                case "a_first":
                    phase = "arterial" if i % 2 == 0 else "portal"
                case "p_first":
                    phase = "portal" if i % 2 == 0 else "arterial"
                case "all_a":
                    phase = "arterial"
                case "all_p":
                    phase = "portal"
                case _:
                    raise ValueError(f"Invalid layer_policy: {self.layer_policy}")
            module_list.append(
                DynoDinoFusionLayer(
                    d_model=self.d_model,
                    nhead=self.nhead,
                    dim_feedforward=self.dim_feedforward,
                    activation=self.activation,
                    first_phase=phase,
                )
            )
        return module_list

    def forward(self, x: Tensor) -> Tensor:
        for fusion_layer in self.fusion_layers:
            x = fusion_layer(x)
        # X shape: (B*3, L, C)
        # split x into a, p, d
        B3, L, C = x.shape
        x = x.view(B3 // 3, 3, L, C)
        a, p, d = x[:, 0], x[:, 1], x[:, 2]  # a: arterial, p: portal, d: delayed
        # Cross Attention
        a = self.cross_attn_block1(a, d, d)
        p = self.cross_attn_block2(p, d, d)
        # release memory of d
        del d
        result = (a - p) + self.ffn(a - p)
        return result


class DynoDino(nn.Module):
    def __init__(
        self,
        backbone: DinoVisionTransformer,
        layer_policy: Literal["a_first", "p_first", "all_a", "all_p"] = "a_first",
        layer_num: int = 4,
        d_model: int = 768,
        nhead: int = 12,
        dim_feedforward: int = 1024,
        activation: str = "gelu",
    ):
        super().__init__()
        self.backbone = backbone
        self.fusion_model = DynodinoFusionModel(
            layer_policy=layer_policy,
            layer_num=layer_num,
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            activation=activation,
        )

    def forward(self, x: Tensor) -> Tensor:
        # x shape: (B, 3, H, W)
        B, _, H, W = x.shape
        x = x.view(B * 3, 1, H, W).repeat(1, 3, 1, 1)  # reshape to (B*3, 3, H, W)
        x = self.backbone(x, is_training=True)["x_norm_patchtokens"]
        x = self.fusion_model(x)
        return x
