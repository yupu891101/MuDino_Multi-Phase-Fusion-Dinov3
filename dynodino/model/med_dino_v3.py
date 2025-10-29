from pathlib import Path

import torch
from dinov3.models.vision_transformer import DinoVisionTransformer, vit_base

from dynodino.utils.utils import file_downloader

MODEL_URL = "https://huggingface.co/ricklisz123/MedDINOv3-ViTB-16-CT-3M/resolve/main/model.pth?download=true"
MODEL_PATH = Path(__file__).parent.parent.joinpath("checkpoints/meddinov3.pth").absolute().as_posix()


def get_med_dino_v3(model_path: str = MODEL_PATH) -> DinoVisionTransformer:
    """Load the Med-DINOv3 model weights from the specified path."""
    if not Path(model_path).exists():
        file_downloader(MODEL_URL, MODEL_PATH, False)

    med_dino_weight: dict[str, dict[str, torch.Tensor]] = torch.load(model_path, weights_only=False)
    state_dict = med_dino_weight["teacher"]
    state_dict = {
        k.replace("backbone.", ""): v for k, v in state_dict.items() if "ibot" not in k and "dino_head" not in k
    }

    model = vit_base(drop_path_rate=0.2, layerscale_init=1.0e-05, n_storage_tokens=4, qkv_bias=False, mask_k_bias=True)
    model.load_state_dict(state_dict)

    return model
