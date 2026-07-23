import os

from torch import nn
import timm
from config import DATA_ROOT


# Keep downloaded model weights with the other external/cache data by default.
_NETWORK_HF_CACHE = os.environ.get("LEDXA_HF_CACHE", str(DATA_ROOT / "hf_cache"))


class DINOv3(nn.Module):
    def __init__(self, freeze_backbone=True, pool='token',
                 model_name='vit_small_patch16_dinov3.lvd1689m',
                 load_to_gpu=False):
        super().__init__()

        print(f"Initializing {model_name}...")

        if load_to_gpu:
            # For very large models (e.g. 7B): create on meta device, then load
            # weights directly to CUDA to avoid CPU RAM bottleneck.
            import safetensors.torch
            from timm.models._hub import download_cached_file

            os.makedirs(_NETWORK_HF_CACHE, exist_ok=True)
            os.environ["HF_HOME"] = _NETWORK_HF_CACHE

            # Build architecture without weights
            self.backbone = timm.create_model(
                model_name,
                pretrained=False,
                num_classes=0,
                global_pool=pool,
            )

            # Download checkpoint (uses network cache dir)
            from timm.models._pretrained import PretrainedCfg
            pretrained_cfg = timm.get_pretrained_cfg(model_name)
            hf_hub_id = pretrained_cfg.hf_hub_id
            hf_hub_filename = pretrained_cfg.hf_hub_filename or "model.safetensors"

            from huggingface_hub import hf_hub_download
            ckpt_path = hf_hub_download(
                repo_id=hf_hub_id,
                filename=hf_hub_filename,
                cache_dir=os.path.join(_NETWORK_HF_CACHE, "hub"),
            )
            print(f"  Loading weights directly to CUDA from: {ckpt_path}")
            # Load tensors one at a time to avoid mmapping the entire 27GB file
            from safetensors import safe_open
            state_dict = {}
            with safe_open(ckpt_path, framework="pt", device="cuda") as f:
                for key in f.keys():
                    state_dict[key] = f.get_tensor(key)

            # timm may store with different key prefixes; try loading flexibly
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False)
            print(f"  Loaded: missing={len(missing)}, unexpected={len(unexpected)}")
            if missing:
                print(f"  (missing keys are expected for head/classifier layers)")
        else:
            # Standard loading for smaller models
            self.backbone = timm.create_model(
                model_name,
                pretrained=True,
                num_classes=0,
                global_pool=pool,
            )

        # Freeze logic
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone Frozen.")

    def forward(self, x):
        feats = self.backbone(x)
        return feats
