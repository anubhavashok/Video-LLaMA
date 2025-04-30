# pip install --upgrade peft transformers accelerate
import torch
import torch.nn as nn
from peft import LoraConfig, inject_adapter_in_model, TaskType
import sys
sys.path.insert(0, '/home/bhavashok/Video-LLaMA')
from video_llama.common.registry import registry
from video_llama.models.vm import VideoModel
from video_llama.models.blip2 import Blip2Base, disabled_train
from argparse import Namespace
from video_llama.common.config import Config
from torch.nn import functional as F
from typing import Optional

"""
Video-LLAMA + LoRA (PEFT)
"""


@registry.register_model("video_model_lora")          # <-- name used in YAML
class VideoModelLoRA(Blip2Base):
    """
    A thin wrapper that

      • builds the normal `VideoModel`
      • injects LoRA adapters with PEFT
      • (optionally) freezes the backbone
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
        "pretrain_llama_v2": "configs/models/video_llama.yaml",
        "mistral": "configs/models/video_llama.yaml",
        "phi-2": "configs/models/video_llama.yaml",
        "phi-3": "configs/models/video_llama.yaml",
        "vm": "configs/models/video_llama.yaml",
    }

    def __init__(
        self,
        video_model_config_path: str = "",
        lora_r: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.05,
        target_modules: Optional[list[str]] = None,
        freeze_base: bool = True,
    ):
        super().__init__()

        # 1. build the vanilla backbone -------------------------------------
        args = {'cfg_path': video_model_config_path,
                'options': None,
                'gpu_id': 0}
        cfg = Config(Namespace(**args))
        backbone = VideoModel.from_config(cfg.model_cfg)
        backbone.eval()                     # keep it in eval mode
        for p in backbone.parameters():     # freeze weights
            p.requires_grad = False

        # 2. choose which Linear layers to adapt -----------------------------
        if target_modules is None:
            target_modules = [
                "q_proj", "k_proj", "v_proj", "out_proj",  # attention
                "linear1", "linear2",                      # FFN
            ]

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.FEATURE_EXTRACTION,
            target_modules=target_modules,
        )

        # 3. inject adapters in-place ----------------------------------------
        self.model = inject_adapter_in_model(lora_cfg, backbone)

        # 4. (optional) freeze everything except LoRA ------------------------
        if freeze_base:
            for n, p in self.model.named_parameters():
                if "lora_" not in n:
                    p.requires_grad_(False)

    # --------------------------------------------------------------------- #
    # pass-through API
    def forward(self, *args, **kwargs):
        with self.maybe_autocast():
            return self.model(*args, **kwargs)

    def merge_adapters_and_unload(self):
        """Fuse LoRA weights back into the backbone for inference."""
        from peft import merge_adapter
        merge_adapter(self.model)
        return self.model

    @classmethod
    def from_config(cls, cfg):
        video_model_config_path = cfg.get("video_model_config_path", None)
        lora_r = cfg.get("lora_r", 8)
        lora_alpha = cfg.get("lora_alpha", 16)
        lora_dropout = cfg.get("lora_dropout", 0.05)
        target_modules = cfg.get("target_modules", None)
        freeze_base = cfg.get("freeze_base", True)
        model = cls(
                video_model_config_path=video_model_config_path,
                lora_r=lora_r,
                lora_alpha=lora_alpha,
                lora_dropout=lora_dropout,
                target_modules=target_modules,
                freeze_base=freeze_base,
        )
        ckpt_path = cfg.get("ckpt", "")  # load weights of MiniGPT-4
        if ckpt_path:
            print("Load first Checkpoint: {}".format(ckpt_path))
            ckpt = torch.load(ckpt_path, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        ckpt_path_2 = cfg.get("ckpt_2", "")
        if ckpt_path_2:
            print("Load second Checkpoint: {}".format(ckpt_path_2))
            ckpt = torch.load(ckpt_path_2, map_location="cpu")
            msg = model.load_state_dict(ckpt['model'], strict=False)
        return model

if __name__ == '__main__':
    video_model_config_path = '/home/bhavashok/experiments/askvideos/askvideos/models/configs/video_clip_video_model_nofreeze_v0.1.yaml'
    model = registry.get_model_class("video_model_lora")(
        video_model_config_path=video_model_config_path,
        lora_r=8,
        lora_alpha=16,
    ).cuda()

    # sanity-check
    model.train()
    #print(model.model.print_trainable_parameters())   # only LoRA params

    loss = model(
        {
            "image": torch.randn(2, 3, 16, 224, 224, device="cuda"),
            "text_input": ["foo", "bar"],
            "texts": ["foo", "bar"],
            "type": "video",
        }
    )
    print("loss:", loss)
