# pip install --upgrade peft transformers accelerate
import torch
import torch.nn as nn
import sys
sys.path.insert(0, '/home/bhavashok/Video-LLaMA')
from video_llama.common.registry import registry
from video_llama.models.vm import VideoModel
from video_llama.models.blip2 import Blip2Base, disabled_train, restore_train
from argparse import Namespace
from video_llama.common.config import Config
from torch.nn import functional as F
from typing import Optional
import wandb

"""
Video-LLAMA + Decompression
"""

class DecompressionNet(nn.Module):
    """A lightweight 3‑D up‑convolutional decoder.

    The backbone produces a sequence of *token* features of shape `[B, N, C]`
    where `N = T · (H/P) · (W/P)` for patch‑size *P*.  The decoder first maps
    the token dimension `C → hidden_dim`, re‑arranges the tokens back to a
    `[B, hidden_dim, T, H/P, W/P]` *tube*, and then uses three transposed 3‑D
    convolutions (stride = 2) to upscale both spatial and temporal dimensions
    back to *approximately* `224 × 224` and `T` frames.
    """

    def __init__(
        self,
        *,
        in_dim: int = 1024,
        out_channels: int = 3,
        patch_size: int = 16,
        num_frames: int = 16,
        height: int = 224,
        width: int = 224,
    ) -> None:
        super().__init__()

        self.num_frames = num_frames
        self.patch_h = height // patch_size
        self.patch_w = width // patch_size
        self.patch_size = patch_size
        hidden_dim = self.num_frames * self.patch_h * self.patch_w

        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
        )

        # three upsampling stages ⇒ scale‑factor 2³ = 8.
        # 14 (ViT‑B) · 8 ≃ 224 – we land exactly back on the original res.
        self.decoder = nn.Sequential(
            nn.ConvTranspose3d(1, 128, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GELU(),
            nn.ConvTranspose3d(128, 64, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GELU(),
            nn.ConvTranspose3d(64, 32, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.GELU(),
            nn.ConvTranspose3d(32, out_channels, kernel_size=(1, 2, 2), stride=(1, 2, 2)),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # x: [B, N, C]
        B, N = x.shape
        x = self.proj(x)  # [B, N, hidden]
        #x = x.transpose(1, 2)  # [B, hidden, N]
        #print(x.size())
        x = x.view(B, -1, self.num_frames, self.patch_h, self.patch_w)
        #print(x.size())
        x = self.decoder(x)  # [B, 3, T, H, W]
        #print(x.size())
        return x

@registry.register_model("video_model_decomp")          # <-- name used in YAML
class VideoModelDecomp(Blip2Base):
    """
    A thin wrapper that

      • builds the normal `VideoModel`
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
        freeze_base: bool = True,
    ):
        super().__init__()

        # 1. build the vanilla backbone -------------------------------------
        args = {'cfg_path': video_model_config_path,
                'options': None,
                'gpu_id': 0}
        cfg = Config(Namespace(**args))
        self.model = VideoModel.from_config(cfg.model_cfg)
        self.model.eval()
        # keep it in eval mode

        # 2. TODO: initialize decompression module -----------------------------
        self.decompression_module = DecompressionNet(num_frames=16, in_dim=self.model.video_proj.out_features)
        self._loss_fn = nn.MSELoss()

        # 4. (optional) freeze everything except LoRA ------------------------
        if freeze_base:
            for n, p in self.model.named_parameters():
                p.requires_grad_(False)
        else:
            for name, param in self.model.named_parameters():
                param.requires_grad = True
            for name, param in self.model.visual_encoder.named_parameters():
                param.requires_grad = True
            restore_train(self.model.visual_encoder)
            self.model.visual_encoder = self.model.visual_encoder.train()
            for name, param in self.model.ln_vision.named_parameters():
                param.requires_grad = True
            restore_train(self.model.ln_vision)
            self.model.ln_vision = self.model.ln_vision.train()

    def loss_fn(self, pred, target):
        return self._loss_fn(pred, target)

    
    def compress(self, video):
        return self.model.encode_video(video)
    
    def decompress(self, feat):
        return self.decompression_module(feat)

    # --------------------------------------------------------------------- #
    # pass-through API
    def forward(self, *args, **kwargs):
        with self.maybe_autocast():
            # TODO: Get original video
            original_video = args[0]['image']
            feat = self.compress(original_video)
            output_video = self.decompress(feat)
            loss = self.loss_fn(output_video, original_video)
            wandb.log({"first_output_frame_28x28": wandb.Image(output_video[0, :, 0, :, :], mode='RGB')})
        return {"loss": loss}

    @classmethod
    def from_config(cls, cfg):
        video_model_config_path = cfg.get("video_model_config_path", None)
        # TODO: Get decompression parameters
        freeze_base = cfg.get("freeze_base", True)
        model = cls(
                video_model_config_path=video_model_config_path,
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
    model = registry.get_model_class("video_model_decomp")(
        video_model_config_path=video_model_config_path,
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
