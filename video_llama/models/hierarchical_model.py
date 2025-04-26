import torch
from torch import nn

import sys
sys.path.insert(0, '/home/bhavashok/Video-LLaMA')
import video_llama
from video_llama.common.registry import registry
from video_llama.models.vm import VideoModel
from video_llama.models.blip2 import Blip2Base, disabled_train
from argparse import Namespace
from video_llama.common.config import Config
from torch.nn import functional as F

@registry.register_model("heirarchical_model")
class HierarchicalModel(Blip2Base):
    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
        "pretrain_llama_v2": "configs/models/video_llama.yaml",
        "mistral": "configs/models/video_llama.yaml",
        "phi-2": "configs/models/video_llama.yaml",
        "phi-3": "configs/models/video_llama.yaml",
        "vm": "configs/models/video_llama.yaml",
        "heirarchical_model": "configs/models/video_llama.yaml",
    }

    def __init__(self, video_model_config_path: str,
                 d_model: int = 1024,
                 n_layers: int = 4,
                 n_heads: int = 8,
                 dropout: float = 0.1):
        super().__init__()
        # ---------- 1. Load (and optionally freeze) the frame encoder ----------
        args = {'cfg_path': video_model_config_path,
                'options': None,
                'gpu_id': 0}
        cfg = Config(Namespace(**args))
        self.video_model = VideoModel.from_config(cfg.model_cfg)
        self.video_model.eval()                     # keep it in eval mode
        for p in self.video_model.parameters():     # freeze weights
            p.requires_grad = False

        # ---------- 2. Build the meta-transformer ----------
        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=4 * d_model,
            dropout=dropout,
            batch_first=True,   # -> [B, S, D]
            norm_first=True
        )
        self.meta_model = nn.TransformerEncoder(enc_layer, num_layers=n_layers)

        self.cls_token = nn.Parameter(torch.zeros(1, d_model))
        self.proj = nn.Identity()#= nn.Linear(d_model, out_dim)
        init_logit_scale = 0.07
        self.temperature_parameter = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.temperature_parameter.requires_grad = True

    def compute_clip_loss_blip(self, image_feats_all, text_feat_all):
        sim_q2t = torch.einsum("if,tf->it", image_feats_all, text_feat_all)

        # image-text similarity: aggregate across all query tokens
        sim_i2t = sim_q2t#.max(-1)
        sim_i2t = sim_i2t / self.temperature_parameter

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        # text_feat_all = [b, 256]
        # image_feats_all = [b, 256]
        sim_t2q = torch.einsum("tf,if->ti", text_feat_all, image_feats_all)
        #sim_t2q = torch.matmul(
        #    text_feat_all.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        #).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i = sim_t2q#.max(-1)
        #sim_t2i = sim_t2q.mean(-1)
        sim_t2i = sim_t2i / self.temperature_parameter  # [batch_size, batch_size*num_gpu]

        batch_size = image_feats_all.size(0)
        targets = torch.arange(batch_size).to(image_feats_all.device)
        #siglip
        #targets = F.one_hot(targets, num_classes=int(batch_size)).float()
        #loss_i2t = F.binary_cross_entropy_with_logits(sim_i2t, targets)
        #loss_t2i = F.binary_cross_entropy_with_logits(sim_t2i, targets)
        loss_i2t = F.cross_entropy(sim_i2t, targets)
        loss_t2i = F.cross_entropy(sim_t2i, targets)
        #print('loss_i2t, loss_t2i: ', loss_i2t, loss_t2i)
        loss = (
                #F.cross_entropy(sim_i2t, targets, label_smoothing=0.1)
                #+ F.cross_entropy(sim_t2i, targets, label_smoothing=0.1)
                #+ F.cross_entropy(sim_t2i, targets)
                loss_i2t + loss_t2i
            ) / 2
        return loss, sim_i2t, sim_t2i

    def _encode_text(self, texts, device):
        """
        Helper → turns a list[str] into a L2-normalised (B, D) embedding
        using the same text tower your Video-LLaMA backbone relies on.
        """
        toks = self.video_model.text_tokenizer(
            texts, padding=True, truncation=True, return_tensors="pt"
        ).to(device)

        last_hidden = self.video_model.text_transformer(
            toks["input_ids"]
        ).last_hidden_state                          # (B, seq, hidden)
        # use <eos> / last token, then projection + normalise
        txt_emb = F.normalize(
            self.video_model.text_proj(last_hidden[:, -1, :]), dim=-1
        )                                            # (B, D)
        return txt_emb

    def forward(self, samples):
        # Need this o be in the format:
        # [1, 3, 16, 16, 224, 224]
        # [B, C, NC, T, H, W]
        # First pass makes it
        # [B, NC, 1024]
        # Second pass makes it:
        # [B, 1024]
        # Texts should also be:
        # [B, 1024] after this.
        # NC can be padded with a max number of clips per frame
        # If each clip is 5s, then maybe 16 is fine as a max.
        # We also need to return the padding for NC.
        images_nested = samples["image"]          # ragged list
        B = len(images_nested)
        device = next(self.parameters()).device

        # ------------------------------------------------------------------ #
        # 1) flatten clips  → encode_video
        # ------------------------------------------------------------------ #
        flat_clips   = [c.to(device) for vid in images_nested for c in vid]
        vid_tensor   = torch.stack(flat_clips, dim=0)        # (ΣNC_i, C,T,H,W)

        with torch.no_grad(), self.maybe_autocast():
            clip_feat = self.video_model.encode_video(vid_tensor)  # (ΣNC_i,…)

        # pool to one vector per clip if the backbone returns sequences
        if clip_feat.ndim == 3:
            clip_feat = clip_feat.mean(dim=1)                 # (ΣNC_i, D)

        # ------------------------------------------------------------------ #
        # 2) regroup by video & pad to max_NC
        # ------------------------------------------------------------------ #
        D = clip_feat.size(-1)
        clip_ptr = 0
        per_video_feats = []
        clip_lens = []
        for vid in images_nested:
            n = len(vid)
            per_video_feats.append(clip_feat[clip_ptr : clip_ptr + n])  # (n, D)
            clip_ptr += n
            clip_lens.append(n)

        max_NC = max(clip_lens)
        pad_val = torch.zeros(D, device=device)

        # (B, max_NC, D) padded clip-level embeddings
        clips_padded = torch.stack([
            torch.cat([v, pad_val.repeat(max_NC - len(v), 1)])
            for v in per_video_feats
        ], dim=0)

        # ------------------------------------------------------------------ #
        # 3) meta-transformer with CLS token & key-padding mask
        # ------------------------------------------------------------------ #
        cls_tok = self.cls_token.expand(B, 1, -1)             # (B,1,D)
        x = torch.cat([cls_tok, clips_padded], dim=1)          # (B, 1+max_NC, D)

        # build mask: False = keep, True = pad
        pad_mask = torch.ones(B, 1 + max_NC, dtype=torch.bool, device=device)
        for i, n in enumerate(clip_lens):
            pad_mask[i, :1 + n] = False                       # keep CLS + real clips

        x = self.meta_model(x, src_key_padding_mask=pad_mask) # (B, 1+max_NC, D)
        video_emb = F.normalize(self.proj(x[:, 0, :]), dim=-1)  # CLS → (B, D)

        # ------------------------------------------------------------------ #
        # 4)   evaluation  vs.  training with contrastive loss
        # ------------------------------------------------------------------ #
        if not self.training or "texts" not in samples:
            return video_emb

        print(samples["texts"])
        texts = samples["texts"]
        text_embs = [self._encode_text(texts[b_idx], device) for b_idx in range(len(texts))]          # (B, D)
        text_embs = torch.stack(text_embs).squeeze()
        print(text_embs.size())
        clip_loss, _, _ = self.compute_clip_loss_blip(video_emb, text_embs)

        return {"loss": clip_loss, "video_emb": video_emb, "text_emb": text_embs}

    @classmethod
    def from_config(cls, cfg):
        # vit_model = cfg.get("vit_model", "eva_clip_g")
        # q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        # img_size = cfg.get("image_size")
        # num_query_token = cfg.get("num_query_token")
        # llama_model = cfg.get("llama_model")

        # drop_path_rate = cfg.get("drop_path_rate", 0)
        # use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        # vit_precision = cfg.get("vit_precision", "fp16")
        # freeze_vit = cfg.get("freeze_vit", True)
        # freeze_qformer = cfg.get("freeze_qformer", True)
        # low_resource = cfg.get("low_resource", False)
        # device_8bit = cfg.get("device_8bit", 0)

        # prompt_path = cfg.get("prompt_path", "")
        # prompt_template = cfg.get("prompt_template", "")
        # max_txt_len = cfg.get("max_txt_len", 32)
        # end_sym = cfg.get("end_sym", '\n')
        
        # frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        # frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)
        # frozen_audio_Qformer = cfg.get("frozen_audio_Qformer", True)

        # llama_proj_model = cfg.get("llama_proj_model", '')
        
        # fusion_header_type = cfg.get("fusion_header_type", 'seqTransf')
        # max_frame_pos = cfg.get("max_frame_pos", 32)
        # fusion_head_layers = cfg.get("fusion_head_layers", 2)
        # num_video_query_token =  cfg.get("num_video_query_token", 32)

        # equip_audio_branch= cfg.get("equip_audio_branch", True)
        # num_audio_query_token =  cfg.get("num_audio_query_token", 8)
        # imagebind_ckpt_path = cfg.get("imagebind_ckpt_path", '/mnt/workspace/ckpt')

        # use_clip_loss = cfg.get("use_clip_loss", False)
        # use_generation_loss = cfg.get("use_generation_loss", False)
        # use_itm_loss = cfg.get("use_itm_loss", True)
        # use_reconstruction_loss = cfg.get('use_reconstruction_loss', False)
        # clip_dim_size = cfg.get("clip_dim_size", 256)
        # num_videoq_hidden_layers = cfg.get('num_videoq_hidden_layers', 4)

        # use_position_embeddings = cfg.get('use_position_embeddings', False)
        # num_video_transformer_layers = cfg.get('num_video_transformer_layers', 6)
        # num_video_transformer_heads = cfg.get('num_video_transformer_heads', 4)

        # model_type = cfg.get("model_type", None)
        video_model_config_path = cfg.get("video_model_config_path", None)
        d_model = cfg.get("d_model", 1024)
        n_layers = cfg.get("n_layers", 4)
        n_heads = cfg.get("n_heads", 8)
        dropout = cfg.get("dropout", 0.1)
        model = cls(
            video_model_config_path=video_model_config_path,
            d_model=d_model,
            n_layers=n_layers,
            n_heads=n_heads,
            dropout=dropout,
            # vit_model=vit_model,
            # q_former_model=q_former_model,
            # img_size=img_size,
            # drop_path_rate=drop_path_rate,
            # use_grad_checkpoint=use_grad_checkpoint,
            # vit_precision=vit_precision,
            # freeze_vit=freeze_vit,
            # freeze_qformer=freeze_qformer,
            # num_query_token=num_query_token,
            # llama_model=llama_model,
            # prompt_path=prompt_path,
            # prompt_template=prompt_template,
            # max_txt_len=max_txt_len,
            # end_sym=end_sym,
            # low_resource=low_resource,
            # device_8bit=device_8bit,
            # fusion_header_type=fusion_header_type,
            # max_frame_pos=max_frame_pos,
            # fusion_head_layers=fusion_head_layers,
            # frozen_llama_proj=frozen_llama_proj,
            # frozen_video_Qformer=frozen_video_Qformer,
            # frozen_audio_Qformer=frozen_audio_Qformer,
            # num_video_query_token=num_video_query_token,
            # num_audio_query_token = num_audio_query_token,
            # imagebind_ckpt_path = imagebind_ckpt_path,
            # equip_audio_branch = equip_audio_branch,
            # llama_proj_model = llama_proj_model,
            # use_clip_loss = use_clip_loss,
            # use_generation_loss = use_generation_loss,
            # use_itm_loss = use_itm_loss,
            # use_reconstruction_loss = use_reconstruction_loss,
            # clip_dim_size = clip_dim_size,
            # model_type = model_type,
            # num_videoq_hidden_layers = num_videoq_hidden_layers,
            # use_position_embeddings=use_position_embeddings,
            # num_video_transformer_layers=num_video_transformer_layers,
            # num_video_transformer_heads=num_video_transformer_heads,
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
    model = HierarchicalModel(video_model_config_path)
    model = model.cuda()

    samples = [
        {
            'image': torch.rand(2, 3, 16, 224, 224).cuda(),
            'text_input': ['sample text1', 'sample text2'],
            'texts': ['sample text1', 'sample text2'],
            'type': 'video',
        }
    ]
  
    out = model(samples[0])
    print("video-level embedding:", out.shape)