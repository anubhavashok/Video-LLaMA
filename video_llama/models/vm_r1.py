import logging
import random
from typing import Tuple

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from torch.nn import functional as F
from torch.nn import Transformer, TransformerDecoder, TransformerDecoderLayer, TransformerEncoder, TransformerEncoderLayer

import video_llama
from video_llama.common.registry import registry
from video_llama.models.blip2 import Blip2Base, disabled_train
#from video_llama.models.modeling_llama import LlamaForCausalLM
# from video_llama.models.Qformer import BertEncoder
from transformers import LlamaTokenizer,BertConfig
# from transformers.models.bert.modeling_bert import BertEncoder
import einops
import copy
from video_llama.models.Qformer import BertConfig, BertLMHeadModel
from video_llama.models.ImageBind.models.imagebind_model import ImageBindModel,ModalityType
from video_llama.models.ImageBind.models import imagebind_model
import numpy as np
import copy
from transformers import AutoModelForCausalLM, AutoModel, AutoTokenizer, LlamaForCausalLM, MistralForCausalLM, PreTrainedModel, PreTrainedTokenizerBase, AutoConfig

def load_model_with_quantization_fallback(
    model_name: str = "/mnt/h/models/DeepSeek-R1-Distill-Llama-8B/",
    trust_remote_code: bool = True,
    **kwargs
) -> Tuple[PreTrainedModel, PreTrainedTokenizerBase]:
  try:
      model = LlamaForCausalLM.from_pretrained(
          model_name,
          trust_remote_code=trust_remote_code,
          device_map={"":torch.cuda.current_device()},
          **kwargs
      )
      tokenizer = AutoTokenizer.from_pretrained(model_name)
      print("Model loaded successfully with original configuration")
      print(model_name)
      print(model)
      return model, tokenizer
  except ValueError as e:
      if "Unknown quantization type" in str(e):
          print(
              "Quantization type not supported directly. "
              "Attempting to load without quantization..."
          )

          config = AutoConfig.from_pretrained(
              model_name,
              trust_remote_code=trust_remote_code
          )
          if hasattr(config, "quantization_config"):
              delattr(config, "quantization_config")

          try:
              model = LlamaForCausalLM.from_pretrained(
                  model_name,
                  config=config,
                  trust_remote_code=trust_remote_code,
                  device_map={"":torch.cuda.current_device()},
                  **kwargs
              )
              tokenizer = AutoTokenizer.from_pretrained(
                  model_name,
                  trust_remote_code=trust_remote_code
              )
              print("Model loaded successfully without quantization")
              return model, tokenizer

          except Exception as inner_e:
              print(f"Failed to load model without quantization: {str(inner_e)}")
              raise
      else:
          print(f"Unexpected error during model loading: {str(e)}")
          raise


@registry.register_model("video_model_reasoning")
class VideoModelReasoning(Blip2Base):
    """
    Video transformer model.
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
        vit_model="eva_clip_g",
        q_former_model="https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth",
        img_size=224,
        drop_path_rate=0,
        use_grad_checkpoint=False,
        vit_precision="fp16",
        freeze_vit=True,
        freeze_qformer=True,
        num_query_token=32,
        llama_model="",
        prompt_path="",
        prompt_template="",
        max_txt_len=32,
        end_sym='\n',
        low_resource=False,  # use 8 bit and put vit in cpu
        device_8bit=0,  # the device of 8bit model should be set when loading and cannot be changed anymore.

        frozen_llama_proj=True,
        frozen_video_Qformer=True,
        frozen_audio_Qformer=True,

        llama_proj_model='',
        fusion_header_type= "seqTransf",
        max_frame_pos= 32,
        fusion_head_layers = 2,
        num_video_query_token = 32,
        num_audio_query_token = 8,
        imagebind_ckpt_path = '/mnt/workspace/ckpt',
        equip_audio_branch = True,
        use_clip_loss = True,
        use_generation_loss = True,
        use_itm_loss = True,
        clip_dim_size = 256,
        num_videoq_hidden_layers = 4,
        model_type = None,
        freeze_lm = False,
        use_reconstruction_loss = False,
        mask_ratio = 0.75,
        reconstruction_hidden_dim = 512,
    ):
        super().__init__()

        self.model_type = model_type
        self.tokenizer = self.init_tokenizer()
        self.low_resource = low_resource

        print('Loading VIT')
        self.visual_encoder, self.ln_vision = self.init_vision_encoder(
            vit_model, img_size, drop_path_rate, use_grad_checkpoint, vit_precision
        )
        if freeze_vit:
            for name, param in self.visual_encoder.named_parameters():
                param.requires_grad = False
            self.visual_encoder = self.visual_encoder.eval()
            self.visual_encoder.train = disabled_train
            for name, param in self.ln_vision.named_parameters():
                param.requires_grad = False
            self.ln_vision = self.ln_vision.eval()
            self.ln_vision.train = disabled_train
            logging.info("freeze vision encoder")
        print('Loading VIT Done')

        num_heads = 4
        num_layers = 6
        #num_patches = 1+self.visual_encoder.patch_size**2
        video_encoder_layer = TransformerEncoderLayer(d_model=self.visual_encoder.num_features, nhead=num_heads, dim_feedforward=self.visual_encoder.num_features, batch_first=True)
        self.video_transformer = TransformerEncoder(video_encoder_layer, num_layers=num_layers)
        self.text_tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        self.text_transformer = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
        
        print('Loading LLM')
        #if self.model_type == 'llama_r1':
        self.llama_model, self.llama_tokenizer = load_model_with_quantization_fallback(llama_model)
        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLM Done')

        # logging.info('Loading LLAMA proj')
        # # TODO: Check attribute and compute correct shape.
        self.llama_proj = nn.Linear(
            self.visual_encoder.num_features, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = self.llama_proj.load_state_dict(llama_proj_weight['model'], strict=False)

        if frozen_llama_proj:
            #  todo frozen  llama_proj
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = False
            logging.info('LLAMA proj is frozen')
        else:
            for name, param in self.llama_proj.named_parameters():
                param.requires_grad = True
            logging.info('LLAMA proj is not frozen')

        logging.info('Loading llama_proj Done')
        self.max_txt_len = max_txt_len
        self.end_sym = end_sym

        if prompt_path:
            with open(prompt_path, 'r') as f:
                raw_prompts = f.read().splitlines()
            filted_prompts = [raw_prompt for raw_prompt in raw_prompts if "<ImageHere>" in raw_prompt]
            self.prompt_list = [prompt_template.format(p) for p in filted_prompts]
            print('Load {} training prompts'.format(len(self.prompt_list)))
            print('Prompt Example \n{}'.format(random.choice(self.prompt_list)))
        else:
            self.prompt_list = []


        if freeze_lm:
            for name, param in self.text_transformer.named_parameters():
                param.requires_grad = False
            self.text_transformer.eval()
        else:
            for name, param in self.text_transformer.named_parameters():
                param.requires_grad = True
            self.text_transformer.train()


        #self.video_transformer = Transformer(d_model=self.visual_encoder.num_features, nhead=num_heads, dim_feedforward=self.visual_encoder.num_features)
        clip_dim = 1024
        self.video_proj = nn.Linear(self.visual_encoder.num_features, clip_dim)
        self.text_proj = nn.Linear(384, clip_dim)
        init_logit_scale = 0.07
        self.temperature_parameter = nn.Parameter(torch.ones([]) * init_logit_scale)
        self.temperature_parameter.requires_grad = True

        # Parameters for masking and reconstruction
        self.use_reconstruction_loss = use_reconstruction_loss
        if self.use_reconstruction_loss:
            self.mask_ratio = mask_ratio
            # Suppose your visual encoder outputs features of shape [B*T, num_patches, hidden_dim]
            # We'll need a decoder head to reconstruct original pixel patches (or encoded patches).
            # This is a simplistic example: a small MLP to go from encoder dim -> image patch dim.
            self.reconstruction_decoder = nn.Sequential(
                nn.Linear(self.visual_encoder.num_features, reconstruction_hidden_dim),
                nn.ReLU(),
                nn.Linear(reconstruction_hidden_dim, self.visual_encoder.num_features)
            )


    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def mask_patches(self, video, patch_size=16):
        """
        Mask a subset of patches in the video.
        video: B x C x T x H x W
        We'll assume your vision encoder takes patches of size 16x16 (adjust if needed).
        
        Steps:
        1. Extract patches from video
        2. Create a mask
        3. Apply the mask to patches
        """
        B, C, T, H, W = video.shape
        # Compute the number of patches along H and W
        # For simplicity, assume H and W are divisible by patch_size
        num_patches_h = H // patch_size
        num_patches_w = W // patch_size
        num_patches = num_patches_h * num_patches_w
        
        # Rearrange video into patches: [B, C, T, num_patches, patch_size, patch_size]
        patches = einops.rearrange(video, 
                                   'b c t (nh ph) (nw pw) -> b t (nh nw) c ph pw',
                                   ph=patch_size, pw=patch_size, nh=num_patches_h, nw=num_patches_w)
        
        # Now patches is B x T x (num_patches) x C x ph x pw
        # Flatten time and batch for easier masking: [B*T, num_patches, C, ph, pw]
        BT = B*T
        patches = patches.reshape(BT, num_patches, C, patch_size, patch_size)
        
        # Generate a random mask for patches
        num_masked = int(self.mask_ratio * num_patches)
        # For simplicity, mask the same number of patches in each sample
        # A better approach: mask randomly per sample.
        mask = torch.zeros(BT, num_patches, dtype=torch.bool, device=video.device)
        for i in range(BT):
            # Randomly choose patches to mask
            perm = torch.randperm(num_patches, device=video.device)
            masked_indices = perm[:num_masked]
            mask[i, masked_indices] = True

        # Create a masked version of the patches
        masked_patches = patches.clone()
        # You can either replace masked patches with 0 or a learned embedding
        masked_patches[mask] = 0.0

        # Reshape masked_patches back to video format if needed
        # But actually, we will pass the masked patches through the encoder.
        # If your visual encoder expects normal images, you'll need to reassemble them.
        # For demonstration, let's reassemble into the original video shape:
        masked_video = einops.rearrange(masked_patches, 
                                        'bt np c ph pw -> bt c (ph npw) (pw nph)',
                                        # This is tricky, we must remember how we took patches apart.
                                        # We'll invert properly:
                                        b=B, t=T, nh=num_patches_h, nw=num_patches_w, 
                                        np=num_patches, ph=patch_size, pw=patch_size)
        # Wait, the above rearrangement is complex. Let's do it step by step.
        # We know np = nh * nw. The original shape was: b, t, nh*nw, c, ph, pw
        # So we can do:
        masked_patches = masked_patches.reshape(B, T, num_patches_h, num_patches_w, C, patch_size, patch_size)
        masked_video = einops.rearrange(masked_patches, 
                                        'b t nh nw c ph pw -> b c t (nh ph) (nw pw)')
        
        # masked_video is now a video with masked patches zeroed out.
        # Return both the mask and the original patches for reconstruction
        return masked_video, mask, patches  # patches are the original patches we need to reconstruct

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

    def prompt_wrap(self, img_embeds, atts_img, prompt):
        batch_size = img_embeds.shape[0]
        # print(prompt)
        p_before, p_after = prompt.split('<ImageHere>')
        p_before_tokens = self.llama_tokenizer(
            p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_after_tokens = self.llama_tokenizer(
            p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
        p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
        p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
        print('p_before_embeds.size(), img_embeds.size(), p_after_embeds.size(): ', p_before_embeds.size(), img_embeds.size(), p_after_embeds.size())
        wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
        wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
        
        return wrapped_img_embeds, wrapped_atts_img

    def encode_video(self, video):
        video = video.cuda()
        batch_size,_,time_length,_,_ = video.size()
        video = einops.rearrange(video, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            feats = self.visual_encoder(video)
        feats = self.ln_vision(feats)
        feats = einops.rearrange(feats, '(b t) c h -> b (t c) h', b=batch_size, t=time_length)
        output_video_feats = self.video_transformer(feats)
        output_video_feats = output_video_feats[:, -1, :]
        video_feats_clip = F.normalize(self.video_proj(output_video_feats), dim=-1)
        return video_feats_clip

    def encode_text(self, text):
        tokens = self.text_tokenizer(text,  padding=True, truncation=True, return_tensors='pt')
        output_text_feats = self.text_transformer(tokens['input_ids'].cuda()).last_hidden_state
        output_text_feats = output_text_feats[:, -1, :]
        text_feats_clip = F.normalize(self.text_proj(output_text_feats), dim=-1)
        return text_feats_clip

    def forward(self, samples):
        video = samples['image'].cuda()
        text = samples['text_input']
        
        if self.use_reconstruction_loss:
            video, mask, original_patches = self.mask_patches(video)
        
        batch_size,_,time_length,_,_ = video.size()
        video = einops.rearrange(video, 'b c t h w -> (b t) c h w')
        
        with self.maybe_autocast():
            feats = self.visual_encoder(video)
            #print(feats.size()) # 16, 257, 1408
        feats = self.ln_vision(feats)
        feats = einops.rearrange(feats, '(b t) c h -> b (t c) h', b=batch_size, t=time_length)
        #print(feats.size()) # 1, (16*257), 1408
        # TODO: Add positional embeddings
        output_video_feats = self.video_transformer(feats)
        #output_video_feats = output_video_feats[:, -1, :]
        print('output_video_feats.size(): ', output_video_feats.size())

        llm_projected_video_feats = self.llama_proj(output_video_feats)
        llm_projected_video_atts = torch.ones(llm_projected_video_feats.size()[:-1], dtype=torch.long).to(llm_projected_video_feats.device)
        print('llm_projected_video_feats.size(): ', llm_projected_video_feats.size())
        print('llm_projected_video_atts.size(): ', llm_projected_video_atts.size())

        # TODO: assemble video features.
        if self.prompt_list:
            prompt = random.choice(self.prompt_list)
            #img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
            wrapped_feats, wrapped_atts = self.prompt_wrap(llm_projected_video_feats, llm_projected_video_atts, prompt)

        # TODO: Make sure we have a list in samples['text_input']. Not sure why we need this?
        #text = [t + self.end_sym for t in samples["text_input"]]
        to_regress_tokens = self.llama_tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            truncation=True,
            max_length=self.max_txt_len,
            add_special_tokens=False
        ).to(output_video_feats.device)

        targets = to_regress_tokens.input_ids.masked_fill(
            to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
        )

        # TODO: Understand this
        empty_targets = (
            torch.ones([wrapped_atts.shape[0], wrapped_atts.shape[1]+1],
                    dtype=torch.long).to(output_video_feats.device).fill_(-100)  # plus one for bos
        )
        targets = torch.cat([empty_targets, targets], dim=1)

        bos = torch.ones([batch_size, 1],
                        dtype=to_regress_tokens.input_ids.dtype,
                        device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
        bos_embeds = self.llama_model.model.embed_tokens(bos)
        atts_bos = wrapped_atts[:, :1]

        to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
        inputs_embeds = torch.cat([bos_embeds, wrapped_feats, to_regress_embeds], dim=1)
        attention_mask = torch.cat([atts_bos, wrapped_atts, to_regress_tokens.attention_mask], dim=1)
        with self.maybe_autocast():
            outputs = self.llama_model(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                return_dict=True,
                labels=targets,
            )
        return {"loss": outputs.loss}
        if self.use_reconstruction_loss:
            masked_feats = output_video_feats[mask]
            reconstructed_patches = self.reconstruction_decoder(masked_feats)
            reconstructed_pixels = self.reconstruction_head(reconstructed_patches)
            
            ph, pw = 16, 16  # your patch size
            reconstructed_pixels = reconstructed_pixels.view(-1, 3, ph, pw)

            # original_patches[mask]: [#masked_positions, C, ph, pw]
            target_pixels = original_patches[mask]

            reconstruction_loss = F.mse_loss(reconstructed_pixels, target_pixels)

        video_feats_clip = F.normalize(self.video_proj(output_video_feats), dim=-1)
        tokens = self.text_tokenizer(text,  padding=True, truncation=True, return_tensors='pt')
        output_text_feats = self.text_transformer(tokens['input_ids'].cuda()).last_hidden_state
        output_text_feats = output_text_feats[:, -1, :]
        text_feats_clip = F.normalize(self.text_proj(output_text_feats), dim=-1)
        clip_loss, sim_i2t, sim_t2i = self.compute_clip_loss_blip(video_feats_clip, text_feats_clip)
        
        loss_dict = {'loss': clip_loss}

        if self.use_reconstruction_loss:
            loss_dict.update('reconstruction_loss', reconstruction_loss)
            loss_dict.update('clip_loss', clip_loss)
            loss_dict['loss'] += 0.1 * reconstruction_loss
        
        # TODO: Arrange things into LLM and compute loss
        return loss_dict

    @classmethod
    def from_config(cls, cfg):
        vit_model = cfg.get("vit_model", "eva_clip_g")
        q_former_model = cfg.get("q_former_model", "https://storage.googleapis.com/sfr-vision-language-research/LAVIS/models/BLIP2/blip2_pretrained_flant5xxl.pth")
        img_size = cfg.get("image_size")
        num_query_token = cfg.get("num_query_token")
        llama_model = cfg.get("llama_model")

        drop_path_rate = cfg.get("drop_path_rate", 0)
        use_grad_checkpoint = cfg.get("use_grad_checkpoint", False)
        vit_precision = cfg.get("vit_precision", "fp16")
        freeze_vit = cfg.get("freeze_vit", True)
        freeze_qformer = cfg.get("freeze_qformer", True)
        low_resource = cfg.get("low_resource", False)
        device_8bit = cfg.get("device_8bit", 0)

        prompt_path = cfg.get("prompt_path", "")
        prompt_template = cfg.get("prompt_template", "")
        max_txt_len = cfg.get("max_txt_len", 32)
        end_sym = cfg.get("end_sym", '\n')
        
        frozen_llama_proj = cfg.get("frozen_llama_proj", True)
        frozen_video_Qformer = cfg.get("frozen_video_Qformer", True)
        frozen_audio_Qformer = cfg.get("frozen_audio_Qformer", True)

        llama_proj_model = cfg.get("llama_proj_model", '')
        
        fusion_header_type = cfg.get("fusion_header_type", 'seqTransf')
        max_frame_pos = cfg.get("max_frame_pos", 32)
        fusion_head_layers = cfg.get("fusion_head_layers", 2)
        num_video_query_token =  cfg.get("num_video_query_token", 32)

        equip_audio_branch= cfg.get("equip_audio_branch", True)
        num_audio_query_token =  cfg.get("num_audio_query_token", 8)
        imagebind_ckpt_path = cfg.get("imagebind_ckpt_path", '/mnt/workspace/ckpt')

        use_clip_loss = cfg.get("use_clip_loss", False)
        use_generation_loss = cfg.get("use_generation_loss", False)
        use_itm_loss = cfg.get("use_itm_loss", True)
        use_reconstruction_loss = cfg.get('use_reconstruction_loss', False)
        clip_dim_size = cfg.get("clip_dim_size", 256)
        num_videoq_hidden_layers = cfg.get('num_videoq_hidden_layers', 4)

        model_type = cfg.get("model_type", None)

        model = cls(
            vit_model=vit_model,
            q_former_model=q_former_model,
            img_size=img_size,
            drop_path_rate=drop_path_rate,
            use_grad_checkpoint=use_grad_checkpoint,
            vit_precision=vit_precision,
            freeze_vit=freeze_vit,
            freeze_qformer=freeze_qformer,
            num_query_token=num_query_token,
            llama_model=llama_model,
            prompt_path=prompt_path,
            prompt_template=prompt_template,
            max_txt_len=max_txt_len,
            end_sym=end_sym,
            low_resource=low_resource,
            device_8bit=device_8bit,
            fusion_header_type=fusion_header_type,
            max_frame_pos=max_frame_pos,
            fusion_head_layers=fusion_head_layers,
            frozen_llama_proj=frozen_llama_proj,
            frozen_video_Qformer=frozen_video_Qformer,
            frozen_audio_Qformer=frozen_audio_Qformer,
            num_video_query_token=num_video_query_token,
            num_audio_query_token = num_audio_query_token,
            imagebind_ckpt_path = imagebind_ckpt_path,
            equip_audio_branch = equip_audio_branch,
            llama_proj_model = llama_proj_model,
            use_clip_loss = use_clip_loss,
            use_generation_loss = use_generation_loss,
            use_itm_loss = use_itm_loss,
            use_reconstruction_loss = use_reconstruction_loss,
            clip_dim_size = clip_dim_size,
            model_type = model_type,
            num_videoq_hidden_layers = num_videoq_hidden_layers,
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
    #model, tokenizer = load_model_with_quantization_fallback("/mnt/h/models/DeepSeek-R1-Distill-Llama-8B")
    #print(model, tokenizer)
    #exit()
    llama_model = "/mnt/h/models/DeepSeek-R1-Distill-Llama-8B"
    prompt_path = "prompts/alignment_image.txt"
    prompt_template = '###Human: {} ###Assistant: '
    vm = VideoModelReasoning(prompt_path=prompt_path, prompt_template=prompt_template, llama_model=llama_model)
    vm = vm.cuda()

    samples = [
        {
            'image': torch.rand(2, 3, 16, 224, 224),
            'text_input': ['sample text1', 'sample text2'],
            'texts': ['sample text1', 'sample text2'],
            'type': 'video',
        }
    ]
  
    vm(samples[0])
