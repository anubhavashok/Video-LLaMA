import logging
import random

import torch
from torch.cuda.amp import autocast as autocast
import torch.nn as nn
from torch.nn import functional as F

from video_llama.common.registry import registry
from video_llama.models.blip2 import Blip2Base, disabled_train
from video_llama.models.modeling_llama import LlamaForCausalLM
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
from transformers import AutoModelForCausalLM, AutoTokenizer, MistralForCausalLM

# from flamingo_pytorch import PerceiverResampler
@registry.register_model("video_llama")
class VideoLLAMA(Blip2Base):
    """
    BLIP2 GPT-LLAMA model.
    """

    PRETRAINED_MODEL_CONFIG_DICT = {
        "pretrain_vicuna": "configs/models/video_llama.yaml",
        "pretrain_llama_v2": "configs/models/video_llama.yaml",
        "mistral": "configs/models/video_llama.yaml",
        "phi-2": "configs/models/video_llama.yaml",
        "phi-3": "configs/models/video_llama.yaml",
    }

    @classmethod
    def init_video_Qformer(cls, num_query_token, vision_width,num_hidden_layers =2):
        #num_hidden_layers = 4
        encoder_config = BertConfig.from_pretrained("bert-base-uncased")
        encoder_config.num_hidden_layers = num_hidden_layers
        encoder_config.encoder_width = vision_width
        # insert cross-attention layer every other block
        encoder_config.add_cross_attention = True
        encoder_config.cross_attention_freq = 1
        encoder_config.query_length = num_query_token
        Qformer = BertLMHeadModel(config=encoder_config)
        query_tokens = nn.Parameter(
            torch.zeros(1, num_query_token, encoder_config.hidden_size)
        )
        query_tokens.data.normal_(mean=0.0, std=encoder_config.initializer_range)
        return Qformer, query_tokens

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

        print('Loading Q-Former')
        self.Qformer, self.query_tokens = self.init_Qformer(
            num_query_token, self.visual_encoder.num_features
        )
        self.Qformer.cls = None
        self.Qformer.bert.embeddings.word_embeddings = None
        self.Qformer.bert.embeddings.position_embeddings = None
        for layer in self.Qformer.bert.encoder.layer:
            layer.output = None
            layer.intermediate = None
        if vit_model != 'clip_vit_L':
            self.load_from_pretrained(url_or_filename=q_former_model)

        if freeze_qformer:
            for name, param in self.Qformer.named_parameters():
                param.requires_grad = False
            self.Qformer = self.Qformer.eval()
            self.Qformer.train = disabled_train
            self.query_tokens.requires_grad = False
            logging.info("freeze Qformer")
        logging.info('Loading Q-Former Done')

        logging.info('Loading LLAMA Tokenizer')

        if self.model_type == 'phi-2' or self.model_type == 'phi-3':
            print(llama_model)
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                    llama_model, use_fast=True)
        elif self.model_type == 'mistral':
            self.llama_tokenizer = AutoTokenizer.from_pretrained(
                    llama_model, use_fast=False)
        else:
            self.llama_tokenizer = LlamaTokenizer.from_pretrained(llama_model, use_fast=False)
        if self.llama_tokenizer.pad_token is None:
            self.llama_tokenizer.pad_token = self.llama_tokenizer.unk_token 
        DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
        DEFAULT_AUDIO_PATCH_TOKEN = '<AudioHere>'
        self.llama_tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        self.llama_tokenizer.add_tokens([DEFAULT_AUDIO_PATCH_TOKEN], special_tokens=True)
        
        self.IMAGE_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_IMAGE_PATCH_TOKEN]
        self.AUDIO_PATCH_TOKEN_ID = self.llama_tokenizer.get_vocab()[DEFAULT_AUDIO_PATCH_TOKEN]

        logging.info('Loading LLAMA Model')
        self.model_type = model_type
        if self.low_resource:
            if self.model_type == 'mistral':
                self.llama_model = MistralForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map={"":torch.cuda.current_device()}
                    #device_map='auto'#{'': device_8bit}
                )#.cuda()
            elif self.model_type == 'phi-2' or self.model_type == 'phi-3':
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                        llama_model,
                        torch_dtype=torch.bfloat16,
                        load_in_8bit=True,
                        device_map={"":torch.cuda.current_device()},
                        trust_remote_code=True
                        )
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.bfloat16,
                    load_in_8bit=True,
                    device_map={"":torch.cuda.current_device()}
                    #device_map='auto'#{'': device_8bit}
                )#.cuda()
        else:
            if self.model_type == 'mistral':
                self.llama_model = MistralForCausalLM.from_pretrained(
                    llama_model,
                    torch_dtype=torch.bfloat16,
                    #attn_implementation="flash_attention_2",
                ).cuda()
            elif self.model_type == 'phi-2' or self.model_type == 'phi-3':
                self.llama_model = AutoModelForCausalLM.from_pretrained(
                        llama_model,
                        torch_dtype=torch.bfloat16,
                        trust_remote_code=True
                ).cuda()
                print(self.llama_model)
            else:
                self.llama_model = LlamaForCausalLM.from_pretrained(
                    llama_model,
                    #ignore_mismatched_sizes=True,
                    torch_dtype=torch.bfloat16,
                ).cuda()

        for name, param in self.llama_model.named_parameters():
            param.requires_grad = False
        logging.info('Loading LLAMA Done')


        logging.info('Loading LLAMA proj')
        self.llama_proj = nn.Linear(
            self.Qformer.config.hidden_size, self.llama_model.config.hidden_size
        )
        if llama_proj_model:
            print("load llama proj weight: {}".format(llama_proj_model))
            llama_proj_weight = torch.load(llama_proj_model, map_location="cpu")
            msg = self.load_state_dict(llama_proj_weight['model'], strict=False)

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

        self.video_frame_position_embedding = nn.Embedding(max_frame_pos, self.Qformer.config.hidden_size)

        self.num_video_query_token = num_video_query_token
        self.video_Qformer,self.video_query_tokens = self.init_video_Qformer(num_query_token = num_video_query_token,\
            vision_width=self.Qformer.config.hidden_size, num_hidden_layers=num_videoq_hidden_layers)
        
        #self.video_query_token_linear = nn.Linear(num_query_token * self.Qformer.config.hidden_size, self.Qformer.config.hidden_size)
        #self.video_Qformer.cls = None
        #self.video_Qformer.bert.embeddings.word_embeddings = None
        #self.video_Qformer.bert.embeddings.position_embeddings = None
        #for layer in self.video_Qformer.bert.encoder.layer:
        #    layer.output = None
        #    layer.intermediate = None


        if frozen_video_Qformer:
            #  todo frozen  llama_proj
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = False
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = False
            self.video_query_tokens.requires_grad = False
            
            logging.info('video_Qformer is frozen')
        else:
            for name, param in self.video_Qformer.named_parameters():
                param.requires_grad = True
            for name, param in self.video_frame_position_embedding.named_parameters():
                param.requires_grad = True
            self.video_query_tokens.requires_grad = True
            logging.info('video_Qformer is not frozen')

        if frozen_video_Qformer and (not frozen_audio_Qformer):
            self.train_flag = 1 # 只训练audio_Qformer
        elif not(frozen_video_Qformer) and frozen_audio_Qformer:
            self.train_flag = 0 # 训练video_Qformer
        elif not(frozen_video_Qformer) and not(frozen_audio_Qformer):
            self.train_flag = 2 # video_Qformer and AL trained
        else:
            self.train_flag = 3

        if equip_audio_branch:
            print (f'Initializing audio encoder from {imagebind_ckpt_path} ...')
            self.audio_encoder,self.audio_hidden_size = \
                imagebind_model.imagebind_huge()
            self.audio_encoder.load_state_dict(torch.load("{}/imagebind_huge.pth".format(imagebind_ckpt_path)))
            # free vision encoder
            for name, param in self.audio_encoder.named_parameters():
                param.requires_grad = False
            self.audio_encoder.eval()
            print ('audio encoder initialized.')
            
            self.num_audio_query_token = num_audio_query_token
            self.audio_Qformer,self.audio_query_tokens = self.init_video_Qformer(num_query_token = self.num_audio_query_token,\
                vision_width=self.audio_hidden_size, num_hidden_layers =2)
            self.audio_Qformer.cls = None
            self.audio_Qformer.bert.embeddings.word_embeddings = None
            self.audio_Qformer.bert.embeddings.position_embeddings = None
            for layer in self.audio_Qformer.bert.encoder.layer:
                layer.output = None
                layer.intermediate = None
            self.audio_llama_proj = nn.Linear(
                self.audio_Qformer.config.hidden_size, self.llama_model.config.hidden_size
            )
            self.audio_position_embedding = nn.Embedding(8, self.audio_hidden_size)

            if frozen_audio_Qformer:
                #  todo frozen  llama_proj
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = False
                self.audio_query_tokens.requires_grad = False
                for name, param in self.audio_llama_proj.named_parameters():
                    param.requires_grad = False
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = False
                logging.info('audio_Qformer and audio-LLAMA proj is frozen')
            else:
                for name, param in self.audio_Qformer.named_parameters():
                    param.requires_grad = True
                self.audio_query_tokens.requires_grad = True
                for name, param in self.audio_llama_proj.named_parameters():
                    param.requires_grad = True
                for name, param in self.audio_position_embedding.named_parameters():
                    param.requires_grad = True
                logging.info('audio_Qformer is not frozen')

        self.use_clip_loss = use_clip_loss
        self.use_generation_loss = use_generation_loss
        self.use_itm_loss = use_itm_loss
        self.clip_dim_size = clip_dim_size
        #if True:#use_clip_loss:
        if use_clip_loss:
            #embed_dim = 1024 #self.clip_dim_size #256
            embed_dim = self.clip_dim_size
            init_logit_scale = 0.07#np.log(1 / 0.07)
            self.temperature_parameter = nn.Parameter(torch.ones([]) * init_logit_scale)
            self.vision_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
            self.text_proj = nn.Linear(self.Qformer.config.hidden_size, embed_dim)
            for name, param in self.text_proj.named_parameters():
                param.requires_grad = True
            for name, param in self.vision_proj.named_parameters():
                param.requires_grad = True
            self.temperature_parameter.requires_grad = True
            del self.llama_model
            del self.llama_proj
            self.llama_proj = None
            self.llama_model = None
            self.itm_head = nn.Linear(self.Qformer.config.hidden_size, 2)

        #  self.audio_hidden_size
    def vit_to_cpu(self):
        self.ln_vision.to("cpu")
        self.ln_vision.float()
        self.visual_encoder.to("cpu")
        self.visual_encoder.float()

    def encode_videoQformer_visual(self, image, instructions=None, output_video_embedding=False, output_post_vis_encoder=False):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            if output_post_vis_encoder:
                return image_embeds
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )

            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # frame attention

            ## TODO: Experiment
            # TODO: Create video query tokens and add frame position embeddings.
            #video_query_tokens = self.video_query_token_linear(einops.rearrange(frame_hidden_state, 'b t q h -> b t (q h)'))
            #frame_hidden_state = frame_hidden_state.mean(2, keepdim=True)
            ## End

            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)

            # InstructBLIP style.
            if instructions:
                instruction_tokens = self.tokenizer(instructions,
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors='pt').to(image.device)
                video_query_output = self.video_Qformer.bert(
                    instruction_tokens,
                    query_embeds=video_query_tokens,
                    encoder_hidden_states=frame_hidden_state,
                    encoder_attention_mask=frame_atts,
                    use_cache=True,
                    return_dict=True,
                )
            else:
                video_query_output = self.video_Qformer.bert(
                    query_embeds=video_query_tokens,
                    encoder_hidden_states=frame_hidden_state,
                    encoder_attention_mask=frame_atts,
                    use_cache=True,
                    return_dict=True,
                    )
            #video_query_output = copy.deepcopy(video_query_output)
            past_key_values = video_query_output.past_key_values
            video_hidden = video_query_output.last_hidden_state
            #print('video_hidden.size() ', video_hidden.size())
            if self.llama_proj is not None:
                inputs_llama = self.llama_proj(video_hidden)
                atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
            else:
                inputs_llama = None
                atts_llama = None
            video_query_output.past_key_values = past_key_values
            if output_video_embedding:
                return inputs_llama, atts_llama, video_query_tokens, video_query_output, frame_hidden_state, frame_atts
        return inputs_llama, atts_llama
    
    def encode_videoQformer_visual_image(self, image, output_video_embedding=False):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
        return query_output
    
    def prompt_wrap(self, img_embeds, atts_img, prompt):
        if prompt:
            batch_size = img_embeds.shape[0]
            # print(prompt)
            p_before, p_after = prompt.split('<ImageHere>')
            p_before_tokens = self.llama_tokenizer(
                p_before, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_after_tokens = self.llama_tokenizer(
                p_after, return_tensors="pt", add_special_tokens=False).to(img_embeds.device)
            p_before_embeds = self.llama_model.model.embed_tokens(p_before_tokens.input_ids).expand(batch_size, -1, -1)
            p_after_embeds = self.llama_model.model.embed_tokens(p_after_tokens.input_ids).expand(batch_size, -1, -1)
            wrapped_img_embeds = torch.cat([p_before_embeds, img_embeds, p_after_embeds], dim=1)
            wrapped_atts_img = atts_img[:, :1].expand(-1, wrapped_img_embeds.shape[1])
            
            return wrapped_img_embeds, wrapped_atts_img
        else:
            return img_embeds, atts_img
    #  input audio shape [b t c h w] 
    def encode_audioQformer(self, audio,modality_type=ModalityType.AUDIO):
        device = audio.device
        with self.maybe_autocast():
            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio,modality_type=modality_type)
            batch_size,time_length = audio.size()[:2]


            position_ids = torch.arange(time_length, dtype=torch.long, device=device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)

            audio_position_embeddings = self.audio_position_embedding(position_ids)
            audio_imagebind_finalout = audio_imagebind_finalout + audio_position_embeddings

            audio_query_tokens = self.audio_query_tokens.expand(audio_imagebind_finalout.shape[0], -1, -1)
            frame_atts = torch.ones(audio_imagebind_finalout.size()[:-1], dtype=torch.long).to(device)

            audio_query_output = self.audio_Qformer.bert(
                query_embeds=audio_query_tokens, #[32,768]
                encoder_hidden_states=audio_imagebind_finalout,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            audio_hidden = audio_query_output.last_hidden_state

            inputs_llama = self.audio_llama_proj(audio_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(device)
    
        return inputs_llama, atts_llama

    def encode_videoQformer_audiovideo(self, image, audio):
        device = image.device
        
        # input shape b,c,t,h,w
        batch_size,_,time_length,_,_ = image.size()
        image = einops.rearrange(image, 'b c t h w -> (b t) c h w')
        with self.maybe_autocast():
            # embed image features with blip2, out: (b t) q h
            image_embeds = self.ln_vision(self.visual_encoder(image)).to(device)
            image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(device)

            query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
            query_output = self.Qformer.bert(
                query_embeds=query_tokens,
                encoder_hidden_states=image_embeds,
                encoder_attention_mask=image_atts,
                return_dict=True,
            )
            
            # add frame_pos embedding
            position_ids = torch.arange(time_length, dtype=torch.long, device=query_tokens.device)
            position_ids = position_ids.unsqueeze(0).expand(batch_size, -1)
            frame_position_embeddings = self.video_frame_position_embedding(position_ids)
            q_hidden_state = query_output.last_hidden_state

            frame_position_embeddings = frame_position_embeddings.unsqueeze(-2)
            frame_hidden_state = einops.rearrange(q_hidden_state, '(b t) q h -> b t q h',b=batch_size,t=time_length)
            frame_hidden_state = frame_position_embeddings + frame_hidden_state

            # encode audio 
            audio_feature, audio_imagebind_finalout = self.audio_encoder.get_audio_feature(audio,modality_type=ModalityType.AUDIO) # [batch,8*1,768]    8*32, 768
            audio_frame_position_embeddings = frame_position_embeddings.squeeze(-2)
            audio_feature = audio_feature + audio_frame_position_embeddings

            # frame attention a
            frame_hidden_state =  einops.rearrange(frame_hidden_state, 'b t q h -> b (t q) h',b=batch_size,t=time_length)
            frame_hidden_state = torch.cat([frame_hidden_state,audio_feature],dim = 1)
            video_query_tokens = self.video_query_tokens.expand(frame_hidden_state.shape[0], -1, -1)
            frame_atts = torch.ones(frame_hidden_state.size()[:-1], dtype=torch.long).to(device)

            video_query_output = self.video_Qformer.bert(
                query_embeds=video_query_tokens, #[32,768]
                encoder_hidden_states=frame_hidden_state,
                encoder_attention_mask=frame_atts,
                return_dict=True,
                )
            video_hidden = video_query_output.last_hidden_state

            inputs_llama = self.llama_proj(video_hidden)
            atts_llama = torch.ones(inputs_llama.size()[:-1], dtype=torch.long).to(image_embeds.device)
    
        return inputs_llama, atts_llama

    def compute_clip_loss_blip(self, image_feats_all, text_feat_all):
        sim_q2t = torch.einsum("iqf,tf->itq", image_feats_all, text_feat_all)
        #sim_q2t = torch.matmul(
        #    image_feats_all.unsqueeze(1), text_feat_all.unsqueeze(-1)
        #).squeeze()
        # [batch_size, batch_size*num_gpu, num_query_tokens]

        # image-text similarity: aggregate across all query tokens
        sim_i2t, _ = sim_q2t.max(-1)
        #sim_i2t = sim_q2t.mean(-1)
        sim_i2t = sim_i2t / self.temperature_parameter

        # text-query similarity: [batch_size, batch_size*num_gpu, num_query_tokens]
        # text_feat_all = [b, 256]
        # image_feats_all = [b, 32, 256]
        sim_t2q = torch.einsum("tf,iqf->tiq", text_feat_all, image_feats_all)
        #sim_t2q = torch.matmul(
        #    text_feat_all.unsqueeze(1).unsqueeze(1), image_feats_all.permute(0, 2, 1)
        #).squeeze()

        # text-image similarity: aggregate across all query tokens
        sim_t2i, _ = sim_t2q.max(-1)
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


    def compute_clip_loss(self, video_embeds, text_embeds):
        # Compute weighted img_embeds?
        #video_embeds = img_embeds.mean(axis=2)
        #print(video_embeds.size(), text_embeds.size())
        normalized_video_embeds = video_embeds.norm(dim=1, p=2)
        normalized_text_embeds = text_embeds.norm(dim=1, p=2)

        logits = torch.matmul(video_embeds.float(), text_embeds.float().transpose(1, 0)) * torch.exp(self.temperature_parameter)
        #print(logits.size())
        batch_size = text_embeds.size(0)
        labels = torch.arange(batch_size).to(video_embeds.device)
        
        loss_video = F.cross_entropy(logits, labels)
        loss_text = F.cross_entropy(logits, labels)
        loss = (loss_video + loss_text) / 2
        return loss

    def text_global_pool(self, text_embs, input_ids):
        #print(input_ids.size())
        idxs = input_ids.argmax(dim=1)
        #print(idxs)
        eot_embs = text_embs[:, idxs, :]
        #print(eot_embs.size())
        return eot_embs[:, 0, :]

    def image_global_pool(self, image_embs):
        # We just take the first token for some reason in CLIP.
        return image_embs[:, 0, :]

    def forward(self, samples):
        if 'conv_type' in samples.keys() and samples['conv_type']=='multi':
            im_patch_token_id = self.IMAGE_PATCH_TOKEN_ID
            image = samples["images"]
            input_ids = samples['input_ids']
            texts = samples['texts']
            if len(image.size())==4:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)

            if self.train_flag == 0:
                num_patch_tokens = self.num_video_query_token
                #img_embeds, atts_img = self.encode_videoQformer_visual(image)
                img_embeds, atts_img, video_query_tokens, video_query_output, frame_hidden_state, frame_atts = self.encode_videoQformer_visual(image, output_video_embedding=True)
                video_embedding = video_query_output.last_hidden_state
            elif self.train_flag == 1:
                num_patch_tokens = self.num_audio_query_token
                image = einops.rearrange(image, 'b c t h w -> b t c h w')
                img_embeds, atts_img = self.encode_audioQformer(image, modality_type=ModalityType.VISION)
            
            temp_input_ids = copy.deepcopy(input_ids)
            temp_input_ids[temp_input_ids == im_patch_token_id] = 0
            temp_input_embedding = self.llama_model.model.embed_tokens(temp_input_ids)

            # TODO: Insert CLIP loss here between img_embeds and text.
            # clip_text_feature = self.global_pool(temp_input_embedding, input_ids)
            # if self.use_clip_loss:
            #     clip_loss = self.compute_clip_loss(img_embeds, att_img, temp_input_embedding)
            if self.use_clip_loss:
                #with self.maybe_autocast():
                #    text_only_outputs = self.llama_model(
                #        inputs_embeds=temp_input_embedding,
                #        attention_mask=attention_mask,
                #        return_dict=True,
                #        labels=None,
                #    )
                #    clip_text_feature = self.text_global_pool(text_only_outputs.hidden_states, input_ids)
                # Image text matching.
                #texts = samples['texts']
                texts = [text.split('</Video>')[1] for text in texts]
                texts = ['seconds. '.join(text.split('seconds. ')[1:]) for text in texts]
                blip_text_tokens = self.tokenizer(texts,
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors='pt').to(image.device)
                clip_text_feature = self.video_Qformer.bert(
                        blip_text_tokens.input_ids,
                        blip_text_tokens.attention_mask,
                        return_dict=True)
                clip_text_feature = clip_text_feature.last_hidden_state
                clip_text_feature = clip_text_feature[:, 0, :]
                #clip_text_feature = clip_text_feature[:, inputs['input_ids'].argmax(dim=-1)].squeeze(1)
                clip_text_feature = F.normalize(self.text_proj(clip_text_feature), dim=-1)
                clip_image_feature = F.normalize(self.vision_proj(video_embedding), dim=-1)
                #clip_image_feature = self.image_global_pool(img_embeds)
                clip_loss, sim_i2t, sim_t2i = self.compute_clip_loss_blip(clip_image_feature, clip_text_feature)
            if self.use_itm_loss:
                # TODO: Check that sim_i2t and sim_t2i exist.
                # TODO: Check that blip_text_tokens exists.
                decoder_input_ids = blip_text_tokens.input_ids.clone()
                query_atts = torch.ones(video_query_tokens.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                # TODO: Get labels for pairs.
                bs = image.size(0)
                # Sample according to predicted similarity score.
                sim_t2i[:, :].fill_diagonal_(-10000)
                sim_i2t[:, :].fill_diagonal_(-10000)
                weights_t2i = F.softmax(sim_t2i, dim=1)
                weights_i2t = F.softmax(sim_i2t, dim=1)

                # select a negative video for each text
                video_embeds_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                    # Note: here image_embs is actually the frame_hidden_state
                    # This is what goes into the Video QFormer as encoder hidden states
                    video_embeds_neg.append(frame_hidden_state[neg_idx])
                video_embeds_neg = torch.stack(video_embeds_neg, dim=0)

                # select a negative text for each video
                text_ids_neg = []
                text_atts_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                    text_ids_neg.append(decoder_input_ids[neg_idx])
                    text_atts_neg.append(blip_text_tokens.attention_mask[neg_idx])

                # Assemble texts for ITM.
                text_ids_neg = torch.stack(text_ids_neg, dim=0)
                text_atts_neg = torch.stack(text_atts_neg, dim=0)
                text_ids_all = torch.cat(
                    [blip_text_tokens.input_ids, blip_text_tokens.input_ids, text_ids_neg], dim=0
                )  # pos, pos, neg
                text_atts_all = torch.cat(
                    [blip_text_tokens.attention_mask, blip_text_tokens.attention_mask, text_atts_neg],
                    dim=0,
                )

                # Assemble videos for ITM.
                video_query_tokens_itm = self.video_query_tokens.expand(text_atts_all.shape[0], -1, -1)
                query_atts_itm = torch.ones(video_query_tokens_itm.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

                video_embeds_all = torch.cat(
                    [frame_hidden_state, video_embeds_neg, frame_hidden_state], dim=0
                )  # pos, neg, pos
                video_atts_all = torch.ones(video_embeds_all.size()[:-1], dtype=torch.long).to(
                    image.device
                )

                output_itm = self.video_Qformer.bert(
                    text_ids_all,
                    query_embeds=video_query_tokens_itm,
                    attention_mask=attention_mask_all,
                    encoder_hidden_states=video_atts_all,
                    encoder_attention_mask=video_atts_all,
                    return_dict=True)

                itm_embeddings = output_itm.last_hidden_state[:, : video_query_tokens_itm.size(1), :]

                itm_logit = self.itm_head(itm_embeddings)
                itm_logit = itm_logit.mean(dim=1)
                itm_labels = torch.cat(
                    [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                    dim=0,
                ).to(image.device)
                itm_loss = F.cross_entropy(itm_logit, itm_labels)
            if self.use_generation_loss:
                # Image captioning.
                decoder_input_ids = blip_text_tokens.input_ids.clone()
                #decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
                labels = decoder_input_ids.masked_fill(
                    decoder_input_ids == self.tokenizer.pad_token_id, -100
                )
                # TODO: See where query_tokens is.
                query_atts = torch.ones(video_query_tokens.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                attention_mask = torch.cat([query_atts, blip_text_tokens.attention_mask], dim=1)
                #print(decoder_input_ids.size())
                #print(attention_mask.size())
                #print(video_query_output.past_key_values[0].size())
                lm_output = self.video_Qformer(
                        decoder_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=video_query_output.past_key_values,
                        return_dict=True,
                        labels=labels,
                )
                caption_loss = lm_output.loss
                #print('Texts: \n', texts)
                #print('clip_loss: ', clip_loss)
                #print('caption_loss: ', clip_loss)

            losses = None
            if self.use_clip_loss:
                losses = clip_loss
                if self.use_itm_loss:
                    losses += itm_loss
                if self.use_generation_loss:
                    losses += caption_loss
                return {'loss': losses}
            
            #exit()

            new_input_embeds=[]
            cur_image_idx = 0
            for cur_input_ids, cur_input_embeds in zip(input_ids, temp_input_embedding):
                cur_image_features = img_embeds[cur_image_idx]

                if (cur_input_ids == im_patch_token_id).sum() != num_patch_tokens:
                        raise ValueError("The number of image patch tokens should be the same as the number of image patches.")
                masked_indices = torch.where(cur_input_ids == im_patch_token_id)[0]
                mask_index_start = masked_indices[0]
                if (masked_indices != torch.arange(mask_index_start, mask_index_start+num_patch_tokens, device=masked_indices.device, dtype=masked_indices.dtype)).any():
                    raise ValueError("The image patch tokens should be consecutive.")
                
                cur_new_input_embeds = torch.cat((cur_input_embeds[:mask_index_start], cur_image_features, cur_input_embeds[mask_index_start+num_patch_tokens:]), dim=0)
                new_input_embeds.append(cur_new_input_embeds)
                
                cur_image_idx+=1
            inputs_embeds = torch.stack(new_input_embeds, dim=0)
            targets = samples['labels']
            #print('targets: ', targets)
            attention_mask = samples['attention_mask']
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            if self.use_clip_loss:
                loss = outputs.loss + clip_loss + caption_loss
            else:
                loss = outputs.loss
            #print('loss: ', loss)
            return {"loss": loss}
        else:
            image = samples["image"]

            if len(image.size()) != 5:
                time = 1
                image = einops.repeat(image, 'b c h w -> b c t h w',t = time)
            
            if self.train_flag == 1:
                image = einops.rearrange(image, 'b c t h w -> b t c h w')
                img_embeds, atts_img = self.encode_audioQformer(image, modality_type=ModalityType.VISION)
            else:
                #img_embeds, atts_img = self.encode_videoQformer_visual(image)
                img_embeds, atts_img, video_query_tokens, video_query_output, frame_hidden_state, frame_atts = self.encode_videoQformer_visual(image, output_video_embedding=True)
                video_embedding = video_query_output.last_hidden_state

               
            if self.use_clip_loss:
                #with self.maybe_autocast():
                #    text_only_outputs = self.llama_model(
                #        inputs_embeds=temp_input_embedding,
                #        attention_mask=attention_mask,
                #        return_dict=True,
                #        labels=None,
                #    )
                #    clip_text_feature = self.text_global_pool(text_only_outputs.hidden_states, input_ids)
                # Image text matching.
                texts = samples['texts']
                #texts = [text.split('</Video>')[1] for text in texts]
                #texts = ['seconds. '.join(text.split('seconds. ')[1:]) for text in texts]
                blip_text_tokens = self.tokenizer(texts,
                        padding='max_length',
                        truncation=True,
                        max_length=self.max_txt_len,
                        return_tensors='pt').to(image.device)
                clip_text_feature = self.video_Qformer.bert(
                        blip_text_tokens.input_ids,
                        blip_text_tokens.attention_mask,
                        return_dict=True)
                clip_text_feature = clip_text_feature.last_hidden_state
                #print('clip_text_feature.size(): ', clip_text_feature.size())
                clip_text_feature = clip_text_feature[:, 0, :]
                #clip_text_feature = clip_text_feature[:, -1, :]
                #clip_text_feature = clip_text_feature[:, inputs['input_ids'].argmax(dim=-1)].squeeze(1)
                clip_text_feature = F.normalize(self.text_proj(clip_text_feature), dim=-1)
                clip_image_feature = F.normalize(self.vision_proj(video_embedding), dim=-1)
                #clip_image_feature = self.image_global_pool(img_embeds)
                clip_loss, sim_i2t, sim_t2i = self.compute_clip_loss_blip(clip_image_feature, clip_text_feature)
            if self.use_itm_loss:
                # TODO: Check that sim_i2t and sim_t2i exist.
                # TODO: Check that blip_text_tokens exists.
                decoder_input_ids = blip_text_tokens.input_ids.clone()
                query_atts = torch.ones(video_query_tokens.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                # TODO: Get labels for pairs.
                bs = image.size(0)
                # Sample according to predicted similarity score.
                sim_t2i[:, :].fill_diagonal_(-10000)
                sim_i2t[:, :].fill_diagonal_(-10000)
                weights_t2i = F.softmax(sim_t2i, dim=1).detach()
                weights_i2t = F.softmax(sim_i2t, dim=1).detach()

                # select a negative video for each text
                video_embeds_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_t2i[b], 1).item()
                    # Note: here image_embs is actually the frame_hidden_state
                    # This is what goes into the Video QFormer as encoder hidden states
                    video_embeds_neg.append(frame_hidden_state[neg_idx])
                video_embeds_neg = torch.stack(video_embeds_neg, dim=0)

                # select a negative text for each video
                text_ids_neg = []
                text_atts_neg = []
                for b in range(bs):
                    neg_idx = torch.multinomial(weights_i2t[b], 1).item()
                    text_ids_neg.append(decoder_input_ids[neg_idx])
                    text_atts_neg.append(blip_text_tokens.attention_mask[neg_idx])

                # Assemble texts for ITM.
                text_ids_neg = torch.stack(text_ids_neg, dim=0)
                text_atts_neg = torch.stack(text_atts_neg, dim=0)

                text_ids_all = torch.cat(
                    [blip_text_tokens.input_ids, blip_text_tokens.input_ids, text_ids_neg], dim=0
                )  # pos, pos, neg
                text_atts_all = torch.cat(
                    [blip_text_tokens.attention_mask, blip_text_tokens.attention_mask, text_atts_neg],
                    dim=0,
                )

                # Assemble videos for ITM.
                video_query_tokens_itm = self.video_query_tokens.expand(text_atts_all.shape[0], -1, -1)
                query_atts_itm = torch.ones(video_query_tokens_itm.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                attention_mask_all = torch.cat([query_atts_itm, text_atts_all], dim=1)

                video_embeds_all = torch.cat(
                    [frame_hidden_state, video_embeds_neg, frame_hidden_state], dim=0
                )  # pos, neg, pos
                video_atts_all = torch.ones(video_embeds_all.size()[:-1], dtype=torch.long).to(
                    image.device
                )

                output_itm = self.video_Qformer.bert(
                    text_ids_all,
                    query_embeds=video_query_tokens_itm,
                    attention_mask=attention_mask_all,
                    encoder_hidden_states=video_embeds_all,
                    encoder_attention_mask=video_atts_all,
                    return_dict=True)

                itm_embeddings = output_itm.last_hidden_state[:, : video_query_tokens_itm.size(1), :]

                itm_logit = self.itm_head(itm_embeddings)
                itm_logit = itm_logit.mean(dim=1)
                itm_labels = torch.cat(
                    [torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                    dim=0,
                ).to(image.device)
                itm_loss = F.cross_entropy(itm_logit, itm_labels)
            if self.use_generation_loss:
                # Image captioning.
                decoder_input_ids = blip_text_tokens.input_ids.clone()
                #print('BOS token id: ', self.tokenizer.bos_token_id)
                #decoder_input_ids[:, 0] = self.tokenizer.bos_token_id
                labels = decoder_input_ids.masked_fill(
                    decoder_input_ids == self.tokenizer.pad_token_id, -100
                ).to(image.device)
                query_atts = torch.ones(video_query_tokens.size()[:-1], dtype=torch.long).to(
                    image.device
                )
                attention_mask = torch.cat([query_atts, blip_text_tokens.attention_mask], dim=1).to(image.device)
                #print(decoder_input_ids.size())
                #print(attention_mask.size())
                #print(len(video_query_output.past_key_values))
                #print(len(video_query_output.past_key_values[0]))
                #print(video_query_output.past_key_values[0][0].size())
                lm_output = self.video_Qformer(
                        decoder_input_ids,
                        attention_mask=attention_mask,
                        past_key_values=video_query_output.past_key_values,
                        return_dict=True,
                        labels=labels,
                )
                #output_ids = lm_output.logits.argmax(dim=-1)
                #output_text = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
                #print('Original text: ', texts)
                #print('Output: ', output_text)
                caption_loss = lm_output.loss
                #print('Texts: \n', texts)
                #print('clip_loss: ', clip_loss)
                #print('caption_loss: ', caption_loss)

            losses = None
            if self.use_clip_loss:
                losses = clip_loss
                if self.use_itm_loss:
                    losses += itm_loss
                if self.use_generation_loss:
                    losses += caption_loss
                return {'loss': losses}

            if self.prompt_list:
                prompt = random.choice(self.prompt_list)
                img_embeds, atts_img = self.prompt_wrap(img_embeds, atts_img, prompt)
 
            self.llama_tokenizer.padding_side = "left" if self.model_type == 'phi-3' else "right"

            text = [t + self.end_sym for t in samples["text_input"]]

            to_regress_tokens = self.llama_tokenizer(
                text,
                return_tensors="pt",
                padding="longest",
                truncation=True,
                max_length=self.max_txt_len,
                add_special_tokens=False
            ).to(image.device)

            targets = to_regress_tokens.input_ids.masked_fill(
                to_regress_tokens.input_ids == self.llama_tokenizer.pad_token_id, -100
            )

            empty_targets = (
                torch.ones([atts_img.shape[0], atts_img.shape[1]+1],
                        dtype=torch.long).to(image.device).fill_(-100)  # plus one for bos
            )
            targets = torch.cat([empty_targets, targets], dim=1)
            #print('targets: ', targets)

            batch_size = img_embeds.shape[0]
            bos = torch.ones([batch_size, 1],
                            dtype=to_regress_tokens.input_ids.dtype,
                            device=to_regress_tokens.input_ids.device) * self.llama_tokenizer.bos_token_id
            bos_embeds = self.llama_model.model.embed_tokens(bos)
            atts_bos = atts_img[:, :1]

            to_regress_embeds = self.llama_model.model.embed_tokens(to_regress_tokens.input_ids)
            inputs_embeds = torch.cat([bos_embeds, img_embeds, to_regress_embeds], dim=1)
            attention_mask = torch.cat([atts_bos, atts_img, to_regress_tokens.attention_mask], dim=1)

            #print('to_regress_embeds: ', to_regress_embeds)
            #print('inputs_embeds: ', inputs_embeds)
            #print('attention_mask: ', attention_mask)
            with self.maybe_autocast():
                outputs = self.llama_model(
                    inputs_embeds=inputs_embeds,
                    attention_mask=attention_mask,
                    return_dict=True,
                    labels=targets,
                )
            #print('logits: ', outputs.logits)
            #print('outputs: ', outputs.__dict__.keys())
            #print('loss: ', loss)
            if self.use_clip_loss:
                loss = outputs.loss + clip_loss + caption_loss
            else:
                loss = outputs.loss
            #print('loss: ', loss)
            return {"loss": loss}
 
        return {"loss": loss}

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
