import os
import glob
import torch
from torch.nn import functional as F
from argparse import Namespace
from transformers import StoppingCriteria, StoppingCriteriaList

from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.processors.video_processor import load_video
from video_llama.conversation.conversation_video import default_conversation

DEVICE = 'cuda'


def init(args):
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(f'cuda:{args.gpu_id}')
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(
        vis_processor_cfg.name).from_config(vis_processor_cfg)
    return model, vis_processor


def upload_video(model, video_path, vis_processor, conv, img_list, output_video_embedding=False, output_image_embeddings=False, output_past_key_values=False, output_hidden_states=False, output_post_vis_encoder=False):
    if not isinstance(video_path, str):
        raise NotImplementedError("Non-string video paths are not implemented.")

    msg = ""
    # image = self.vis_processor(image).unsqueeze(0).to(DEVICE)
    video, msg = load_video(
        video_path=video_path,
        n_frms=16,
        #n_frms=8,
        height=224,
        width=224,
        sampling="uniform", return_msg=True
    )
    print(video_path)
    video = vis_processor.transform(video).unsqueeze(0).to(DEVICE)
    conv.append_message(conv.roles[0], "<Video><ImageHere></Video> " + msg)

    if output_image_embeddings:
        return conv, model.encode_videoQformer_visual_image(video)
    if output_video_embedding:
        return conv, model.encode_videoQformer_visual(video, output_video_embedding=True)[-1].last_hidden_state
    if output_past_key_values:
        return conv, model.encode_videoQformer_visual(video, output_video_embedding=True)[-1].past_key_values
    if output_hidden_states:
        return conv, model.encode_videoQformer_visual(video, output_video_embedding=True)[-1].hidden_states
    if output_post_vis_encoder:
        return conv, model.encode_videoQformer_visual(video, output_post_vis_encoder=True)

    image_emb, _, _ = model.encode_videoQformer_visual(video)
    img_list.append(image_emb)
    return conv, img_list

def get_text_embedding_qformer(prompts, model):
    all_embds = []
    for prompt in prompts:
        inputs = model.tokenizer(prompt,
                padding='max_length',
                truncation=True,
                max_length=320,
                return_tensors='pt').to('cuda')
        embds = model.video_Qformer.bert(
                inputs.input_ids,
                inputs.attention_mask,
                return_dict=True).last_hidden_state
        #embds = model.Qformer.bert(inputs['input_ids']).last_hidden_state
        #embds = embds[:, inputs['input_ids'].argmax(dim=-1)].squeeze(1)
        embds = F.normalize(model.text_proj(embds[:, 0, :]), dim=-1)
        all_embds.append(embds)
    return torch.cat(all_embds)

def get_all_video_embeddings(video_paths, model, vis_processor):
    video_embs = []
    convs = []
    for video_path in video_paths:
        conv = default_conversation.copy()
        conv.system = "You are able to understand the visual content that the user provides. Answer the questions from the video."
        conv, embs = upload_video(
            model, video_path, vis_processor, conv, [], output_video_embedding=True)
        convs.append(conv)
        embs = F.normalize(model.vision_proj(embs), dim=-1)
        video_embs.append(embs)
    return video_embs, convs

#def compute_itc(text_emb, video_emb):
#    sim_q2t = torch.einsum("iqf,tf->itq", video_emb, text_emb)
#    #sim_t2q = torch.einsum("tf,iqf->tiq", text_emb, video_emb)
#    sim = sim_q2t
#    dist, _ = sim.max(-1)
#    probs = F.softmax(dist)
#    return probs[:, 0]/probs[:, 1]

#def compute_itm(text_emb, video_emb):
#    sim_q2t = torch.einsum("iqf,tf->itq", video_emb, text_emb)
#    #sim_t2q = torch.einsum("tf,iqf->tiq", text_emb, video_emb)
#    sim = sim_q2t
#    dist, _ = sim.max(-1)
#    probs = F.softmax(dist)
#    return probs[:, 0]/probs[:, 1]


def embed_text_itc(model, prompt):
    inputs = model.tokenizer(prompt,
            padding='max_length',
            truncation=True,
            max_length=320,
            return_tensors='pt').to('cuda')
    embds = model.video_Qformer.bert(
            inputs.input_ids,
            inputs.attention_mask,
            return_dict=True).last_hidden_state
    embds = F.normalize(model.text_proj(embds[:, 0, :]), dim=-1)
    return embds

def compute_itc(model, prompts, video_emb):
    text_embs = []
    for prompt in prompts:
        text_embs.append(embed_text_itc(model, prompt))
    text_embs = torch.cat(text_embs)

    sim_q2t = torch.einsum("iqf,tf->itq", video_emb, text_embs)
    sim_q2t, _ = sim_q2t.max(dim=-1)
    return sim_q2t[:, 0]
    print('sim_q2t: ', sim_q2t, sim_q2t.size())
    sim_t2q = torch.einsum("tf,iqf->tiq", text_embs, video_emb)
    sim_t2q, _ = sim_t2q.max(dim=-1)
    sim_t2q = sim_t2q.transpose(0, 1)
    print('sim_t2q: ', sim_t2q, sim_t2q.size())
    sim = (sim_q2t + sim_t2q)/2
    print(sim, sim.size())
    #dist, _ = sim.max(-1)
    dist = sim
    return dist[:, 0]
    probs = F.softmax(dist, dim=-1)
    print(probs, probs.size())
    return probs[:, 0]#/probs[:, 1]

def embed_text_itm(model, prompt, video_emb):
    inputs = model.tokenizer(prompt,
            padding='max_length',
            truncation=True,
            max_length=320,
            return_tensors='pt').to('cuda')

    query_tokens = model.video_query_tokens.expand(video_emb.shape[0], -1, -1)
    query_atts = torch.ones(query_tokens.size()[:-1], dtype=torch.long).to(video_emb.device)

    attention_mask = torch.cat([query_atts, inputs.attention_mask], dim=1)
    video_atts = torch.ones(video_emb.size()[:-1], dtype=torch.long).to(video_emb.device)
    print(query_tokens.size(), query_atts.size(), attention_mask.size(), video_emb.size(), video_atts.size())

    output_itm = model.video_Qformer.bert(
            inputs.input_ids,
            query_embeds=query_tokens,
            attention_mask=attention_mask,
            encoder_hidden_states=video_emb,
            encoder_attention_mask=video_atts,
            return_dict=True)
    itm_embeddings = output_itm.last_hidden_state[:, : query_tokens.size(1), :]
    itm_logit = itm_embeddings.mean(dim=1)

    return itm_logit


def compute_itm(model, prompts, video_emb):
    text_embs = []
    for prompt in prompts:
        text_embs.append(embed_text_itm(model, prompt, video_emb))
    text_embs = torch.cat(text_embs)

    sim_q2t = torch.einsum("iqf,tf->itq", video_emb, text_embs)
    #sim_t2q = torch.einsum("tf,iqf->tiq", text_emb, video_emb)
    sim = sim_q2t
    dist, _ = sim.max(-1)
    probs = F.softmax(dist, dim=-1)
    return probs[:, 0]/probs[:, 1]

def generate_text_caption(model, vis_processor, prompt, video_path):
    conv = default_conversation.copy()
    conv.system = "You are able to understand the visual content that the user provides. Answer the questions from the video."
    conv, video_embs = upload_video(
            model, video_path, vis_processor, conv, [], output_video_embedding=True) #output_past_key_values=True)
 
    inputs = model.tokenizer(prompt,
            padding='max_length',
            truncation=True,
            max_length=320,
            return_tensors='pt').to('cuda')
    #video_atts = torch.ones(model.video_query_tokens.size()[:-1], dtype=torch.long).to(model.device)
    #video_embs = video_embs.unsqueeze(0)
    video_atts = torch.ones(video_embs.size()[:-1], dtype=torch.long).to(model.device)
    print(video_atts.size(), inputs.attention_mask.size())
    print(video_embs.size())
    #attention_mask = torch.cat([video_atts, inputs.attention_mask], dim=1)
    query_tokens = model.query_tokens.expand(video_embs.shape[0], -1, -1)
    #print(query_tokens.size(), attention_mask.size())
    print(video_embs.size(), video_atts.size())
    print(inputs.input_ids.size())
    model_kwargs = {
            "query_embeds": query_tokens,
            #"attention_mask": attention_mask, 
            "encoder_hidden_states": video_embs,
            "encoder_attention_mask": video_atts,
    }
    logits = model.video_Qformer.generate(
        inputs.input_ids,
        #attention_mask=attention_mask,
        #past_key_values=past_key_values,
        do_sample=True,
        max_length=320,
        min_length=10,
        min_new_tokens=100,
        #top_p=0.9,
        #eos_token_id=model.tokenizer.sep_token_id,
        #pad_token_id=model.tokenizer.pad_token_id,
        return_logits=True,
        return_dict=True,
        **model_kwargs
    )
    print(logits)
    print(logits.size())
    #output_ids = logits.argmax(dim=-1)#[0]
    #decoded_text = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    output_ids = logits
    decoded_text = model.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
    return decoded_text

def compute_dist(model, prompts, video_embs, mode='itc'):
    dists = []
    for video_emb in video_embs:
        if mode == 'itc':
            dist = compute_itc(model, prompts, video_emb)
        if mode == 'itm':
            dist = compute_itm(model, prompts, video_emb)
        dists.append(dist)
    return dists

def compute_video_emb_dist(query_video_emb, video_emb):
    return torch.linalg.norm(query_video_emb-video_emb)

def compute_dist_videoq(model, query_video_emb, video_embs):
    dists = []
    for video_emb in video_embs:
        dist = compute_video_emb_dist(query_video_emb, video_emb)
        dists.append(dist)
    return dists

def rank_matches(prompts, video_paths, model, vis_processor):
    #text_embs = get_text_embedding_qformer(prompts, model)
    video_embs, convs = get_all_video_embeddings(video_paths, model, vis_processor)
    dists = compute_dist(model, prompts, video_embs)

    sorted_dists = sorted(list(zip(video_paths, dists)), key=lambda x: x[1])
    for video_path, dists in sorted_dists:
        print(video_path, dists.cpu().detach().numpy().item())

def rank_matches_videoq(query_video_path, video_paths, model, vis_processor):
    query_video_embs, _ = get_all_video_embeddings([query_video_path], model, vis_processor)
    video_embs, convs = get_all_video_embeddings(video_paths, model, vis_processor)

    dists = compute_dist_videoq(model, query_video_embs[0], video_embs)
    sorted_dists = sorted(list(zip(video_paths, dists)), key=lambda x: x[1])
    for video_path, dists in sorted_dists:
        print(video_path, dists.cpu().detach().numpy().item())

def rank_matches_query_modulated(query, query_video_path, video_paths, model, vis_processor):
    #text_embs = get_text_embedding_qformer(prompts, model)
    video_embs, convs = get_all_video_embeddings(video_paths, model, vis_processor)
    #query_video_emb, convs = get_all_video_embeddings([query_video_path], model, vis_processor)

    conv = default_conversation.copy()
    conv.system = "You are able to understand the visual content that the user provides. Answer the questions from the video."
    _, query_video_emb = upload_video(model, query_video_path, vis_processor, conv, [], output_video_embedding=True)
    print(query_video_emb.size())
    #query_video_emb = query_video_emb.unsqueeze(0)
    #query_video_emb = query_video_emb[0]
    #query_text_emb = embed_text_itc(model, query)
    query_emb = embed_text_itm(model, query, query_video_emb)
    query_emb = model.vision_proj(query_emb)
    #query_emb = (query_video_emb + query_text_emb)/2
    dists = compute_dist_videoq(model, query_emb, video_embs)

    sorted_dists = sorted(list(zip(video_paths, dists)), key=lambda x: x[1])
    for video_path, dists in sorted_dists:
        print(video_path, dists.cpu().detach().numpy().item())



if __name__ == '__main__':
    #eval_config = 'eval_configs/video_llama_eval_only_vl_askyoutube_instruct_ft_no_transcripts_clip.yaml'
    #eval_config = 'eval_configs/video_llama_eval_only_vl_askyoutube_instruct_ft_clip.yaml'
    #eval_config = 'eval_configs/video_llama_eval_only_vl_askyoutube_webvid_videochatgpt_clip_instruct_ft_mistral.yaml'
    #eval_config = 'eval_configs/video_llama_eval_only_vl_askyoutube_webvid_videochatgpt_clip_instruct_ft_llm_and_clip_tr.yaml'
    eval_config = 'eval_configs/video_clip_kinetics_all_data.yaml'
    eval_config = 'eval_configs/video_clip_kinetics_msrvtt_meanqt_centercrop.yaml'

    gpu_id = 0
    model_type = 'vicuna'
    args = {'cfg_path': eval_config, 'gpu_id': gpu_id,
            'model_type': model_type, 'options': None}
    args = Namespace(**args)
    model, vis_processor = init(args)

    # Run video llama model with sample video.
    #video_paths = glob.glob('examples/*.mp4')
    video_paths = glob.glob('/mnt/g/video_caption_dataset/education/national_geographic/data/chunked_videos_30s/*/*.mp4')#[-150:-50]
    #video_paths = glob.glob('/mnt/g/video_caption_dataset/education/daily_dose_of_internet/data/chunked_videos_30s/*/*.mp4')#[-150:-50]

    query = 'What are the animals in this video?'
    query_video_path = '/home/bhavashok/experiments/askvideos_clip/data/a_zebra_in_an_african_savannah_k1I4_jvO4fI.mp4'
    rank_matches_query_modulated(query, query_video_path, video_paths, model, vis_processor)
    exit()
    #pos = 'Does this video have a dog in it?'
    #neg = 'This video has a dog in it.'
    #prompt = 'cobra'
    #prompt = '<Video>'
    
    #pos = 'This video has a bear in it.'
    #neg = 'This video does not have a bear in it.'
    #pos = 'snake'#in one corner the cobra in the other corner the mongoose'
    #pos = 'polar bear'
    #pos = 'in one corner the cobra in the other corner the mongoose'
    pos = 'snake'#snake'#polar bear'
    neg = ' '
    rank_matches([pos, neg], video_paths, model, vis_processor)
    #rank_matches_videoq(video_paths[-119], video_paths, model, vis_processor)
    #output_text = generate_text_caption(model, vis_processor, 'The ', video_paths[-119])
    #print(output_text)

    # TODO: Use video tokens for RAG here.
    # Assemble RAG payload using all the videos.
    # Answer the initial query.
