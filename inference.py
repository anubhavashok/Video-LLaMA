from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.processors.video_processor import ToTHWC, ToUint8, load_video
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation, SeparatorStyle, conv_llava_llama_2
from argparse import Namespace
import os
import torch
import glob
from transformers import StoppingCriteria, StoppingCriteriaList
from torch.nn import functional as F

device = 'cuda'


def init(args):
    cfg = Config(args)
    model_config = cfg.model_cfg
    model_config.device_8bit = args.gpu_id
    model_cls = registry.get_model_class(model_config.arch)
    model = model_cls.from_config(model_config).to(
        'cuda:{}'.format(args.gpu_id))
    model.eval()
    vis_processor_cfg = cfg.datasets_cfg.webvid.vis_processor.train
    vis_processor = registry.get_processor_class(
        vis_processor_cfg.name).from_config(vis_processor_cfg)
    return model, vis_processor


def get_context_emb(model, conv, img_list):
    prompt = conv.get_prompt()
    print(prompt)
    prompt_segs = prompt.split('<ImageHere>')
    assert len(prompt_segs) == len(img_list) + \
        1, "Unmatched numbers of image placeholders and images."
    seg_tokens = [
        model.llama_tokenizer(
            seg, return_tensors="pt", add_special_tokens=i == 0).to(device).input_ids
        # only add bos to the first seg
        for i, seg in enumerate(prompt_segs)
    ]
    seg_embs = [model.llama_model.model.embed_tokens(
        seg_t) for seg_t in seg_tokens]
    mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list)
                  for emb in pair] + [seg_embs[-1]]
    mixed_embs = torch.cat(mixed_embs, dim=1)
    return mixed_embs


class StoppingCriteriaSub(StoppingCriteria):

    def __init__(self, stops=[], encounters=1):
        super().__init__()
        self.stops = stops

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor):
        for stop in self.stops:
            if torch.all((stop == input_ids[0][-len(stop):])).item():
                return True

        return False


def inference(prompt, model, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
              repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
    # TODO: Add prompt to conv.
    #conv.append_message(conv.roles[1], None)
    conv = default_conversation.copy()
    conv.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
    # video_emb = upload_video(
    #     model, video_path, vis_processor, conv, [], output_llama_embedding=True)
    conv, img_list = upload_video(
        model, video_path, vis_processor, conv, [])
    conv = ask(prompt, conv)
    embs = get_context_emb(model, conv, img_list)

    current_max_len = embs.shape[1] + max_new_tokens
    if current_max_len - max_length > 0:
        print('Warning: The number of tokens in current conversation exceeds the max length. '
              'The model will not see the contexts outside the range.')
    begin_idx = max(0, current_max_len - max_length)

    embs = embs[:, begin_idx:]

    if conv.sep == "###":
        stop_words_ids = [torch.tensor([835]).to(device),
                          torch.tensor([2277, 29937]).to(device)]  # '###' can be encoded in two different ways.
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])
    else:
        stop_words_ids = [torch.tensor([2]).to(device)]
        stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])

    outputs = model.llama_model.generate(
        inputs_embeds=embs,
        max_new_tokens=max_new_tokens,
        stopping_criteria=stopping_criteria,
        num_beams=num_beams,
        do_sample=True,
        min_length=min_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        temperature=temperature,
    )
    output_token = outputs[0]
    # the model might output a unknow token <unk> at the beginning. remove it
    if output_token[0] == 0:
        output_token = output_token[1:]
    # some users find that there is a start token <s> at the beginning. remove it
    if output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(
        output_token, add_special_tokens=False)
    if conv.sep == "###":
        output_text = output_text.split('###')[0]  # remove the stop sign '###'
        output_text = output_text.split('Assistant:')[-1].strip()
    else:
        output_text = output_text.split(
            conv.sep2)[0]  # remove the stop sign '###'
        output_text = output_text.split(conv.roles[1]+':')[-1].strip()
    conv.messages[-1][1] = output_text
    return output_text


def upload_video(model, video_path, vis_processor, conv, img_list, output_video_embedding=False, output_image_embeddings=False):
    msg = ""
    if isinstance(video_path, str):  # is a video path
        ext = os.path.splitext(video_path)[-1].lower()
        print(video_path)
        # image = self.vis_processor(image).unsqueeze(0).to(self.device)
        video, msg = load_video(
            video_path=video_path,
            n_frms=4,
            height=224,
            width=224,
            sampling="uniform", return_msg=True
        )
        video = vis_processor.transform(video)
        video = video.unsqueeze(0).to(device)
    else:
        raise NotImplementedError

    conv.append_message(conv.roles[0], "<Video><ImageHere></Video> " + msg)
    if output_image_embeddings:
        return conv, model.encode_videoQformer_visual_image(video)
    if output_video_embedding:
        return conv, model.encode_videoQformer_visual(video, output_video_embedding=True)[-1]
    image_emb, _, _ = model.encode_videoQformer_visual(video)
    img_list.append(image_emb)
    return conv, img_list

def get_text_embedding_qformer(prompt, model):
    inputs = model.tokenizer(prompt, return_tensors='pt').to('cuda')
    embds = model.video_Qformer.bert(inputs['input_ids']).last_hidden_state
    #embds = model.Qformer.bert(inputs['input_ids']).last_hidden_state
    embds = F.normalize(model.text_proj(embds[:, 0, :]), dim=-1)
    return embds

def get_text_embedding(prompt, model):
    tokens = model.llama_tokenizer(prompt, return_tensors="pt", add_special_tokens=True).to(device).input_ids
    embds = model.video_Qformer.bert(tokens).last_hidden_state
    return embds[:, 0, :]
    embds = model.llama_model.model.embed_tokens(tokens)
    # TODO: Run through llm.
    stop_words_ids = [torch.tensor([835]).to(device),
                          torch.tensor([2277, 29937]).to(device)]  # '###' can be encoded in two different ways.
    stopping_criteria = StoppingCriteriaList(
            [StoppingCriteriaSub(stops=stop_words_ids)])

    outputs = model.llama_model(
        inputs_embeds=embds,
        #max_new_tokens=200,
        #stopping_criteria=stopping_criteria,
        output_hidden_states=True,
    )
    print(outputs.hidden_states.size())
    return outputs.hidden_states[:, -1, :]
    output_token = outputs[0]
    # the model might output a unknow token <unk> at the beginning. remove it
    if output_token[0] == 0:
        output_token = output_token[1:]
    # some users find that there is a start token <s> at the beginning. remove it
    if output_token[0] == 1:
        output_token = output_token[1:]
    output_text = model.llama_tokenizer.decode(
        output_token, add_special_tokens=False)
    return output_text

def get_all_video_embeddings(prompt, video_paths, model, vis_processor, raw=True):
    video_embs = []
    convs = []
    for video_path in video_paths:
        conv = default_conversation.copy()
        #conv.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        conv.system = "You are able to understand the visual content that the user provides. Answer the questions from the video."
        # video_emb = upload_video(
        #     model, video_path, vis_processor, conv, [], output_llama_embedding=True)

        conv, embs = upload_video(
            model, video_path, vis_processor, conv, [], output_video_embedding=True)

        # conv = ask(prompt, conv)
        #print(embs.size())
        #exit()
        # Single video, safe to take first embedding.
        #embs = embs[0]
        convs.append(conv)
        if raw:
            print(embs.size())
            #embs = embs[0]
            #video_embs.append(embs[:, -1, :])
            embs = F.normalize(model.vision_proj(embs), dim=-1)
            video_embs.append(embs)
            # conv, img_list = upload_video(
            #     model, video_path, vis_processor, conv, [])
            # print(img_list[0].size())
            # # exit()
            # conv = ask(prompt, conv)
            # embs = get_context_emb(model, conv, img_list)
            # print(embs.size())
        else:
            #embs = get_context_emb(model, conv, embs)
            embs = embs[0]
            outputs = model.llama_model(
                inputs_embeds=embs,
                #max_new_tokens=200,
                #stopping_criteria=stopping_criteria,
                output_hidden_states=True,
            )
            print(outputs.hidden_states.size())
            embs = outputs.hidden_states[:, -1, :] 
            #embs = outputs.hidden_states.mean(1)
            video_embs.append(embs)
    return video_embs, convs


def compute_dist(emb, other_embs):
    dists = []
    for other_emb in other_embs:
        dist = torch.norm(emb - other_emb, p=2)
        dists.append(dist)
    return dists

def compute_dist_clip(text_emb, video_embs):
    dists = []
    #text_emb = F.normalize(text_emb, dim=-1)
    #video_embs = [F.normalize(video_emb, dim=-1) for video_emb in video_embs]
    for video_emb in video_embs:
        print('video_emb.size(): ', video_emb.size())
        print('text_emb.size(): ', text_emb.size())
        #sim = torch.matmul(text_emb.unsqueeze(1), video_emb.permute(0, 2, 1))
        #sim = torch.matmul(video_emb.unsqueeze(1), text_emb.unsqueeze(-1))
        #sim = sim.squeeze()
        sim_q2t = torch.einsum("iqf,tf->itq", video_emb, text_emb)
        sim_t2q = torch.einsum("tf,iqf->tiq", text_emb, video_emb)
        #sim = (sim_q2t + sim_t2q)/2
        sim = sim_q2t
        #sim = (sim_q2t.max(-1)[0] + sim_t2q.max(-1)[0])/2
        #print(sim.size())
        dist = sim.max(-1)
        #dist = sim.mean(-1)
        #dist = sim
        dists.append(dist)
    return dists



def compute_dist_mean(emb, other_embs, cosine=True):
    dists = []
    for other_emb in other_embs:
        print('other_emb.size(): ', other_emb.size())
        print('emb.size(): ', emb.size())
        if cosine:
            dist = torch.nn.functional.cosine_similarity(emb, other_emb)
        else:
            dist = torch.norm(emb - other_emb, p=2)
            #dist = torch.norm(emb.mean(0) - other_emb.mean(0), p=2)
        dists.append(dist)
    return dists


def ask(text, conv):
    print(len(conv.messages) > 0)
    print(conv.messages[-1][0] == conv.roles[0])
    print(('</Video>' in conv.messages[-1][1]
          or '</Image>' in conv.messages[-1][1]))
    if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
            and ('</Video>' in conv.messages[-1][1] or '</Image>' in conv.messages[-1][1]):  # last message is image.
        conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
    else:
        conv.append_message(conv.roles[0], text)
    return conv

def create_payload(model, convs, img_lists):
    # TODO: On upload video, combine the messages to assemble joint payload.
    # TODO: For each video in img_list, assemble with video title.
    combined_conv = copy.deepcopy(convs[0])
    combined_emb = torch.Tensor()
    for conv in convs:
        prompt = conv.get_prompt()
        print(prompt)
        prompt_segs = prompt.split('<ImageHere>')
        assert len(prompt_segs) == len(img_list) + \
            1, "Unmatched numbers of image placeholders and images."
        seg_tokens = [
            model.llama_tokenizer(
                seg, return_tensors="pt", add_special_tokens=i == 0).to(device).input_ids
            # only add bos to the first seg
            for i, seg in enumerate(prompt_segs)
        ]
        seg_embs = [model.llama_model.model.embed_tokens(
            seg_t) for seg_t in seg_tokens]
        mixed_embs = [emb for pair in zip(seg_embs[:-1], img_list)
                for emb in pair] + [seg_embs[-1]]
        mixed_embs = torch.cat(mixed_embs, dim=1)
        combined_emb = torch.cat(combined_emb, mixed_embs)
    return combined_embs


if __name__ == '__main__':
    # TODO: Load multiple videos and get video closest to query.
    video_path = 'examples/applausing.mp4'
    #eval_config = 'eval_configs/video_llama_eval_only_vl.yaml'
    #eval_config = 'eval_configs/video_llama_eval_only_vl_askyoutube_instruct_ft_no_transcripts.yaml'
    eval_config = 'eval_configs/video_llama_eval_only_vl_askyoutube_instruct_ft_no_transcripts_clip.yaml'
    gpu_id = 0
    model_type = 'vicuna'
    args = {'cfg_path': eval_config, 'gpu_id': gpu_id,
            'model_type': model_type, 'options': None}
    args = Namespace(**args)

    model, vis_processor = init(args)
    # Run video llama model with sample video.
    #video_paths = glob.glob('examples/*.mp4')
    video_paths = glob.glob('/mnt/g/video_caption_dataset/education/national_geographic/data/chunked_videos_30s/*/*.mp4')[-160:-40]
    print(video_paths)
    #video_paths = video_paths[:100]
    #prompt = 'Does this video have a dog in it?'
    #prompt = 'This video has a dog in it.'
    prompt = 'This video has a cobra in it.'
    #prompt = 'cobra'
    #prompt = '<Video>'
    text_emb = get_text_embedding_qformer(prompt, model)
    print(text_emb.size())
    video_embs, convs = get_all_video_embeddings(prompt,
                                          video_paths, model, vis_processor)
    #dists = compute_dist_mean(text_emb, video_embs)
    dists = compute_dist_clip(text_emb, video_embs)
    print(sorted(list(zip(video_paths, dists)), key=lambda x: x[1]))
    # TODO: Use video tokens for RAG here.
    # Assemble RAG payload using all the videos.
    # Answer the initial query.
    exit()
    for i, v_emb_i in enumerate(video_embs):
        print(i, compute_dist_mean(v_emb_i, video_embs))
    exit()
    # Try to extract aggregated embedding.
    output = inference(prompt, model)
    print(output)
