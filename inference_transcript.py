from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.processors.video_processor import ToTHWC, ToUint8, load_video
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation, SeparatorStyle, conv_llava_llama_2
from argparse import Namespace
import os
import torch
print(torch.cuda.device_count())
import glob
from transformers import StoppingCriteria, StoppingCriteriaList
import json

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
    #print('mixed_embs.size() ', mixed_embs.size())
    #exit()
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


def inference(prompt, video_path, model, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
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
    # TODO: Outputs[1] should be attention maps.
    # Check what the attention scores on img embs are like here.
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


def upload_video(model, video_path, vis_processor, conv, img_list, output_video_embedding=False, output_llama_embedding=False):
    msg = ""
    if isinstance(video_path, str):  # is a video path
        ext = os.path.splitext(video_path)[-1].lower()
        print(video_path)
        # image = self.vis_processor(image).unsqueeze(0).to(self.device)
        video, msg = load_video(
            video_path=video_path,
            n_frms=32,
            height=224,
            width=224,
            sampling="uniform", return_msg=True
        )
        video = vis_processor.transform(video)
        video = video.unsqueeze(0).to(device)
    else:
        raise NotImplementedError

    if output_video_embedding:
        return model.encode_videoQformer_visual(video, output_video_embedding=True)
    if output_llama_embedding:
        return model.encode_videoQformer_visual(video, output_llama_embedding=True)
    image_emb, _ = model.encode_videoQformer_visual(video)
    #print('image_emb.size() ', image_emb.size())
    # TODO: Try converting img_embeds to tokens and then decoding.
    #print(model.llama_model.model.embed_tokens.weight.data.size())
    #for ie in image_emb.squeeze():
    #    distance = torch.norm(model.llama_model.model.embed_tokens.weight.data - ie, dim=1)
    #    tokens = torch.argmin(distance)
    #    print(distance[tokens])
    #    print(tokens)
    #    out = model.llama_tokenizer.decode(tokens)
    #    print(out)
    img_list.append(image_emb)
    conv.append_message(conv.roles[0], "<Video><ImageHere></Video> " + msg)
    return conv, img_list


def get_all_video_embeddings(prompt, video_paths, model, vis_processor):
    video_embs = []
    for video_path in video_paths:
        conv = default_conversation.copy()
        conv.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."
        # video_emb = upload_video(
        #     model, video_path, vis_processor, conv, [], output_llama_embedding=True)
        embs = upload_video(
            model, video_path, vis_processor, conv, [], output_llama_embedding=True)
        # conv, img_list = upload_video(
        #     model, video_path, vis_processor, conv, [])
        # print(img_list[0].size())
        # # exit()
        # conv = ask(prompt, conv)
        # embs = get_context_emb(model, conv, img_list)
        # print(embs.size())
        video_embs.append(embs)
    return video_embs


def compute_dist(emb, other_embs):
    dists = []
    for other_emb in other_embs:
        dist = torch.norm(emb - other_emb, p=2)
        dists.append(dist)
    return dists


def compute_dist_mean(emb, other_embs):
    dists = []
    for other_emb in other_embs:
        print(other_emb.size(), emb.size())
        dist = torch.norm(emb.mean(0) - other_emb.mean(0), p=2)
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


def convert_qa_caption(caption):
    transcript = caption['transcript']
    qa = caption['qa']['qa']
    prompt = f"Answer the questions given the following transcript: {transcript}\n\n"
    prompt = prompt + 'Question: ' + qa[0]['q']
    return prompt


if __name__ == '__main__':
    # TODO: Load multiple videos and get video closest to query.
    #eval_config = 'eval_configs/video_llama_eval_only_vl.yaml'
    #eval_config = 'eval_configs/video_llama_eval_only_vl_7b.yaml'
    eval_config = 'eval_configs/video_llama_eval_only_vl_askyoutube_instruct.yaml'
    gpu_id = 0
    model_type = 'vicuna'
    args = {'cfg_path': eval_config, 'gpu_id': gpu_id,
            'model_type': model_type, 'options': None}
    args = Namespace(**args)

    model, vis_processor = init(args)
    # Run video llama model with sample video.
    # Load transcript, chunk.
    #video_path = '/mnt/g/video_caption_dataset/data/chunked_videos_30s/n4xw2fmSCrs/chunk_0.mp4'
    #transcript_path = '/mnt/g/video_caption_dataset/data/captions/n4xw2fmSCrs/qa_captions_30s_vicuna1.5_8bit_13b_instruct.json'
    #video_path = '/mnt/g/video_caption_dataset/product_reviews/automate_your_life/data/chunked_videos_30s/-0dGaEKQ5Kc/chunk_0.mp4'
    #transcript_path = '/mnt/g/video_caption_dataset/product_reviews/automate_your_life/data/captions/-0dGaEKQ5Kc/qa_captions_30s_vicuna1.5_8bit_13b_instruct.json'
    video_path = '/mnt/g/video_caption_dataset/product_reviews/automate_your_life/data/chunked_videos_30s/-OQ8quTqVdw/chunk_0.mp4'
    transcript_path = '/mnt/g/video_caption_dataset/product_reviews/automate_your_life/data/captions/-OQ8quTqVdw/qa_captions_30s_vicuna1.5_8bit_13b_instruct.json'
    captions = json.load(open(transcript_path))
    caption = captions[0]
    print(caption)
    prompt = convert_qa_caption(caption)
    print(prompt)
    # Try to extract aggregated embedding.
    output = inference(prompt, video_path, model)
    print(output)
