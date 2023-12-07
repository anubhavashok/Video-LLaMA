from video_llama.common.config import Config
from video_llama.common.registry import registry
from video_llama.processors import Blip2ImageEvalProcessor
from video_llama.processors.video_processor import ToTHWC, ToUint8, load_video
from video_llama.conversation.conversation_video import Chat, Conversation, default_conversation, SeparatorStyle, conv_llava_llama_2
from argparse import Namespace
import os
import torch
import glob
import numpy as np
from transformers import StoppingCriteria, StoppingCriteriaList
from torch.nn import functional as F
from datasets import load_dataset

device = 'cuda'
import sys
sys.path.append('/home/bhavashok/MMMU')
from eval import data_utils

def parse_sample(sample):
    id = sample['id']
    question = sample['question']
    options = eval(sample['options'])
    answer = sample['answer']
    image = sample['image_1'].convert('RGB')
    question_type = sample['question_type']
    #print(len(options))
    #print(options)
    index2ans, all_choices = data_utils.get_multi_choice_info(options)
    processed = {
        'id': id,
        'question_type': question_type,
        'answer': answer,
        'all_choices': all_choices,
        'index2ans': index2ans,
        'question': question,
        'image': image,
        # Remove image, question and add response afer LLM
    }
    return processed

def create_prompt(sample):
    sample = parse_sample(sample)
    # Process image

    # Create prompt
    index2ans = sample['index2ans']
    choice_str = '\n'.join([f'{k}: {index2ans[k]}' for k in index2ans])

    prompt = f'''
Question: {sample["question"]}\n
Choices:\n
{choice_str}\n
Respond with a single letter corresponding to the choice that answers the question correctly (e.g. A, B, C or D).\n
Answer: ''' 

    return prompt, sample

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


def upload_image(model, image, vis_processor, conv):
    msg = ""
    raw_image = image
    # PIL image
    image = vis_processor(raw_image).unsqueeze(0).unsqueeze(2).to(model.device)

    # Torch tensor
    #elif isinstance(image, torch.Tensor):
    #    image = image.unsqueeze(0)
    #    image = image.to(self.device)

    image_emb, _ = model.encode_videoQformer_visual(image)
    conv.append_message(conv.roles[0], "<Image><ImageHere></Image> "+ msg)
    return conv, [image_emb]



def eval_inference(sample, model, vis_processor, max_new_tokens=300, num_beams=1, min_length=1, top_p=0.9,
              repetition_penalty=1.0, length_penalty=1, temperature=1.0, max_length=2000):
    conv = default_conversation.copy()
    conv.system = "You are able to understand the visual content that the user provides. Follow the instructions carefully and explain your answers in detail."

    prompt, sample = create_prompt(sample)
    image = sample['image']
    #print(image)
    #print(np.array(image).shape)
    conv, img_list = upload_image(
        model, image, vis_processor, conv)
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
        do_sample=False, #True,
        min_length=min_length,
        top_p=top_p,
        repetition_penalty=repetition_penalty,
        length_penalty=length_penalty,
        temperature=temperature,
        output_scores=True,
        return_dict_in_generate=True,
    )
    #print(outputs)
    output_token = outputs.sequences#outputs[0]
    logits = outputs.scores[0]
    choice_idxs = [model.llama_tokenizer.encode(choice)[1] for choice in sample['index2ans'].keys()]
    #print(choice_idxs)
    #print(logits[:, choice_idxs])
    if logits.size(1) == 0 or len(choice_idxs) == 0:
        sample['response'] = ""
        return sample
    print(logits.size(), len(choice_idxs))
    max_choice = logits[:, choice_idxs].argmax().item()
    max_choice = list(sample['index2ans'].keys())[max_choice]
    sample['response'] = max_choice
    return sample 
    # the model might output a unknow token <unk> at the beginning. remove it
    if output_token[0] == 0:
        output_token = output_token[1:]
    # some users find that there is a start token <s> at the beginning. remove it
    if output_token[0] == 1:
        output_token = output_token[1:]
    print(output_token.size())
    return output_token
    # TODO: Get argmax based on choices.
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


def ask(text, conv):
    #print(len(conv.messages) > 0)
    #print(conv.messages[-1][0] == conv.roles[0])
    #print(('</Video>' in conv.messages[-1][1]
    #      or '</Image>' in conv.messages[-1][1]))
    if len(conv.messages) > 0 and conv.messages[-1][0] == conv.roles[0] \
            and ('</Video>' in conv.messages[-1][1] or '</Image>' in conv.messages[-1][1]):  # last message is image.
        conv.messages[-1][1] = ' '.join([conv.messages[-1][1], text])
    else:
        conv.append_message(conv.roles[0], text)
    return conv

def eval_all():
    eval_config = 'eval_configs/video_llama_eval_only_vl_askyoutube_instruct_ft_no_transcripts_clip.yaml'
    args = {'cfg_path': eval_config, 'gpu_id': 0,
            'model_type': 'vicuna', 'options': None}
    model, vis_processor = init(Namespace(**args))
    vis_processor = Blip2ImageEvalProcessor()

    categories = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']
    outputs = []
    for category in categories:
        dataset = load_dataset('MMMU/MMMU', category)
        for i in range(len(dataset['test'])):
            sample = dataset['test'][i]
            sample = eval_inference(sample, model, vis_processor)
            print(sample['response'])
            outputs.append(sample)
    json.dump(outputs, open('output.json', 'w'))

if __name__ == '__main__':
    eval_all()
