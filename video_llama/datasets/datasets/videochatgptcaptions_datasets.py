"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import os
import glob
from video_llama.datasets.datasets.base_dataset import BaseDataset
from video_llama.datasets.datasets.caption_datasets import CaptionDataset
from video_llama.conversation.conversation_video import Conversation, SeparatorStyle
from video_llama.processors import transforms_video,AlproVideoTrainProcessor
from video_llama.processors.video_processor import ToTHWC,ToUint8,load_video
from typing import Dict, Optional, Sequence
import pandas as pd
import decord
from decord import VideoReader
import random
import torch
from torch.utils.data.dataloader import default_collate
import json
import transformers
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer
import copy
import pickle

class VideoChatGPTCaptionsDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_root):
        """
        vis_root (string): Root directory of video (e.g. webvid_eval/video/)
        ann_root (string): Root directory of annotations (e.g. webvid_eval/annotations/)
        split (string): val or test
        """
        super().__init__(vis_processor=vis_processor, text_processor=text_processor)

        ts_df = []

        self.video_paths = self._load_video_paths(vis_root)
        print(self.video_paths)
        self.annotation = self._load_annotations(ann_root)
        print(len(self.annotation))
        self.vis_root = vis_root
        self.resize_size = 224
        self.num_frm = self.vis_processor.n_frms
        self.frm_sampling_strategy = 'headtail'

    def _load_video_paths(self, vis_root):
        return dict([(os.path.splitext(os.path.basename(path))[0], path) for path in glob.glob(os.path.join(vis_root, '*.mp4'))])

    def _load_annotations(self, ann_root, num_captions=10):
        data = []
        for video_id in self.video_paths:
            annotation_path = os.path.join(ann_root, f'{video_id}.txt')
            if not os.path.exists(annotation_path):
                continue
            annotation = open(annotation_path).read()
            annotation = annotation.replace('\ufeff', '')
            d = {'video_id': video_id, 'caption': annotation}
            data.append(d)
        return data

    def _check_video_chunk_exists(self, video_id, i):
        video_path = os.path.join(self.video_paths[video_id])
        return os.path.exists(video_path)

    def __getitem__(self, index):
        #index = index % 2
        num_retries = 10  # skip error videos
        for _ in range(num_retries):
            #sample = self.annotation.iloc[index]
            sample = self.annotation[index]
            sample_dict = sample#.to_dict()

            video_id = sample_dict['video_id']
            text = sample_dict['caption']

            try:
                video_path = self.video_paths[video_id]
                video = self.vis_processor(video_path)
                #print('video: ', video)
            except:
                print(f"Failed to load examples with video: {video_id}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            caption = self.text_processor(text)

            # print(video.size())
            if video is None or caption is None \
                    or video.size()!=torch.Size([3,self.vis_processor.n_frms,224,224]):
                print(f"Failed to load examples with video: {video_path}. "
                            f"Will randomly sample an example as a replacement.")
                index = random.randint(0, len(self) - 1)
                continue
            else:
                break
        else:  
            raise RuntimeError(f"Failed to fetch video after {num_retries} retries.")
        # "image_id" is kept to stay compatible with the COCO evaluation format
        return {
            "image": video,
            "text_input": caption,
            'texts': text,
            "type":'video',
        }

    def __len__(self):
        return len(self.annotation)

    # def collater(self, samples):
    #     new_result = {}
    #     new_result['image'] = default_collate( [sample["image"] for sample in samples])
    #     new_result['text_input'] = default_collate( [sample["text_input"] for sample in samples])
    #     return new_result



def convert_source_vicuna_format(sources):
    new_sources = []
    for source in sources:
        new_source = []
        for i, sentence in enumerate(source):
            role_0_msg = sentence['q']
            role_1_msg = sentence['a']
            new_source.append({
                'from':'human',
                'value': role_0_msg,
            })
            new_source.append({
                'from':'gpt',
                'value': role_1_msg,
            })
        new_sources.append(new_source)
    return new_sources

DEFAULT_IMAGE_PATCH_TOKEN = '<ImageHere>'
video_conversation = Conversation(
    system="",
    roles=("Human", "Assistant"),
    messages=[],
    offset=0,
    sep_style=SeparatorStyle.SINGLE,
    sep="###",
)
IGNORE_INDEX = -100

def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "###"
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = video_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = video_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation

def _tokenize_fn(strings: Sequence[str],
                tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=512,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    texts = tokenizer.batch_decode(input_ids)
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
        texts=texts,
    )

def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        # TODO: Add the transcript here
        #header = f"Answer the questions given the following transcript: {transcript}\n\n"
        header = f"{video_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    conversations_tokenized = _tokenize_fn(conversations, tokenizer)
    input_ids = conversations_tokenized["input_ids"]
    texts = conversations_tokenized["texts"]
    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source],
                                      tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets, texts=texts)

def preprocess_multimodal(
    conversation_list: Sequence[str],
    multimodal_cfg: dict,
    cur_token_len: int,
    msg=''
) -> Dict:
    is_multimodal = True
    image_token_len = cur_token_len
    conversation_list[0]["q"] = "<Video>"+DEFAULT_IMAGE_PATCH_TOKEN * image_token_len +"</Video> " + msg + conversation_list[0]["q"]
    return [conversation_list]

class WebvidDatasetEvalDataset(BaseDataset):
    def __init__(self, vis_processor, text_processor, vis_root, ann_paths):
        """
        vis_root (string): Root directory of images (e.g. coco/images/)
        ann_root (string): directory to store the annotation file
        split (string): val or test
        """
        super().__init__(vis_processor, text_processor, vis_root, ann_paths)

    def __getitem__(self, index):

        ann = self.annotation[index]

        vname = ann["video"]
        video_path = os.path.join(self.vis_root, vname)

        video = self.vis_processor(video_path)

        return {
            "video": video,
            "image_id": ann["image_id"],
            "instance_id": ann["instance_id"],
        }


if __name__ == '__main__':
    from video_llama.processors.base_processor import BaseProcessor
    from video_llama.processors import transforms_video, AlproVideoTrainProcessor
    vis_root = '/mnt/g/video_chatgpt/Test_Videos'
    ann_root = '/mnt/g/video_chatgpt/Test_Human_Annotated_Captions'
    vis_processor = AlproVideoTrainProcessor(
        image_size=224,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frms=16,
        )
    text_processor = BaseProcessor()
    dataset = VideoChatGPTCaptionsDataset(vis_processor, text_processor, vis_root, ann_root)
    print(dataset[0])
