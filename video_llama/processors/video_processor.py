"""
 Copyright (c) 2022, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause
"""

import torch
from video_llama.common.registry import registry
from decord import VideoReader
import decord
import numpy as np
from video_llama.processors import transforms_video
from video_llama.processors.base_processor import BaseProcessor
from video_llama.processors.randaugment import VideoRandomAugment
from video_llama.processors import functional_video as F
from omegaconf import OmegaConf
from torchvision import transforms
import random as rnd


MAX_INT = registry.get("MAX_INT")
decord.bridge.set_bridge("torch")

def load_video(video_path, n_frms=MAX_INT, height=-1, width=-1, shortest_side=None, sampling="uniform", return_msg = False):
    decord.bridge.set_bridge("torch")
    if shortest_side is not None:
        vr = VideoReader(uri=video_path, width=shortest_side)
        #height, width = vr.height, vr.width
        #print(height, width)
        #if height < width:
        #    height = shortest_side
        #    width = width * (shortest_side / height)
        #else:
        #    width = shortest_side
        #height = height * (shortest_side / width)
        #vr = VideoReader(uri=video_path, height=height, width=width)
    else:
        vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen

    n_frms = min(n_frms, vlen)

    if sampling == "uniform":
        indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()
    elif sampling == "headtail":
        indices_h = sorted(rnd.sample(range(vlen // 2), n_frms // 2))
        indices_t = sorted(rnd.sample(range(vlen // 2, vlen), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(indices)
    # print(type(temp_frms))
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    # " " should be added in the start and end
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg

def load_video_timestamps(video_path, n_frms=MAX_INT, height=-1, width=-1, shortest_side=None, sampling="uniform", return_msg=False, start_timestamp=0, end_timestamp=None):
    decord.bridge.set_bridge("torch")

    # Load video using shortest side or specified height and width
    if shortest_side is not None:
        vr = VideoReader(uri=video_path, width=shortest_side)
    else:
        vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    fps = float(vr.get_avg_fps())

    # Convert timestamps to frame indices
    start_frame = int(start_timestamp * fps)
    end_frame = int(end_timestamp * fps) if end_timestamp is not None else vlen

    # Ensure valid frame range
    start_frame = max(0, start_frame)
    end_frame = min(vlen, end_frame)

    # Adjust the number of frames to sample based on start and end frames
    n_frms = min(n_frms, end_frame - start_frame)
    #print(start_frame, end_frame, n_frms, start_timestamp, end_timestamp)

    if sampling == "uniform":
        indices = np.arange(start_frame, end_frame, (end_frame - start_frame) / n_frms).astype(int).tolist()
    elif sampling == "headtail":
        mid_point = (start_frame + end_frame) // 2
        indices_h = sorted(rnd.sample(range(start_frame, mid_point), n_frms // 2))
        indices_t = sorted(rnd.sample(range(mid_point, end_frame), n_frms // 2))
        indices = indices_h + indices_t
    else:
        raise NotImplementedError

    # Retrieve frames and adjust format
    temp_frms = vr.get_batch(indices)
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    # Create message with frame timestamps
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds between timestamps {start_timestamp} and {end_timestamp}."
    return frms, msg


def load_video_all_frames(video_path, frame_sep=4, height=-1, width=-1, return_msg = False):
    decord.bridge.set_bridge("torch")
    vr = VideoReader(uri=video_path, height=height, width=width)

    vlen = len(vr)
    start, end = 0, vlen
    min_num_frames = min(vlen, 8)

    # Select first frame, then each frame has to be frame_sep seconds apart from each other.
    n_frms = max(min_num_frames, (vlen / vr.get_avg_fps()) // frame_sep)
    indices = np.arange(start, end, vlen / n_frms).astype(int).tolist()

    # get_batch -> T, H, W, C
    temp_frms = vr.get_batch(indices)
    # print(type(temp_frms))
    tensor_frms = torch.from_numpy(temp_frms) if type(temp_frms) is not torch.Tensor else temp_frms
    frms = tensor_frms.permute(3, 0, 1, 2).float()  # (C, T, H, W)

    if not return_msg:
        return frms

    fps = float(vr.get_avg_fps())
    sec = ", ".join([str(round(f / fps, 1)) for f in indices])
    # " " should be added in the start and end
    msg = f"The video contains {len(indices)} frames sampled at {sec} seconds. "
    return frms, msg


class AlproVideoBaseProcessor(BaseProcessor):
    def __init__(self, mean=None, std=None, n_frms=MAX_INT):
        if mean is None:
            mean = (0.48145466, 0.4578275, 0.40821073)
        if std is None:
            std = (0.26862954, 0.26130258, 0.27577711)

        self.normalize = transforms_video.NormalizeVideo(mean, std)

        self.n_frms = n_frms


class ToUint8(object):
    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.to(torch.uint8)

    def __repr__(self):
        return self.__class__.__name__


class ToTHWC(object):
    """
    Args:
        clip (torch.tensor, dtype=torch.uint8): Size is (C, T, H, W)
    Return:
        clip (torch.tensor, dtype=torch.float): Size is (T, H, W, C)
    """

    def __init__(self):
        pass

    def __call__(self, tensor):
        return tensor.permute(1, 2, 3, 0)

    def __repr__(self):
        return self.__class__.__name__


class ResizeVideo(object):
    def __init__(self, target_size, interpolation_mode="bilinear"):
        self.target_size = target_size
        self.interpolation_mode = interpolation_mode

    def __call__(self, clip):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: central cropping of video clip. Size is
            (C, T, crop_size, crop_size)
        """
        return F.resize(clip, self.target_size, self.interpolation_mode)

    def __repr__(self):
        return self.__class__.__name__ + "(resize_size={0})".format(self.target_size)


@registry.register_processor("alpro_video_train")
class AlproVideoTrainProcessor(AlproVideoBaseProcessor):
    def __init__(
        self,
        image_size=384,
        mean=None,
        std=None,
        min_scale=0.5,
        max_scale=1.0,
        n_frms=MAX_INT,
    ):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W)
                transforms_video.RandomResizedCropVideo(
                    image_size,
                    scale=(min_scale, max_scale),
                    interpolation_mode="bicubic",
                ),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                self.normalize,
            ]
        )

    def __call__(self, vpath, start_timestamp=None, end_timestamp=None):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        if start_timestamp is not None and end_timestamp is not None:
            clip = load_video_timestamps(
                video_path=vpath,
                n_frms=self.n_frms,
                height=self.image_size,
                width=self.image_size,
                sampling="uniform",
                start_timestamp=start_timestamp,
                end_timestamp=end_timestamp,
            )
        else:
            clip = load_video(
                video_path=vpath,
                n_frms=self.n_frms,
                height=self.image_size,
                width=self.image_size,
                sampling="headtail",
            )
        return self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        min_scale = cfg.get("min_scale", 0.5)
        max_scale = cfg.get("max_scale", 1.0)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            min_scale=min_scale,
            max_scale=max_scale,
            n_frms=n_frms,
        )


@registry.register_processor("alpro_video_eval")
class AlproVideoEvalProcessor(AlproVideoBaseProcessor):
    def __init__(self, image_size=256, mean=None, std=None, n_frms=MAX_INT):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        # Input video size is (C, T, H, W)
        self.transform = transforms.Compose(
            [
                # frames will be resized during decord loading.
                ToUint8(),  # C, T, H, W
                ToTHWC(),  # T, H, W, C
                transforms_video.ToTensorVideo(),  # C, T, H, W
                self.normalize,  # C, T, H, W
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        clip = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            height=self.image_size,
            width=self.image_size,
        )

        return self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(image_size=image_size, mean=mean, std=std, n_frms=n_frms)


@registry.register_processor("resize_centercrop_video_train")
class ResizeCenterCropVideoTrainProcessor(AlproVideoBaseProcessor):
    def __init__(
        self,
        image_size=384,
        mean=None,
        std=None,
        n_frms=MAX_INT,
    ):
        super().__init__(mean=mean, std=std, n_frms=n_frms)

        self.image_size = image_size

        self.transform = transforms.Compose(
            [
                # Video size is (C, T, H, W)
                transforms_video.ResizedCenterCropVideo(
                    (image_size, image_size)
                ),
                ToTHWC(),  # C, T, H, W -> T, H, W, C
                ToUint8(),
                transforms_video.ToTensorVideo(),  # T, H, W, C -> C, T, H, W
                self.normalize,
            ]
        )

    def __call__(self, vpath):
        """
        Args:
            clip (torch.tensor): Video clip to be cropped. Size is (C, T, H, W)
        Returns:
            torch.tensor: video clip after transforms. Size is (C, T, size, size).
        """
        clip = load_video(
            video_path=vpath,
            n_frms=self.n_frms,
            shortest_side=self.image_size,
            #sampling="headtail",
        )
        return self.transform(clip)

    @classmethod
    def from_config(cls, cfg=None):
        if cfg is None:
            cfg = OmegaConf.create()

        image_size = cfg.get("image_size", 256)

        mean = cfg.get("mean", None)
        std = cfg.get("std", None)

        n_frms = cfg.get("n_frms", MAX_INT)

        return cls(
            image_size=image_size,
            mean=mean,
            std=std,
            n_frms=n_frms,
        )


