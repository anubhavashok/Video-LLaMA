import os
import logging
import warnings

from video_llama.common.registry import registry
from video_llama.datasets.builders.base_dataset_builder import BaseDatasetBuilder
from video_llama.datasets.datasets.webvid_datasets import WebvidDataset
from video_llama.datasets.datasets.askyoutube_datasets import AskYoutubeDataset
from video_llama.datasets.datasets.kinetics_datasets import KineticsDataset
from video_llama.datasets.datasets.msrvtt_datasets import MSRVTTDataset
from video_llama.datasets.datasets.msvd_datasets import MSVDDataset
from video_llama.datasets.datasets.valley_datasets import ValleyDataset
from video_llama.datasets.datasets.finevideo_datasets import FineVideoDataset, FineVideoActivityDataset
from video_llama.datasets.datasets.videochatgptcaptions_datasets import VideoChatGPTCaptionsDataset

@registry.register_builder("webvid")
class WebvidBuilder(BaseDatasetBuilder):
    train_dataset_cls = WebvidDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/webvid/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets


@registry.register_builder("askyoutube")
class AskYoutubeBuilder(BaseDatasetBuilder):
    train_dataset_cls = AskYoutubeDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/askyoutube/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets

@registry.register_builder("kinetics")
class KineticsBuilder(BaseDatasetBuilder):
    train_dataset_cls = KineticsDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/kinetics/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets

@registry.register_builder("msrvtt")
class MSRVTTBuilder(BaseDatasetBuilder):
    train_dataset_cls = MSRVTTDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/msrvtt/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets

@registry.register_builder("valley")
class ValleyBuilder(BaseDatasetBuilder):
    train_dataset_cls = ValleyDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/valley/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets

@registry.register_builder("msvd")
class MSVDBuilder(BaseDatasetBuilder):
    train_dataset_cls = MSVDDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/msvd/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets

@registry.register_builder("videochatgptcaptions")
class VideoChatGPTCaptionsBuilder(BaseDatasetBuilder):
    train_dataset_cls = VideoChatGPTCaptionsDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/msvd/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets


@registry.register_builder("finevideo")
class FineVideoBuilder(BaseDatasetBuilder):
    train_dataset_cls = FineVideoDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/finevideo/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets

@registry.register_builder("finevideo_activity")
class FineVideoActivityBuilder(BaseDatasetBuilder):
    train_dataset_cls = FineVideoActivityDataset
    DATASET_CONFIG_DICT = {"default": "configs/datasets/finevideo_activity/defaults.yaml"}
    
    def _download_ann(self):
        pass

    def _download_vis(self):
        pass

    def build(self):
        self.build_processors()
        datasets = dict()
        split = "train"

        build_info = self.config.build_info
        print("build_info: ", build_info)
        dataset_cls = self.train_dataset_cls
        datasets[split] = dataset_cls(
            vis_processor=self.vis_processors[split],
            text_processor=self.text_processors[split],
            vis_root=build_info.videos_dir,
            ann_root=build_info.anno_dir
        )

        return datasets
