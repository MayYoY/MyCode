import numpy as np
import pandas as pd
import glob
import os
import mat73
from tqdm.auto import tqdm

from . import utils


class Preprocess:
    """
    实现预处理, 包括: 人脸检测, 标准化 (or 归一化), ...
    最终视频与对应 bvp 信号保存为 .npy
    """

    def __init__(self, output_path, config):
        self.length = 0
        self.output_path = output_path
        self.config = config
        self.dirs = self.get_data(self.config.input_path)

    @staticmethod
    def get_data(input_path):
        """读取目录下文件名"""
        data_dirs = glob.glob(input_path + os.sep + "*.mat")
        if not data_dirs:
            raise ValueError("Path doesn't exist!")
        dirs = list()
        for data_dir in data_dirs:
            subject = os.path.split(data_dir)[-1]
            # index 样本编号; path 路径
            dirs.append({"index": subject, "path": data_dir})
        return dirs

    def read_process(self):
        """Preprocesses the raw data."""
        file_num = len(self.dirs)
        progress_bar = tqdm(list(range(file_num)))
        file_list = []
        for i in progress_bar:
            # read file
            data_path = self.dirs[i]['path']
            progress_bar.set_description(f"Processing {data_path}")
            frames = self.read_video(data_path)  # T x H x W x C
            # 转换为人脸检测所需格式
            frames = (np.round(frames * 255)).astype(np.uint8)
            bvps = self.read_wave(data_path)  # T,
            # detect -> crop -> resize -> transform -> chunk -> save
            frames_clips, bvps_clips = self.preprocess(frames, bvps)
            file_list += self.save(frames_clips, bvps_clips, self.dirs[i]['index'])
        file_list = pd.DataFrame(file_list, columns=['input_files'])
        file_list.to_csv(self.config.record_path, index=False)

    def save(self, frames_clips: np.array, bvps_clips: np.array, filename) -> list:
        """Saves the preprocessing data."""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        count = 0
        file_list = []
        for i in range(len(bvps_clips)):
            input_path_name = self.output_path + os.sep + f"{filename}_input{count}.npy"
            label_path_name = self.output_path + os.sep + f"{filename}_label{count}.npy"
            file_list.append(self.output_path + os.sep + f"{filename}_input{count}.npy")
            # T x H x W x C -> C x T x H x W
            np.save(input_path_name, frames_clips[i].transpose((3, 0, 1, 2)).astype(np.float32))
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return file_list

    def preprocess(self, frames, bvps):
        """
        主体部分, resize -> normalize / standardize
        :param frames: array
        :param bvps: array
        """
        frames = utils.resize(frames, self.config.DYNAMIC_DETECTION,
                              self.config.DYNAMIC_DETECTION_FREQUENCY,
                              self.config.W, self.config.H,
                              self.config.LARGE_FACE_BOX,
                              self.config.CROP_FACE,
                              self.config.LARGE_BOX_COEF)
        # 视频 transform
        x = list()
        for data_type in self.config.DATA_TYPE:
            f_c = frames.copy()
            if data_type == "Raw":
                x.append(f_c[:-1, :, :, :])
            elif data_type == "Difference":
                x.append(utils.diff_normalize_data(f_c))
            elif data_type == "Standardize":
                x.append(utils.standardize(f_c)[:-1, :, :, :])
            else:
                raise ValueError("Unsupported data type!")
        # 标签 transform
        x = np.concatenate(x, axis=3)
        if self.config.LABEL_TYPE == "Raw":
            bvps = bvps[:-1]
        elif self.config.LABEL_TYPE == "Difference":
            bvps = utils.diff_normalize_label(bvps)
        elif self.config.LABEL_TYPE == "Standardize":
            bvps = utils.standardize(bvps)[:-1]
        else:
            raise ValueError("Unsupported label type!")
        # 分块
        if self.config.DO_CHUNK:
            frames_clips, bvps_clips = utils.chunk(x, bvps, self.config.CHUNK_LENGTH)
        else:
            frames_clips = np.array([x])
            bvps_clips = np.array([bvps])

        return frames_clips, bvps_clips

    @staticmethod
    def read_video(data_path):
        """读取视频 T x H x W x C, C = 3"""
        mat = mat73.loadmat(data_path)
        frames = mat['Xsub']  # load raw frames
        return np.asarray(frames)

    @staticmethod
    def read_wave(data_path):
        """读取 ppg 信号"""
        mat = mat73.loadmat(data_path)
        ppg = mat['d_ppg']  # load raw frames
        return np.asarray(ppg)
