"""
VIPL-HR dataset, containing 2,378 RGB videos of 107 subjects
captured with different head movements, lighting conditions and acquisition devices
VIPL-HR/data/
|   |-- p1/
|       |-- v1/
|           |-- source1/
|               |-- gt_HR.csv
|               |-- gt_SpO2.csv
|               |-- time.txt
|               |-- video.avi
|               |-- wave.csv
|           |...
|           |-- source4/
|               |...
|       |...
|       |-- v9/
type choice: RGB(s1-3) or NIR(s4)
use cubic spline interpolation for alignment; user can specify the fps
record file: path, fold, task(vi), source(si)
"""


import cv2 as cv
import numpy as np
import pandas as pd
import torch
import os
import glob
import re

from tqdm.auto import tqdm
from typing import Tuple
from scipy import interpolate, io
from torch.utils import data

from . import utils


class Preprocess:
    def __init__(self, output_path, config):
        self.output_path = output_path
        self.config = config
        # [p1, p2, ..., pn]
        self.dirs = glob.glob(self.config.input_path + os.sep + "data" + os.sep + "*")
        self.folds = self.get_fold()

    def get_fold(self):
        ret = {}
        files = glob.glob(self.config.input_path + os.sep + "fold" + os.sep + "*.mat")
        for f in files:
            i = int(f[-5])
            temp = io.loadmat(f)[f"fold{i}"][0]  # subject_idx of fold(i + 1)
            for idx in temp:
                ret[idx] = i
        return ret

    def read_process(self):
        """Preprocesses the raw data."""
        file_num = len(self.dirs)
        progress_bar = tqdm(list(range(file_num)))
        csv_info = {"input_files": [], "fold": [], "task": [], "source": [], "Fs": []}
        for pi in self.dirs:  # i_th subject
            tasks = glob.glob(pi + os.sep + "*")  # [v1(, v1-2), v2, ...]
            for ti in tasks:
                sources = glob.glob(ti + os.sep + "*")  # [source1, source2, ...]
                for si in sources:
                    frames, Fs = self.read_video(si)  # T x H x W x C, [0, 255]
                    waves = self.read_wave(si)  # T_w,
                    fun = interpolate.CubicSpline(range(len(waves)), waves)
                    x_new = np.linspace(0, len(waves), num=len(frames))
                    gts = fun(x_new)  # T
                    # detect -> crop -> resize -> transform -> chunk -> save
                    # n x len x H x W x C, n x len
                    frames_clips, gts_clips = self.preprocess(frames, gts)

                    p_idx = re.findall("p(\d\d?\d?)", pi)[0]  # p1, p2, ..., p10, p100, p101
                    if not re.findall("v(\d-\d)", ti):
                        t_idx = re.findall("v(\d)", ti)[0]
                    else:
                        t_idx = re.findall("v(\d-\d)", ti)[0]  # v1-2
                    s_idx = re.findall("source(\d)", si)[0]
                    single_info = {"filename": f"p{p_idx}_v{t_idx}_source{s_idx}",
                                   "fold": self.folds[int(p_idx)], "task": t_idx,
                                   "source": s_idx, "Fs": Fs}
                    # file_list, fold_list, task_list, source_list
                    temp = self.save(frames_clips, gts_clips, single_info)
                    csv_info["input_files"] += temp[0]
                    csv_info["fold"] += temp[1]
                    csv_info["task"] += temp[2]
                    csv_info["source"] += temp[3]
                    csv_info["Fs"] += temp[4]
            progress_bar.update(1)

        csv_info = pd.DataFrame(csv_info)
        csv_info.to_csv(self.config.record_path, index=False)

    def save(self, frames_clips: np.array, gts_clips: np.array,
             single_info: dict) -> Tuple[list, list, list, list, list]:
        """Saves the preprocessing data."""
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path, exist_ok=True)
        file_list = []
        for i in range(len(gts_clips)):
            input_path_name = self.output_path + os.sep + f"{single_info['filename']}_input{i}.npy"
            label_path_name = self.output_path + os.sep + f"{single_info['filename']}_label{i}.npy"
            file_list.append(self.output_path + os.sep + f"{single_info['filename']}_input{i}.npy")
            # T x H x W x C -> C x T x H x W
            np.save(input_path_name, frames_clips[i].transpose((3, 0, 1, 2)))
            np.save(label_path_name, gts_clips[i])
        fold_list = [single_info["fold"]] * len(gts_clips)
        task_list = [single_info["task"]] * len(gts_clips)
        source_list = [single_info["source"]] * len(gts_clips)
        Fs_list = [single_info["Fs"]] * len(gts_clips)
        return file_list, fold_list, task_list, source_list, Fs_list

    def preprocess(self, frames, gts):
        """
        主体部分, resize -> normalize / standardize
        :param frames: array, T x H x W x C
        :param gts: array, T,
        """
        frames = utils.resize(frames, self.config.DYNAMIC_DETECTION,
                              self.config.DYNAMIC_DETECTION_FREQUENCY,
                              self.config.W, self.config.H,
                              self.config.LARGE_FACE_BOX,
                              self.config.CROP_FACE,
                              self.config.LARGE_BOX_COEF,
                              self.config.detector)
        # 视频 transform, 丢弃最后一帧
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
        # 标签 transform, 丢弃最后一帧
        x = np.concatenate(x, axis=3)
        y = np.zeros((gts.shape[0] - 1), dtype=np.float64)
        if self.config.LABEL_TYPE == "Raw":
            y[:] = gts[:-1]
        elif self.config.LABEL_TYPE == "Difference":
            y[:] = utils.diff_normalize_label(gts[:])
        elif self.config.LABEL_TYPE == "Standardize":
            y[:] = utils.standardize(gts[:])[:-1]
        else:
            raise ValueError("Unsupported label type!")
        # 分块
        if self.config.DO_CHUNK:
            frames_clips, gts_clips = utils.chunk(x, y, self.config.CHUNK_LENGTH)
        else:
            frames_clips = np.array([x])  # n x len x H x W x C
            gts_clips = np.array([y])  # n x len

        return frames_clips, gts_clips

    @staticmethod
    def read_video(data_path):
        """读取视频 T x H x W x C, C = 3"""
        vid = cv.VideoCapture(data_path + os.sep + "video.avi")
        vid.set(cv.CAP_PROP_POS_MSEC, 0)  # 设置从 0 开始读取
        ret, frame = vid.read()
        frames = list()
        while ret:
            frame = cv.cvtColor(np.array(frame), cv.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0
            frames.append(frame)
            ret, frame = vid.read()

        frames = np.asarray(frames)
        # Warning: 不同视频的实际帧率与 readme 有偏差, 需要额外记录
        if data_path[-1] == "2":
            Fs = 30
        else:
            Fs = len(frames) * 1000 / np.loadtxt(data_path + os.sep + "time.txt")[-1]
        return frames, Fs

    @staticmethod
    def read_wave(data_path):
        """
        读取 bvp 信号
        :param data_path:
        :return np.array T, ; bvp
        """
        waves = pd.read_csv(data_path + os.sep + "wave.csv")["Wave"].values
        return waves


class FramePreprocess:
    """
    以 .png 格式保存截取的人脸区域, 记录 clip range 节省存储空间
    """
    def __init__(self, config):
        self.config = config
        # [p1, p2, ..., pn]
        self.dirs = glob.glob(self.config.input_path + os.sep + "data" + os.sep + "*")
        self.folds = self.get_fold()

    def get_fold(self):
        ret = {}
        files = glob.glob(self.config.input_path + os.sep + "fold" + os.sep + "*.mat")
        for f in files:
            i = int(f[-5])
            temp = io.loadmat(f)[f"fold{i}"][0]  # subject_idx of fold(i + 1)
            for idx in temp:
                ret[idx] = i
        return ret

    def read_process(self):
        file_num = len(self.dirs)
        progress_bar = tqdm(list(range(file_num)))
        csv_info = {"input_files": [], "wave_files": [], "start": [], "end": [],
                    "fold": [], "average_HR": [], "Fs": []}
        for pi in self.dirs:  # i_th subject
            p_idx = re.findall("p(\d\d?\d?)", pi)[0]  # p1, p2, ..., p10, p100, p101
            tasks = glob.glob(pi + os.sep + "*")  # [v1(, v1-2), v2, ...]
            for ti in tasks:
                if not re.findall("v(\d-\d)", ti):
                    t_idx = re.findall("v(\d)", ti)[0]
                else:
                    t_idx = re.findall("v(\d-\d)", ti)[0]  # v1-2
                sources = glob.glob(ti + os.sep + "*")  # [source1, source2, ...]
                for si in sources:
                    # 不需要 NIR
                    if si[-1] == '4':
                        continue
                    # 插值以对齐
                    clip_range, Fs = self.read_video(si)  # T,
                    waves = self.read_wave(si)  # T_w,
                    fun = interpolate.CubicSpline(range(len(waves)), waves)
                    x_new = np.linspace(0, len(waves), num=len(clip_range))
                    gts = fun(x_new)  # T
                    # chunk, n x len, n x len
                    frames_clips, gts_clips = self.preprocess(clip_range, gts)
                    # 命名信息
                    s_idx = re.findall("source(\d)", si)[0]
                    single_info = {"filename": f"p{p_idx}_v{t_idx}_source{s_idx}",
                                   "fold": self.folds[int(p_idx)], "Fs": Fs}
                    # 平均 HR
                    gt_HR = pd.read_csv(si + os.sep + "gt_HR.csv")["HR"].values
                    # input_list, wave_list, start_list, end_list, fold_list, HR_list, Fs_list
                    temp = self.save(frames_clips, gts_clips, gt_HR, single_info, si)
                    csv_info["input_files"] += temp[0]
                    csv_info["wave_files"] += temp[1]
                    csv_info["start"] += temp[2]
                    csv_info["end"] += temp[3]
                    csv_info["fold"] += temp[4]
                    csv_info["average_HR"] += temp[5]
                    csv_info["Fs"] += temp[6]
            progress_bar.update(1)

        csv_info = pd.DataFrame(csv_info)
        csv_info.to_csv(self.config.record_path, index=False)

    def save(self, frames_clips: np.array, gts_clips: np.array,
             gt_HR: np.ndarray, single_info: dict, data_path: str):
        """Saves the preprocessing data."""
        # 生成对应的文件夹
        output_path = data_path + os.sep + "gt" + os.sep + str(self.config.CHUNK_LENGTH)
        os.makedirs(output_path, exist_ok=True)
        wave_list = []  # 标签路径
        start_list = []  # clip 范围
        end_list = []
        HR_list = []  # 平均 HR
        step = len(gt_HR) // len(gts_clips)
        # 保存文件
        for i in range(len(gts_clips)):
            HR_list.append(gt_HR[i * step: (i + 1) * step].mean())  # clip 的平均 HR
            # 保存处理好的 wave clip
            label_path = output_path + os.sep + f"{single_info['filename']}_label{i}.npy"
            np.save(label_path, gts_clips[i])
            wave_list.append(label_path)
            # 记录 clip 的范围
            start_list.append(frames_clips[i, 0])
            end_list.append(frames_clips[i, -1] + 1)
        input_list = [data_path + os.sep + "img"] * len(gts_clips)
        fold_list = [single_info["fold"]] * len(gts_clips)
        Fs_list = [single_info["Fs"]] * len(gts_clips)
        return input_list, wave_list, start_list, end_list, fold_list, HR_list, Fs_list

    def preprocess(self, clip_range, gts):
        """
        normalize / standardize
        :param clip_range: array, T,
        :param gts: array, T,
        """
        # 标签需要 standardize
        y = utils.standardize(gts[:])
        # 分块
        if self.config.DO_CHUNK:
            frames_clips, gts_clips = utils.chunk(clip_range, y, self.config.CHUNK_LENGTH)
        else:
            frames_clips = np.array([clip_range])  # n x len x H x W x C
            gts_clips = np.array([y])  # n x len

        return frames_clips, gts_clips

    def read_video(self, data_path):
        """读取视频, 人脸检测, 保存帧; 返回帧下标, 帧率"""
        vid = cv.VideoCapture(data_path + os.sep + "video.avi")
        vid.set(cv.CAP_PROP_POS_MSEC, 0)  # 设置从 0 开始读取
        ret, frame = vid.read()
        frames = list()
        while ret:
            frame = cv.cvtColor(np.array(frame), cv.COLOR_BGR2RGB)
            frame = np.asarray(frame)
            frame[np.isnan(frame)] = 0
            frames.append(frame)
            ret, frame = vid.read()

        frames = np.asarray(frames)
        # 人脸检测并截取
        frames = utils.resize(frames, self.config.DYNAMIC_DETECTION,
                              self.config.DYNAMIC_DETECTION_FREQUENCY,
                              self.config.W, self.config.H,
                              self.config.LARGE_FACE_BOX,
                              self.config.CROP_FACE,
                              self.config.LARGE_BOX_COEF).astype(np.uint8)
        # 保存处理好的帧
        save_dir = data_path + os.sep + "img"
        os.makedirs(save_dir, exist_ok=True)
        for i, frame in enumerate(frames):
            frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
            cv.imwrite(save_dir + os.sep + f"{i}.png", frame)
        # for return
        clips_range = np.arange(len(frames))
        if data_path[-1] == "2":
            Fs = 30
        else:
            Fs = len(frames) * 1000 / np.loadtxt(data_path + os.sep + "time.txt")[-1]
        return clips_range, Fs

    @staticmethod
    def read_wave(data_path):
        """
        读取 bvp 信号
        :param data_path:
        :return np.array T, ; bvp
        """
        waves = pd.read_csv(data_path + os.sep + "wave.csv")["Wave"].values
        return waves


class VIPL_HR(utils.MyDataset):
    """
    VIPL-HR 包含多种场景, 对 record 的处理比较特殊
    """
    def __init__(self, config):
        super(VIPL_HR, self).__init__(config=config, source="VIPL-HR")
        tasks = self.record["task"].values.tolist()
        sources = self.record["source"].values.tolist()
        folds = self.record["fold"].values.tolist()
        self.Fs = self.record["Fs"].values.tolist()
        # 选定指定视频
        if config.type == "RGB":
            m1 = [s != 4 for s in sources]
        else:
            m1 = [s == 4 for s in sources]  # NIR
        if config.task is not None:
            m2 = [int(t[0]) in config.task for t in tasks]
        else:
            m2 = [True] * len(tasks)
        if config.fold is not None:
            m3 = [f in config.fold for f in folds]
        else:
            m3 = [True] * len(folds)
        temp = []
        for i, file in enumerate(self.input_files):
            if m1[i] and m2[i] and m3[i]:
                temp.append(file)
        self.input_files = temp

    def __getitem__(self, idx):
        x_path = self.input_files[idx]
        y_path = self.input_files[idx].replace("input", "label")
        x = torch.from_numpy(np.load(x_path))  # C x T x H x W
        y = torch.from_numpy(np.load(y_path))  # T,
        Fs = torch.from_numpy(self.Fs[idx])
        if self.config.trans is not None:
            x = self.config.trans(x)
        return x, y, Fs


class FrameVIPL(data.Dataset):
    def __init__(self, config):
        super(FrameVIPL, self).__init__()
        record = pd.read_csv(config.record)
        self.config = config
        self.input_files = []
        self.wave_files = []
        self.starts = []
        self.ends = []
        self.average_hrs = []
        self.Fs = []
        for i in range(len(record)):
            if record.loc[i, "fold"] in config.folds:
                self.input_files.append(record.loc[i, "input_files"])
                self.wave_files.append(record.loc[i, "wave_files"])
                self.starts.append(record.loc[i, "start"])
                self.ends.append(record.loc[i, "end"])
                self.average_hrs.append(record.loc[i, "average_HR"])
                self.Fs.append(record.loc[i, "Fs"])

    def __len__(self):
        return len(self.input_files)

    def __getitem__(self, idx):
        # 读取对应帧
        x_path = self.input_files[idx]
        x = []
        for i in range(self.starts[idx], self.ends[idx]):
            temp = cv.imread(x_path + os.sep + f"{i}.png")
            temp = cv.cvtColor(temp, cv.COLOR_BGR2RGB)
            x.append(temp.transpose(2, 0, 1))
        x = utils.normalize_frame(np.asarray(x))  # 归一化
        x = torch.from_numpy(x)
        # 读取标签信息
        y_path = self.wave_files[idx]
        y = torch.from_numpy(np.load(y_path))  # T,
        average_hr = torch.tensor([self.average_hrs[idx]])
        # 帧率
        Fs = torch.tensor([self.Fs[idx]])
        if self.config.trans is not None:
            x = self.config.trans(x)
        return x, y, average_hr, Fs
