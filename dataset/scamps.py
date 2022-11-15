import torch
import cv2 as cv
import numpy as np
import pandas as pd
import glob
import os
import mat73

from math import ceil
from torch.utils import data
from tqdm.auto import tqdm


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
            np.save(input_path_name, frames_clips[i].transpose((3, 0, 1, 2)))
            np.save(label_path_name, bvps_clips[i])
            count += 1
        return file_list

    def preprocess(self, frames, bvps):
        """
        主体部分, resize -> normalize / standardize
        :param frames: array
        :param bvps: array
        """
        frames = self.resize(frames, self.config.DYNAMIC_DETECTION,
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
                x.append(self.diff_normalize_data(f_c))
            elif data_type == "Standardize":
                x.append(self.standardized_data(f_c)[:-1, :, :, :])
            else:
                raise ValueError("Unsupported data type!")
        # 标签 transform
        x = np.concatenate(x, axis=3)
        if self.config.LABEL_TYPE == "Raw":
            bvps = bvps[:-1]
        elif self.config.LABEL_TYPE == "Difference":
            bvps = self.diff_normalize_label(bvps)
        elif self.config.LABEL_TYPE == "Standardize":
            bvps = self.standardized_label(bvps)[:-1]
        else:
            raise ValueError("Unsupported label type!")
        # 分块
        if self.config.DO_CHUNK:
            frames_clips, bvps_clips = self.chunk(x, bvps, self.config.CHUNK_LENGTH)
        else:
            frames_clips = np.array([x])
            bvps_clips = np.array([bvps])

        return frames_clips, bvps_clips

    def facial_detection(self, frame, larger_box=False, larger_box_size=1.0):
        """
        检测人脸区域
        :param frame:
        :param larger_box: 是否放大 bbox, 处理运动情况
        :param larger_box_size:
        """
        detector = cv.CascadeClassifier(self.config.detector)
        face_zone = detector.detectMultiScale(frame)
        if len(face_zone) < 1:
            print("Warning: No Face Detected!")
            result = [0, 0, frame.shape[0], frame.shape[1]]
        elif len(face_zone) >= 2:
            result = np.argmax(face_zone, axis=0)
            result = face_zone[result[2]]
            print("Warning: More than one faces are detected(Only cropping the biggest one.)")
        else:
            result = face_zone[0]
        if larger_box:
            print("Larger Bounding Box")
            result[0] = max(0, result[0] - (larger_box_size - 1.0) / 2 * result[2])
            result[1] = max(0, result[1] - (larger_box_size - 1.0) / 2 * result[3])
            result[2] = larger_box_size * result[2]
            result[3] = larger_box_size * result[3]
        return result

    def resize(self, frames, dynamic_det, det_length,
               w, h, larger_box, crop_face, larger_box_size):
        """
        :param frames:
        :param dynamic_det: 是否动态检测
        :param det_length: the interval of dynamic detection
        :param w:
        :param h:
        :param larger_box: whether to enlarge the detected region.
        :param crop_face:  whether to crop the frames.
        :param larger_box_size:
        """
        if dynamic_det:
            det_num = ceil(frames.shape[0] / det_length)  # 检测次数
        else:
            det_num = 1
        face_region = []
        # 获取人脸区域
        for idx in range(det_num):
            if crop_face:
                face_region.append(self.facial_detection(frames[det_length * idx],
                                                         larger_box, larger_box_size))
            else:  # 不截取
                face_region.append([0, 0, frames.shape[1], frames.shape[2]])
        face_region_all = np.asarray(face_region, dtype='int')
        resize_frames = np.zeros((frames.shape[0], h, w, 3))  # T x H x W x 3

        # 截取人脸并 resize
        for i in range(0, frames.shape[0]):
            frame = frames[i]
            # 选定人脸区域
            if dynamic_det:
                reference_index = i // det_length
            else:
                reference_index = 0
            if crop_face:
                face_region = face_region_all[reference_index]
                frame = frame[max(face_region[1], 0):min(face_region[1] + face_region[3], frame.shape[0]),
                              max(face_region[0], 0):min(face_region[0] + face_region[2], frame.shape[1])]
            resize_frames[i] = cv.resize(frame, (w, h), interpolation=cv.INTER_AREA)
        return resize_frames

    @staticmethod
    def chunk(frames, bvps, chunk_length, chunk_stride=-1):
        """Chunks the data into clips."""
        """
        # without stride
        clip_num = frames.shape[0] // chunk_length
        frames_clips = [frames[i * chunk_length:(i + 1) * chunk_length]
                        for i in range(clip_num)]
        bvps_clips = [bvps[i * chunk_length:(i + 1) * chunk_length]
                      for i in range(clip_num)]
        return np.array(frames_clips), np.array(bvps_clips)
        """
        # TODO: test the stride function
        if chunk_stride < 0:
            chunk_stride = chunk_length
        # clip_num = (frames.shape[0] - chunk_length + chunk_stride) // chunk_stride
        frames_clips = [frames[i: i + chunk_length]
                        for i in range(0, frames.shape[0] - chunk_length + 1, chunk_stride)]
        bvps_clips = [bvps[i: i + chunk_length]
                      for i in range(0, bvps.shape[0] - chunk_length + 1, chunk_stride)]
        return np.array(frames_clips), np.array(bvps_clips)

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

    @staticmethod
    def diff_normalize_data(data):
        """Difference frames and normalization data"""
        n, h, w, c = data.shape
        normalized_len = n - 1
        normalized_data = np.zeros((normalized_len, h, w, c), dtype=np.float32)
        for j in range(normalized_len - 1):
            normalized_data[j, :, :, :] = (data[j + 1, :, :, :] - data[j, :, :, :]) / (
                    data[j + 1, :, :, :] + data[j, :, :, :] + 1e-7)
        normalized_data = normalized_data / np.std(normalized_data)
        normalized_data[np.isnan(normalized_data)] = 0
        return normalized_data

    @staticmethod
    def diff_normalize_label(label):
        """Difference frames and normalization labels"""
        diff_label = np.diff(label, axis=0)  # 差分
        normalized_label = diff_label / np.std(diff_label)
        normalized_label[np.isnan(normalized_label)] = 0
        return normalized_label

    @staticmethod
    def standardized_data(data):
        """Difference frames and normalization data"""
        data = data - np.mean(data)
        data = data / np.std(data)
        data[np.isnan(data)] = 0
        return data

    @staticmethod
    def standardized_label(label):
        label = label - np.mean(label)
        label = label / np.std(label)
        label[np.isnan(label)] = 0
        return label


class SCAMPS(data.Dataset):
    def __init__(self, config):
        super(SCAMPS, self).__init__()
        self.config = config
        self.record = pd.read_csv(self.config.record_path)["input_files"].values.tolist()
        self.Fs = self.config.Fs

    def __len__(self):
        return len(self.record)

    def __getitem__(self, idx):
        x_path = self.record[idx]
        y_path = self.record[idx].replace("input", "label")
        x = torch.from_numpy(np.load(x_path))
        if self.config.trans is not None:
            x = self.config.trans(x)
        y = torch.from_numpy(np.load(y_path))
        return x, y
