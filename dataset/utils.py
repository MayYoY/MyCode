import numpy as np
import pandas as pd
import cv2 as cv
from math import ceil

import torch
from torch.utils import data


def resize(frames, dynamic_det, det_length,
           w, h, larger_box, crop_face, larger_box_size, detector=None):
    """
    :param frames:
    :param dynamic_det: 是否动态检测
    :param det_length: the interval of dynamic detection
    :param w:
    :param h:
    :param larger_box: whether to enlarge the detected region.
    :param crop_face:  whether to crop the frames.
    :param larger_box_size:
    :param detector: dir of facial detector
    """
    if dynamic_det:
        det_num = ceil(frames.shape[0] / det_length)  # 检测次数
    else:
        det_num = 1
    face_region = []
    # 获取人脸区域
    for idx in range(det_num):
        if crop_face:
            assert detector is not None, "Detector is required if need to crop face!"
            face_region.append(facial_detection(frames[det_length * idx], detector,
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


def facial_detection(frame, detector, larger_box=False, larger_box_size=1.0):
    """
    检测人脸区域
    :param frame:
    :param detector:
    :param larger_box: 是否放大 bbox, 处理运动情况
    :param larger_box_size:
    """
    detector = cv.CascadeClassifier(detector)
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


def chunk(frames, gts, chunk_length, chunk_stride=-1):
    """Chunks the data into clips."""
    if chunk_stride < 0:
        chunk_stride = chunk_length
    # clip_num = (frames.shape[0] - chunk_length + chunk_stride) // chunk_stride
    frames_clips = [frames[i: i + chunk_length]
                    for i in range(0, frames.shape[0] - chunk_length + 1, chunk_stride)]
    bvps_clips = [gts[i: i + chunk_length]
                  for i in range(0, gts.shape[0] - chunk_length + 1, chunk_stride)]
    return np.array(frames_clips), np.array(bvps_clips)


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


def diff_normalize_label(label):
    """Difference frames and normalization labels"""
    diff_label = np.diff(label, axis=0)  # 差分
    normalized_label = diff_label / np.std(diff_label)
    normalized_label[np.isnan(normalized_label)] = 0
    return normalized_label


def standardize(data):
    """Difference frames and normalization data"""
    data = data - np.mean(data)
    data = data / np.std(data)
    data[np.isnan(data)] = 0
    return data


class MyDataset(data.Dataset):
    def __init__(self, config, source="VIPL-HR"):
        super(MyDataset, self).__init__()
        self.config = config
        self.source = source
        self.record = pd.read_csv(self.config.record_path)["input_files"].values.tolist()
        self.Fs = self.config.Fs  # 30

    def __len__(self):
        return len(self.record)

    def __getitem__(self, idx):
        x_path = self.record[idx]
        y_path = self.record[idx].replace("input", "label")
        x = torch.from_numpy(np.load(x_path))  # C x T x H x W
        if self.config.trans is not None:
            x = self.config.trans(x)
        y = torch.from_numpy(np.load(y_path))  # T x num (for ubfc num=3, for pure num=2)
        return x, y
