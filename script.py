import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from torch.utils import data

from dataset import utils, pure
from config import PreprocessConfig, LoadConfig

test_set = utils.MyDataset(LoadConfig, source="PURE")
test_iter = data.DataLoader(test_set, batch_size=LoadConfig.batch_size, shuffle=True)
for x, y in test_iter:
    print(x.shape)
    print(y.shape)
"""ops = pure.Preprocess(PreprocessConfig.output_path, PreprocessConfig)
ops.read_process()"""
