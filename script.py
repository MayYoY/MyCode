import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
import os
from torch.utils import data

from dataset import ubfc_rppg
from config import PreprocessConfig, UBFCConfig

myset = ubfc_rppg.UBFC_rPPG(UBFCConfig)
my_iter = data.DataLoader(myset, batch_size=UBFCConfig.batch_size)
for x, y in my_iter:
    print(x.shape)
    print(y.shape)
