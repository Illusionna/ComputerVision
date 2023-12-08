import os
from torch import (device, cuda)
from utils.Classify_Interface import CLASSIFY_INTERFACE

os.system('cls')

classification = CLASSIFY_INTERFACE(
    log = './Register/logs/GoodWeight.pt',      # 选择训练好的权重文件路径.
    batchSize = 64,                             # 每若干张图片为一组.
    forecastSetsPath = './forecastSets',
    device = device(
        'cuda' if cuda.is_available() else 'cpu'
    )
)

classification.Classify(isShow=True)