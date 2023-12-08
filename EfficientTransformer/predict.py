import os
import torch
from utils.Predict_Interface import PREDICT_INTERFACE
from utils.Preprocess.Process_Interface import PROCESS

os.system('cls')

parameters = {
    'batchSize': 32,    # 每若干张图片为一组.
    'device': torch.device('cuda' if torch.cuda.is_available() else 'cpu')
}

PREDICT_INTERFACE(
    testList = PROCESS.Loader('./Register/IO/testIO.txt'),
    log = './Register/logs/GoodWeight.pt',      # 选择训练好的权重文件路径.
    composeList = PROCESS.Compose(),
    parameterDictionary = parameters
)