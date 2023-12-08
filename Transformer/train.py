import os
import torch
from utils.Train_Interface import TRAIN_INTERFACE
from utils.ViT.Efficient_Attention import TRANSFORMER
from utils.Preprocess.Process_Interface import PROCESS

os.system('cls')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

ViT = TRANSFORMER(
    device = device,
    classNumbers = 5,   # "./datasets" 文件夹下有 5 个目录，所以此处类别数就是 5.
    channelNumbers = 3  # 图片通道数，一般 RGB 三通道，视情况而定.
).Initialize()

optimizer = torch.optim.Adam(
    params = ViT.parameters(),
    lr = 3e-5
)

parameters = {
    'epochs': 50,       # 迭代次数.
    'batchSize': 64,    # 每若干张图片为一组进行训练.
    'model': ViT,
    'device': device,   # CUDA 或者 CPU.
    'optimizer': optimizer,
    'scheduler': torch.optim.lr_scheduler.StepLR(
        optimizer = optimizer,
        step_size = 1,
        gamma = 0.7
    ),
    'criterion': torch.nn.CrossEntropyLoss()
}

TRAIN_INTERFACE(
    trainList = PROCESS.Loader('./Register/IO/trainIO.txt'),
    validationList = PROCESS.Loader('./Register/IO/validationIO.txt'),
    composeList = PROCESS.Compose(),
    parameterDictionary = parameters
)