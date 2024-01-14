'''
# System --> Windows & Python3.8.0
# File ----> train.py
# Author --> Illusionna
# Create --> 2024/01/12 13:18:55
'''
# -*- Encoding: UTF-8 -*-

##################################################################################
## 此 Siamese Network 单个模型参数约 150 MB, 请预先给磁盘腾出空间, 防止磁盘容纳不足. ##
##################### 注意清理缓存 cache 文件夹无效的权重文件. ######################
##################################################################################

import torch
import utils.train_interface as T
from utils.SiameseNet import SIAMESE
from utils.metrics import Illusionna_REWRITE_CONTRASTIVE_LOSS

net = SIAMESE()

instance = T.TRAIN_INTERFACE(
    # 选择训练集所在位置.
    imageTrainFolder = './datasets/face/trainSets',
    # 选择验证集所在位置.
    imageValidateFolder = './datasets/face/validateSets',
    # 选择训练需要的设备, 使用中央处理器 CPU 或者图形处理器 GPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
    # 添加 Siamese 孪生网络模型.
    model = net,
    # 训练迭代次数.
    epoch = 100,
    # 处理器每轮次处理 batchSize 张图片.
    batchSize = 32,
    # 度量函数, 这里采用我重写的损失函数, 函数原型见 "./utils/metrics.py" 文件.
    criterion = Illusionna_REWRITE_CONTRASTIVE_LOSS(),
    # 优化器选择, 默认亚当优化器, 学习率 0.0003.
    optimizer = torch.optim.Adam(
        params = net.parameters(),
        lr = 0.0003
    )
)

instance.Train()    # 实列对象开始训练.