'''
# System --> Windows & Python3.8.0
# File ----> predict.py
# Author --> Illusionna
# Create --> 2024/01/12 22:26:12
'''
# -*- Encoding: UTF-8 -*-

################## 填写路径时, 请使用反斜杠 "/" 号. ##################

import torch
import utils.predict_interface as P

instance = P.PREDICT_INTERFACE(
    # 选择训练好且较为优异的权重参数二进制文件.
    log = './cache/logs/trainLoss0.01798+valLoss0.20010.pt',
    # 选择需要预测的测试集所在文件夹.
    imageTestFolder = './testSets/face',
    # 随机抓取每个子类（即子文件夹）下若干张图片, 代码这么写的原因见 README.pdf 部分.
    randomGrabEachCategoryPictureNumbers = 50,
    # 选择抓取图片文件夹, 默认选择 trainSets 文件夹, 因为里面的图片多.
    randomGrabPictureFolder = './datasets/face/trainSets',
    # 选择训练需要的设备, 使用中央处理器 CPU 或者图形处理器 GPU.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu'),
)

instance.Predict()  # 实列对象开始预测.