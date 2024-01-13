'''
# System --> Windows & Python3.8.0
# File ----> datasetsLoader.py
# Author --> Illusionna
# Create --> 2024/01/12 20:26:34
'''
# -*- Encoding: UTF-8 -*-


import torch
import random
import numpy as np
from PIL import (Image, ImageOps)
import torchvision.datasets as dset
from torch.utils.data import Dataset
import torchvision.transforms as transforms


class DATASETS(Dataset):
    """
    自定义制作数据集类, 公有单继承 torch.utils.data.Dataset 类.
    """
    def __init__(
            self,
            imageFolderDataset:dset.DatasetFolder,
            transform:transforms = None,
            shouldInvert:bool = True
    ) -> None:
        """
        初始化构造函数: 其中 transform 可根据图片分辨率实际情况重新设置; 是否颠倒图片为默认项.
        """
        self.imageFolderDataset = imageFolderDataset    
        self.transform = transform
        self.shouldInvert = shouldInvert

    def __len__(self) -> int:
        """
        迭代器长度函数: 重写父类 Dataset 的 __len__() 函数.
        """
        return len(self.imageFolderDataset.imgs)
    
    def __getitem__(self, index:int) -> tuple:
        """
        获取迭代器键值对函数: 重写父类 Dataset 的 __getitem__() 函数.
        """
        img0Tuple = random.choice(self.imageFolderDataset.imgs)
        shouldGetSameClass = random.randint(0, 1)
        if shouldGetSameClass:
            while True:
                img1Tuple = random.choice(self.imageFolderDataset.imgs) 
                if (img0Tuple[1] == img1Tuple[1]):
                    break
        else:
            while True:
                img1Tuple = random.choice(self.imageFolderDataset.imgs) 
                if (img0Tuple[1] != img1Tuple[1]):
                    break
        img0 = Image.open(img0Tuple[0])
        img1 = Image.open(img1Tuple[0])
        img0 = img0.convert('L')
        img1 = img1.convert('L')
        if self.shouldInvert:
            img0 = ImageOps.invert(img0)
            img1 = ImageOps.invert(img1)
        if self.transform is not None:
            img0 = self.transform(img0)
            img1 = self.transform(img1)
        return (
            img0,
            img1,
            torch.from_numpy(
                np.array(
                    object = [int(img1Tuple[1] != img0Tuple[1])],
                    dtype = np.float32
                )
            )
        )