'''
# System --> Windows & Python3.8.0
# File ----> train_interface.py
# Author --> Illusionna
# Create --> 2024/01/12 13:20:02
'''
# -*- Encoding: UTF-8 -*-


import os
import time
import json
import torch
import psutil
import platform
import torchvision.datasets as dset
from utils.SiameseNet import SIAMESE
from torch.utils.data import DataLoader
from utils.datasetsLoader import DATASETS
import torchvision.transforms as transforms


class TRAIN:
    """
    孪生网络训练类.
    """
    # 受保护默认字典, 其中 transform 可根据图片分辨率实际情况重新设置.
    _defaults = {
        'transform': transforms.Compose(
            [
                transforms.Resize((100, 100)),
                transforms.ToTensor()
            ]
        )
    }

    def __init__(
            self,
            *args,
            imageTrainFolder:str,
            imageValidateFolder:str,
            device:torch.device,
            model:SIAMESE,
            epoch:int,
            batchSize:int,
            criterion:torch.nn,
            optimizer:torch.optim,
            **kwargs
    ) -> None:
        """
        初始化构造函数: 定义若干私有成员属性变量.
        """
        # 更新受保护默认字典.
        self.__dict__.update(self._defaults)
        # ------------------------------------
        self.__imageTrainFolder = imageTrainFolder
        self.__imageValidateFolder = imageValidateFolder
        self.__device = device
        self.__net = model
        self.__epoch = epoch
        self.__batchSize = batchSize
        self.__criterion = criterion
        self.__optimizer = optimizer

    def _Start(self) -> None:
        """
        受保护训练函数: 使用孪生网络对图片集进行训练.
        """
        start = time.time()
        print('<<-------------------------------------训练开始------------------------------------->>')
        recordDictionary = {
            'epoch': [],
            'trainLoss': [],
            'validateLoss': []
        }
        (trainDataloader, validateDataloader) = TRAIN.__Generate(self)
        for epoch in range(0, self.__epoch, 1):
            tmp = []
            temp = []
            stamp = time.time()
            for (i, trainData) in enumerate(iterable=trainDataloader, start=0):
                (img0, img1, label) = trainData
                (img0, img1, label) = (img0.to(self.__device), img1.to(self.__device), label.to(self.__device))
                self.__optimizer.zero_grad()
                (output1, output2) = self.__net(img0, img1)
                trainLoss = self.__criterion(output1, output2, label)
                trainLoss.backward()
                self.__optimizer.step()
                tmp.append(trainLoss.item())
            trainValue = sum(tmp) / len(tmp)
            with torch.no_grad():
                for (j, validateData) in enumerate(iterable=validateDataloader, start=0):
                    (validate0, validate1, validateLabel) = validateData
                    (validate0, validate1, validateLabel) = (validate0.to(self.__device), validate1.to(self.__device), validateLabel.to(self.__device))
                    (stream1, stream2) = self.__net(validate0, validate1)
                    validateLoss = self.__criterion(stream1, stream2, validateLabel)
                    temp.append(validateLoss.item())
                validateValue = sum(temp) / len(temp)
            torch.save(
                self.__net,
                f'./cache/logs/trainLoss{trainValue:.5f}+valLoss{validateValue:.5f}.pt'
            )
            now = time.strftime("%H:%M:%S", time.gmtime(time.time()-stamp))
            print(f'Epoch: {epoch+1} -- Train loss: {trainValue:.5f} -- Validate Loss: {validateValue:.5f} -- Cost time: {now}')
            recordDictionary['epoch'].append(epoch+1)
            recordDictionary['trainLoss'].append(float(trainValue))
            recordDictionary['validateLoss'].append(float(validateValue))
        TRAIN_INTERFACE.__SaveJson(
            dictionary = recordDictionary,
            path = './cache/trainResult.json'
        )
        terminal = time.strftime("%H:%M:%S", time.gmtime(time.time()-start))
        print('<<-------------------------------------训练终止------------------------------------->>')
        print(f'Total cost time: {terminal}')
        print('权重参数结果保存至 "./cache/logs" 文件夹.')
        print('迭代结果保存至 "./cache/trainResult.json" 文件夹.\n')

    def __SaveJson(dictionary:dict, path:str) -> None:
        """
        私有训练结果保存函数: 将训练结果保存至 './cache/trainResult.json' 文件.
        """
        with open(path, 'w') as f:
            f.write(
                json.dumps(
                    dictionary,
                    ensure_ascii = False,
                    indent = 4,
                    separators = (',', ': ')
                )
            )

    def __Generate(self) -> tuple:
        """
        私有生成函数: 用迭代器生成训练集和验证集可迭代对象.
        """
        def Loader(path:str) -> DataLoader:
            """
            子函数: 返回可迭代对象首地址.
            """
            folderDataset = dset.ImageFolder(root=path)
            customDataset = DATASETS(
                imageFolderDataset = folderDataset,
                transform = self.transform,
                shouldInvert = False
            )
            dataLoader = DataLoader(
                dataset = customDataset,
                shuffle = True,
                batch_size = self.__batchSize
            )
            return dataLoader

        trainDataloader = Loader(self.__imageTrainFolder)
        validateDataloader = Loader(self.__imageValidateFolder)
        return (trainDataloader, validateDataloader)


class TRAIN_INTERFACE(TRAIN):
    """
    训练接口类, 公有单继承自定义 TRAIN 类.
    """
    def __init__(self, *args, imageTrainFolder:str, imageValidateFolder:str, device:torch.device, model:SIAMESE, epoch:int, batchSize:int, criterion:torch.nn, optimizer:torch.optim, **kwargs) -> None:
        """
        初始化构造函数: 向父类传参.
        """
        os.system('cls')
        os.system('')
        os.makedirs('./cache/logs', exist_ok=True)
        super().__init__(*args, imageTrainFolder=imageTrainFolder, imageValidateFolder=imageValidateFolder, device=device, model=model.to(device), epoch=epoch, batchSize=batchSize, criterion=criterion, optimizer=optimizer, **kwargs)
        (trainCategoryNumber, trainTotal) = TRAIN_INTERFACE.__Traverse(self, imageTrainFolder)
        (validateCategoryNumber, validateTotal) = TRAIN_INTERFACE.__Traverse(self, imageValidateFolder)
        print(f'图片集 {trainCategoryNumber+validateCategoryNumber} 类, 共计 {trainTotal+validateTotal} 张.')
        print(f'训练集 {trainCategoryNumber} 类, 含 {trainTotal} 张.')
        print(f'验证集 {validateCategoryNumber} 类, 含 {validateTotal} 张.')
        TRAIN_INTERFACE.__Detech(self)

    def Train(self) -> None:
        """
        公有训练函数: Train time!
        """
        self._Start()

    def __Traverse(self, datasetsPath:str) -> tuple:
        """
        私有遍历图片集函数: 初步获取图片集情况.
        """
        categories = os.listdir(datasetsPath)
        total = 0
        for category in categories:
            tmp = os.listdir(f'{datasetsPath}/{category}')
            total = total + len(tmp)
        return (len(categories), total)
    
    def __Detech(self) -> None:
        """
        私有设备检查函数: 获取设备中央处理器和图形处理器情况.
        """
        judge = torch.cuda.is_available()
        if judge == True:
            index = torch.cuda.current_device()
            print(end='\033[33m')
            print('CPU:', platform.machine())
            print(end='\033[33m')
            print('CPU Memory: %.3f GB' % (psutil.virtual_memory().total / (1 << 30)))
            print(end='\033[31m')
            print('Available Memory: %.3f GB' % (psutil.virtual_memory().available / (1 << 30)))
            print(end='\033[37m')
            print(end='\033[33m')
            print('GPU:', torch.cuda.get_device_name(index))
            print('Default device --> cuda:0')
            total = torch.cuda.get_device_properties('cuda').total_memory
            print('GPU Memory: %.3f GB\n' % (total / (1 << 30)))
            print(end='\033[37m')
        else:
            print(end='\033[33m')
            print('CPU:', platform.machine())
            print(end='\033[33m')
            print('CPU Memory: %.3f GB' % (psutil.virtual_memory().total / (1 << 30)))
            print(end='\033[31m')
            print('Available Memory: %.3f GB' % (psutil.virtual_memory().available / (1 << 30)))
            print(end='\033[37m\n')