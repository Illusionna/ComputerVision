'''
# System --> Windows & Python3.8.0
# File ----> predict_interface.py
# Author --> Illusionna
# Create --> 2024/01/12 22:27:03
'''
# -*- Encoding: UTF-8 -*-


import os
import time
import json
import torch
import shutil
import random
from PIL import Image
import torchvision.transforms as transforms


class PREDICT:
    """
    孪生网络预测类.
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
            log:str,
            imageTestFolder:str,
            randomGrabEachCategoryPictureNumbers:int,
            randomGrabPictureFolder:str,
            device:torch.device,
            **kwargs
    ) -> None:
        """
        初始化构造函数: 定义若干私有成员属性变量.
        """
        # 更新受保护默认字典.
        self.__dict__.update(self._defaults)
        # ------------------------------------
        self.__imageTestFolder = imageTestFolder
        self.__randomGrabEachCategoryPictureNumbers = randomGrabEachCategoryPictureNumbers
        self.__randomGrabPictureFolder = randomGrabPictureFolder
        self.__device = device
        self.__net = torch.load(log).to(device)

    def _Start(self) -> None:
        """
        受保护预测函数: 使用孪生网络对图片集进行预测.
        """
        start = time.time()
        PREDICT.__GetRandomPicture(self)
        print('<<-------------------------------------预测开始------------------------------------->>')
        testImagePathList = os.listdir(self.__imageTestFolder)
        grapPicturesList = os.listdir('./cache/GrapPictures')
        firstLevelPathList = []
        firstLevelValueList = []
        pos = 1
        for imageFile in testImagePathList:
            stamp = time.time()
            secondLevelPathList = []
            secondLevelValueList = []
            testImagePath = self.__imageTestFolder + f'/{imageFile}'
            testImage = Image.open(testImagePath).convert('L')
            testImageTensor = torch.unsqueeze(
                input = self.transform(testImage),
                dim = 0
            ).to(self.__device)
            for i in grapPicturesList:
                thirdLevelPathList = []
                thirdLevelValueList = []
                categoryPath = f'./cache/GrapPictures/{i}'
                imageList = os.listdir(categoryPath)
                for j in imageList:
                    currentImagePath = categoryPath + f'/{j}'
                    currentImage = Image.open(currentImagePath).convert('L')
                    currentImageTensor = torch.unsqueeze(
                        input = self.transform(currentImage),
                        dim = 0
                    ).to(self.__device)
                    (output1, output2) = self.__net(testImageTensor, currentImageTensor)
                    euclideanDistance = torch.nn.functional.pairwise_distance(output1, output2)
                    thirdLevelPathList.append(currentImagePath)
                    thirdLevelValueList.append(float(euclideanDistance))
                thirdLevelDictionary = dict(zip(thirdLevelPathList, thirdLevelValueList))
                secondLevelPathList.append(i)
                secondLevelValueList.append(thirdLevelDictionary)
            secondLevelDictionary = dict(zip(secondLevelPathList, secondLevelValueList))
            firstLevelPathList.append(testImagePath)
            firstLevelValueList.append(secondLevelDictionary)
            now = time.strftime("%H:%M:%S", time.gmtime(time.time()-stamp))
            print(f'第 {pos} 张图片 "{testImagePath}" ---- Cost time: {now}')
            pos = -~pos
        print('<<-------------------------------------预测结束------------------------------------->>')
        firstLevelDictionary = dict(zip(firstLevelPathList, firstLevelValueList))
        PREDICT.__SaveJson(firstLevelDictionary)
        terminal = time.strftime("%H:%M:%S", time.gmtime(time.time()-start))
        print(f'Total cost time: {terminal}\n')

    def __GetRandomPicture(self) -> None:
        """
        获取随机图片函数: 从 randomGrabPictureFolder 形参文件夹下随机抓取每个类别 randomGrabEachCategoryPictureNumbers 张图片, 拷贝保存至 "./cache/GrapPictures" 文件夹下.
        """
        os.makedirs('./cache/GrapPictures', exist_ok=True)
        shutil.rmtree('./cache/GrapPictures')
        os.makedirs('./cache/GrapPictures', exist_ok=True)
        L = os.listdir(self.__randomGrabPictureFolder)
        for i in L:
            os.makedirs(f'./cache/GrapPictures/{i}', exist_ok=True)
            tmp = os.listdir(self.__randomGrabPictureFolder + f'/{i}')
            if (self.__randomGrabEachCategoryPictureNumbers <= len(tmp)):
                copyList = random.sample(tmp, self.__randomGrabEachCategoryPictureNumbers)
            else:
                copyList = tmp
                print(f'Warning: 随机抽取图片数量超过 "{self.__randomGrabPictureFolder}/{i}" 总数, 已按照总数处理...')
            for j in copyList:
                shutil.copy(
                    src = f'{self.__randomGrabPictureFolder}/{i}/{j}',
                    dst = f'./cache/GrapPictures/{i}/{j}'
                )
        print('随机图片抓取任务完成, 结果临时寄存于 "./cache/GrapPictures" 文件夹.\n')
    
    def __SaveJson(dictionary:dict) -> None:
        with open('./cache/testDissimilarityResult.json', 'w') as f:
            f.write(
                json.dumps(
                    obj = dictionary,
                    ensure_ascii = False,
                    indent = 4,
                    separators = (',', ': ')
                )
            )
        print('\n预测结果已保存至 "./cache/testDissimilarityResult.json" 文件.')


class PREDICT_INTERFACE(PREDICT):
    """
    预测接口类, 公有单继承自定义 PREDICT 类.
    """
    def __init__(self, *args, log:str, imageTestFolder:str, randomGrabEachCategoryPictureNumbers:int, randomGrabPictureFolder:str, device:torch.device, **kwargs) -> None:
        """
        初始化构造函数: 向父类传参.
        """
        os.system('cls')
        os.system('')
        try:
            f = open(log)
            f.close()
        except FileNotFoundError:
            assert print('!!!!!!!!没有此权重参数文件!!!!!!!!')
        super().__init__(*args, log=log, imageTestFolder=imageTestFolder, randomGrabEachCategoryPictureNumbers=randomGrabEachCategoryPictureNumbers, randomGrabPictureFolder=randomGrabPictureFolder, device=device, **kwargs)

    def Predict(self) -> None:
        """
        公有预测函数: Predict time!
        """
        self._Start()