'''
# System --> Windows & Python3.8.0
# File ----> Classify_Interface.py
# Author --> Illusionna
# Create --> 2023/11/24 18:45:00
'''
# -*- Encoding: UTF-8 -*-


import os
import torch
from PIL import Image
import torchvision.transforms as transforms


class CLASSIFY_INTERFACE:
    _defaults = {
        'transform': transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ]
        )
    }

    def __init__(
            self,
            *args,
            log:str,
            batchSize:int,
            forecastSetsPath:str,
            device:torch.device,
            **kwargs
    ) -> None:
        self.__dict__.update(self._defaults)
        self.log = log
        self.device = device
        self.batchSize = batchSize
        self.forecastIO = forecastSetsPath

    def Classify(self, isShow:bool=True) -> None:
        CLASSIFY_INTERFACE.__Prepare(self)
        with open('./Register/encoding.txt', 'r') as f:
            lineList = f.readlines()
        numbers = []
        text = []
        for s in range(0, len(lineList), 1):
            line = lineList[s]
            line = line.strip()
            idx = line.find(':')
            text.append(line[:idx])
            numbers.append(int(line[(idx+1):]))
        encode = dict(zip(numbers, text))
        if isShow == True:
            print('预测结果:')
            for i in range(0, len(self.prediction), 1):
                print(f'{self.globalIOList[i]}\t\t-->  {self.prediction[i]}  -->  {encode[self.prediction[i]]}')

    def __Prepare(self) -> None:
        (IOList, groupNumbers, remaining) = CLASSIFY_INTERFACE.__GetTestImagePath(self)
        self.imagesPathList = IOList
        ViT = torch.load(self.log)
        ViT.to(self.device)
        ViT.eval()
        superLabels = []
        if self.batchSize > len(IOList):
            tensorList = []
            for n in range(0, len(IOList), 1):
                image = Image.open(IOList[n])
                tensor = self.transform(image)
                tensorList.append(tensor)
            superTensor = torch.stack(tensorList)
            superTensor = superTensor.to(self.device)
            output = ViT(superTensor)
            indexLabel = output.argmax(dim=1)
            superLabels = indexLabel.tolist()
        elif self.batchSize <= 0:
            print("\033[031mError batch size!!!\033[0m")
            assert exit(0)
        else:
            pos = 0
            while pos < (groupNumbers * self.batchSize):
                tensorList = []
                for n in range(0, self.batchSize, 1):
                    image = Image.open(IOList[pos+n])
                    tensor = self.transform(image)
                    tensorList.append(tensor)
                superTensor = torch.stack(tensorList)
                superTensor = superTensor.to(self.device)
                pos = pos + self.batchSize
                output = ViT(superTensor)
                indexLabel = output.argmax(dim=1)
                superLabels.extend(indexLabel.tolist())
            if remaining:
                tensorList = []
                for s in range(pos, len(IOList), 1):
                    image = Image.open(IOList[s])
                    tensor = self.transform(image)
                    tensorList.append(tensor)
                superTensor = torch.stack(tensorList)
                superTensor = superTensor.to(self.device)
                output = ViT(superTensor)
                indexLabel = output.argmax(dim=1)
                superLabels.extend(indexLabel.tolist())
        self.prediction = superLabels

    def __GetTestImagePath(self) -> tuple:
        IOList = []
        tempList = os.listdir(self.forecastIO)
        for i in range(0, len(tempList), 1):
            IO = self.forecastIO + '/' + tempList[i]
            IOList.append(IO)
        del tempList
        groupNumbers = int(len(IOList) / self.batchSize)
        remaining = len(IOList) - groupNumbers * self.batchSize
        self.globalIOList = IOList
        return (IOList, groupNumbers, remaining)