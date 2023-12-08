'''
# System --> Windows & Python3.8.0
# File ----> Process_Interface.py
# Author --> Illusionna
# Create --> 2023/12/08 10:36:14
'''
# -*- Encoding: UTF-8 -*-


import os
from PIL import Image
from torchvision import transforms
from sklearn.model_selection import train_test_split


class PROCESS:
    """
    预处理类.
    """
    def __init__(
            self,
            *args,
            seed:int,
            datasetsPath:str,
            **kwargs
    ) -> None:
        self.seed = seed
        self.IO = datasetsPath
        self.categories = os.listdir(datasetsPath)
        PROCESS.__Information(self)
        os.makedirs('./Register/IO', exist_ok=True)
        os.makedirs('./Register/logs', exist_ok=True)
        PROCESS.__Traverse(self, datasetsPath)

    def __Information(self) -> None:
        """
        图片基本信息初步统计.
        """
        print(end='\033[33m')
        print('图片集统计信息:\n')
        print(end='\033[37m')
        counts = []
        dpi = []
        channels = []
        for i in range(0, len(self.categories), 1):
            count = 0
            categoryIO = self.IO + os.sep + self.categories[i]
            chartList = os.listdir(categoryIO)
            for j in range(0, len(chartList), 1):
                chartIO = categoryIO + os.sep + chartList[j]
                count = -~count
                temp = Image.open(chartIO)
                dpi.append(temp.size)
                channels.append(len(temp.getbands()))
                del temp
            counts.append(count)
        print(end='\033[33m')
        print(f'共计 {sum(counts)} 张图片.\n')
        width = 0
        height = 0
        for i in range(0, len(dpi), 1):
            val = dpi[i]
            width = width + val[0]
            height = height + val[1]
        print(end='\033[37m')
        for i in range(0, len(self.categories), 1):
            print(f'{self.categories[i]}: {counts[i]} 张')
        print('')
        for i in range(0, len(self.categories), 1):
            if counts[i] <= 1:
                print(end='\033[31m')
                print(f'{self.categories[i]} 图片较少.')
                print(end='\033[37m')
        for i in range(0, len(self.categories), 1):
            if counts[i] <= 1:
                print('\n扩充图片后重新执行程序...\n')
                exit(0)
        self.averageWidth = width / len(dpi)
        self.averageHeight = height / len(dpi)
        self.block = int((self.averageWidth + self.averageHeight) / 2)
        while not (self.block % 4 == 0):
            self.block = ~-self.block
        if self.block <= 16:
            print(end='\033[31m')
            print('图片分辨率过小，使用像素较大的图片...\n')
            print(end='\033[37m')
            exit(0)
        else:
            print('|---------------------------------|')
            print('| 图片平均宽度: %.0f dpi' % self.averageWidth)
            print('| 图片平均高度: %.0f dpi' % self.averageHeight)
            print(end='\033[33m')
            print('|---------------------------------|')
            print('|\t      |-宽度: %.0f dpi' % self.block)
            print('| 建议参数预设|-高度: %.0f dpi' % self.block)
            print('|\t      |-通道:', end=' ')
            print(*list(set(channels)))
            print('|---------------------------------|')
            print('\033[37m')

    def __Traverse(self, datasetsPath:str) -> None:
        """
        遍历图片集函数.
        """
        categories = os.listdir(datasetsPath)
        PROCESS.__Encode(categories)
        self.allPathList = []
        for i in range(0, len(categories), 1):
            tempList = os.listdir(datasetsPath + '/' + categories[i])
            for j in range(0, len(tempList), 1):
                self.allPathList.append('./datasets/' + categories[i] + '/' + tempList[j])
        del tempList

    def Rename(self) -> None:
        """
        图像批量化重命名函数.
        """
        pass

    def __Encode(L:list) -> None:
        """
        图片集类型编码函数.
        """
        tempList = []
        for n in range(0, len(L), 1):
            tempList.append(f'{L[n]}:{n}')
        f = open('./Register/encoding.txt', 'w', encoding='UTF-8')
        for line in tempList:
            f.write(line + '\n')
        f.close()
        del tempList

    def Split(self, ratio:tuple=(7, 2, 1)) -> None:
        """
        训练集、测试集、验证集划分函数.
        """
        (self.trainSet, testSet) = train_test_split(
            self.allPathList,
            test_size = ((ratio[1]+ratio[2]) / (ratio[0]+ratio[1]+ratio[2])),
            shuffle = True,
            random_state = self.seed
        )
        (self.testSet, self.validationSet) = train_test_split(
            testSet,
            test_size = (ratio[2] / (ratio[1] + ratio[2])),
            shuffle = True,
            random_state = self.seed
        )
        print('##--------------------------------##')
        print(f'Train set: {len(self.trainSet)}  ({len(self.trainSet) / len(self.allPathList):.2%})')
        print(f'Test set: {len(self.testSet)}  ({len(self.testSet) / len(self.allPathList):.2%})')
        print(f'Validation set: {len(self.validationSet)}  ({len(self.validationSet) / len(self.allPathList):.2%})')
        print('<<-----------Split done----------->>')

    def Compose() -> list:
        """
        多步骤整合函数: 合并调整、剪裁、水平翻转、张量化等操作.
        """
        trainTransforms = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor()
            ]
        )
        validationTransforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ]
        )
        testTransforms = transforms.Compose(
            [
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor()
            ]
        )
        return [trainTransforms, validationTransforms, testTransforms]

    def Save(self) -> None:
        """
        划分数据集结果保存函数.
        """
        f = open('./Register/IO/trainIO.txt', 'w', encoding='UTF-8')
        for line in self.trainSet:
            f.write(line + '\n')
        f.close()
        f = open('./Register/IO/validationIO.txt', 'w', encoding='UTF-8')
        for line in self.validationSet:
            f.write(line + '\n')
        f.close()
        f = open('./Register/IO/testIO.txt', 'w', encoding='UTF-8')
        for line in self.testSet:
            f.write(line + '\n')
        f.close()
        print('<<----Saved to "./Register/IO"---->>')

    def Loader(IO:str) -> list:
        """
        逐行加载 .txt 文本函数.
        """
        IOList = []
        f = open(IO, mode='r')
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            IOList.append(line)
        f.close()
        return IOList