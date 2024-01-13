'''
# System --> Windows & Python3.8.0
# File ----> illustrate_interface.py
# Author --> Illusionna
# Create --> 2024/01/13 22:20:42
'''
# -*- Encoding: UTF-8 -*-


import os
import json
import tkinter as tk
from typing import Literal
import matplotlib.pyplot as plt
from PIL import (Image, ImageTk)
import torchvision.transforms as transforms


class ILLUSTRATE:
    """
    插图类.
    """
    # 受保护默认字典, 其中最大并排图片数量可根据显示器适当调整.
    _defaults = {
        'MAX_FIGURE_NUMBERS': 9,
        'transform': transforms.Compose([transforms.Resize((100, 100))])
    }

    def __init__(self, jsonDataPath:str) -> None:
        """
        初始化构造函数: 定义若干私有成员属性变量.
        """
        os.system('cls')
        # 更新受保护默认字典.
        self.__dict__.update(self._defaults)
        # ------------------------------------
        self.__jsonDataPath = jsonDataPath

    def Illustrate(self, select:Literal['train', 'predict']) -> None:
        """
        初步插图函数: 提供两个字面量 'train', 'predict' 选择项.
        """
        if (select == 'train'):
            with open(file=self.__jsonDataPath, mode='r', encoding='UTF-8') as f:
                data = json.load(f)
            epoch = data['epoch']
            trainData = data['trainLoss']
            valData = data['validateLoss']
            plt.plot(epoch, trainData, label='Train Loss')
            plt.plot(epoch, valData, label='Validation Loss')
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss over Epochs')
            plt.legend()
            plt.show()
        elif (select == 'predict'):
            with open(file=self.__jsonDataPath, mode='r', encoding='UTF-8') as f:
                data = json.load(f)
            for (key, value) in data.items():
                for (subKey, subValue) in value.items():
                    ILLUSTRATE.__GUI(
                        self,
                        testImagePath = key,
                        categoryKey = subKey,
                        grapPicturesCategoryDictionary = subValue
                    )
        else:
            assert print(f'未定义 "{select}" 字面量, 仅提供 "train" 和 "predict" 字面量, 检查 select 选项是否正确!')

    def __GUI(
            self,
            testImagePath:str,
            categoryKey:str,
            grapPicturesCategoryDictionary:dict
    ) -> None:
        """
        "predict" 字面量需要的 GUI 窗口.
        """
        root = tk.Tk()
        root.title(f'不相似度视图, 数值越大越不相似, 当前对比类别为: {categoryKey}')
        testPhoto = ImageTk.PhotoImage(self.transform(Image.open(testImagePath)))
        testImageLabel = tk.Label(
            root,
            fg = 'blue',
            text = testImagePath,
            compound = 'bottom',
            image = testPhoto
        )
        testImageLabel.pack()
        frame = tk.Frame(root)
        pos = 1
        if (len(grapPicturesCategoryDictionary) <= self.MAX_FIGURE_NUMBERS):
            for (key, value) in grapPicturesCategoryDictionary.items():
                exec(f"photo{pos} = ImageTk.PhotoImage(self.transform(Image.open(key)))")
                exec(f"tk.Label(frame, fg='red', bg='yellow', text={value:.7f}, compound='bottom', image=photo{pos}).pack(side='left')")
                pos = -~pos
        else:
            for (key, value) in grapPicturesCategoryDictionary.items():
                exec(f"photo{pos} = ImageTk.PhotoImage(self.transform(Image.open(key)))")
                exec(f"tk.Label(frame, fg='red', bg='yellow', text={value:.7f}, compound='bottom', image=photo{pos}).pack(side='left')")
                pos = -~pos
                if (pos > self.MAX_FIGURE_NUMBERS):
                    break
        frame.pack()
        root.mainloop()