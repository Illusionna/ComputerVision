'''
# System --> Windows & Python3.8.0
# File ----> Illustrate.py
# Author --> Illusionna
# Create --> 2023/11/23 16:46:50
'''
# -*- Encoding: UTF-8 -*-


import os
import json
import scienceplots
import matplotlib.pyplot as plt


class ILLUSTRATE:
    """
    绘图类.
    """
    # # ----------------------------------------------------------------
    # 参考链接: https://zhuanlan.zhihu.com/p/501759830
    # 按照方法: pip install git+https://github.com/garrettj403/SciencePlots.git
    # 默认受保护字典: 绘图风格选项.
    _defaults = {
        'style': {
            0: 'science',
            1: 'notebook',
            2: 'ieee',
            3: 'nature',
            4: 'std-colors',
            5: 'scatter',
            6: 'high-vis',
            7: 'dark_background',
            8: 'notebook',
            9: 'bright',
            10: 'vibrant',
            11: 'muted',
            12: 'retro',
            13: 'grid',
            14: 'high-contrast',
            15: 'light',
            16: 'no-latex',
            17: 'russian-font',
            18: 'cjk-tc-font',
            19: 'cjk-sc-font',
            20: 'cjk-jp-font',
            21: 'cjk-kr-font'
        }
    }
    # # ----------------------------------------------------------------

    def __init__(
            self,
            *args,
            jsonIO:str,
            **kwargs
    ) -> None:
        self.jsonIO = jsonIO
        self.__dict__.update(self._defaults)
        ILLUSTRATE.__Prepare(self)

    def Illustrate(self, style:list) -> None:
        """
        Select your style from the protected default dictionary.
            0: 'science',\n
            1: 'notebook',\n
            2: 'ieee',\n
            3: 'nature',\n
            4: 'std-colors',\n
            5: 'scatter',\n
            6: 'high-vis',\n
            7: 'dark_background',\n
            8: 'notebook',\n
            9: 'bright',\n
            10: 'vibrant',\n
            11: 'muted',\n
            12: 'retro',\n
            13: 'grid',\n
            14: 'high-contrast',\n
            15: 'light',\n
            16: 'no-latex',\n
            17: 'russian-font',\n
            18: 'cjk-tc-font',\n
            19: 'cjk-sc-font',\n
            20: 'cjk-jp-font',\n
            21: 'cjk-kr-font'
        """
        styleList = []
        for i in range(0, len(style), 1):
            styleList.append(self.style[style[i]])
        ILLUSTRATE.__Illustrate(self, styleList)

    def __Illustrate(self, styleList:list) -> None:
        """
        私有绘图函数.
        """
        axisLabels = dict(
            xlabel = 'Epoch',
            ylabel = 'Value'
        )
        characteristicList = [
            'Train Accuracy',
            'Train Loss',
            'Validation Accuracy',
            'Validation Loss'
        ]
        matrix = []
        matrix.append(self.trainAccuracy)
        matrix.append(self.trainLoss)
        matrix.append(self.validationAccuracy)
        matrix.append(self.validationLoss)
        with plt.style.context(styleList):
            (figure, axis) = plt.subplots()
            pos = 0
            for attribute in characteristicList:
                axis.plot(self.epoch, matrix[pos], label=attribute)
                pos = -~pos
            # axis.legend(fontsize=6, loc='lower right')
            axis.legend(fontsize=6, loc='upper right')
            axis.autoscale(tight=False)
            axis.set(**axisLabels)
            axis.set_title('Train result: Accuracy & Loss', fontsize=10)
            # # ------------------------------------------------------------
            # figure.savefig('./Register/results/trainResult.png', dpi=300)
            # # ------------------------------------------------------------
            figure.savefig('./Register/results/trainResult.pdf', bbox_inches='tight')
            print('<<--------------Illustration is saved to "./Register/results"-------------->>')
            del pos

    def __Prepare(self) -> None:
        """
        预备函数.
        """
        os.makedirs('./Register/results', exist_ok=True)
        f = open(self.jsonIO, mode='r')
        content = f.read()
        dataDictionary = json.loads(content)
        f.close()
        self.epoch = dataDictionary['epoch']
        self.trainLoss = dataDictionary['trainLoss']
        self.trainAccuracy = dataDictionary['trainAccuracy']
        self.validationLoss = dataDictionary['validationLoss']
        self.validationAccuracy = dataDictionary['validationAccuracy']
        del dataDictionary