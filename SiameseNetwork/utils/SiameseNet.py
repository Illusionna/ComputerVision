'''
# System --> Windows & Python3.8.0
# File ----> SiameseNet.py
# Author --> Illusionna
# Create --> 2024/01/12 14:56:23
'''
# -*- Encoding: UTF-8 -*-


from torch import (nn, tensor)


class SIAMESE(nn.Module):
    """
    孪生网络架构类, 公有单继承 torch.nn.Module 类.
    """
    def __init__(self) -> None:
        """
        初始化构造函数: 继承父类, 定义 CNN 层和 FC 层.
        """
        super(SIAMESE, self).__init__()
        self.cnn = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(1, 4, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(4),
            nn.ReflectionPad2d(1),
            nn.Conv2d(4, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8),
            nn.ReflectionPad2d(1),
            nn.Conv2d(8, 8, kernel_size=3),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(8)
        )
        self.fc = nn.Sequential(
            nn.Linear(8*100*100, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 500),
            nn.ReLU(inplace=True),
            nn.Linear(500, 5)
        )

    def forward_once(self, x:tensor) -> tensor:
        """
        单前向传播函数: 返回一个张量的输出结果.
        """
        output = self.cnn(x)
        output = output.view(output.size()[0], -1)
        output = self.fc(output)
        return output

    def forward(self, input1:tensor, input2:tensor) -> tuple:
        """
        前向传播函数: 重写 torch.nn.Module 的 forward() 函数, 返回两个张量传播结果, 以元组形式保存.
        """
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return (output1, output2)