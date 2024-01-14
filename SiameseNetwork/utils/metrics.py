'''
# System --> Windows & Python3.8.0
# File ----> metrics.py
# Author --> Illusionna
# Create --> 2024/01/12 15:07:55
'''
# -*- Encoding: UTF-8 -*-


import torch


class Illusionna_REWRITE_CONTRASTIVE_LOSS(torch.nn.Module):
    """
    Loss 度量损失函数类, 公有单继承 torch.nn.Module 类.
    """
    def __init__(self, margin=2.0) -> None:
        """
        初始化构造函数: 继承父类, 定义 margin.
        """
        super(Illusionna_REWRITE_CONTRASTIVE_LOSS, self).__init__()
        self.margin = margin

    def forward(
            self,
            output1:torch.tensor,
            output2:torch.tensor,
            label:torch.tensor
    ) -> torch.tensor:
        """
        计算损失函数: 公式如下.
        """
        euclideanDistance = torch.nn.functional.pairwise_distance(
            x1 = output1,
            x2 = output2,
            keepdim = True
        )
        loss = torch.mean(
            (1 - label) * torch.pow(input=euclideanDistance, exponent=2)
            +
            label * torch.pow(
                input = torch.clamp(
                    input = self.margin - euclideanDistance,
                    min = 0.0
                ),
                exponent = 2
            )
        )
        return loss