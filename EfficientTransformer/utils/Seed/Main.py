'''
# System --> Windows & Python3.8.0
# File ----> Main.py
# Author --> Illusionna
# Create --> 2023/11/27 07:58:14
'''
# -*- Encoding: UTF-8 -*-


import os
from abc import (abstractmethod, ABCMeta)


class MAIN(metaclass=ABCMeta):
    """
    提前准备:
        Ⅰ. 图片集下载: 链接: https://pan.baidu.com/s/1xyMbXGbNYu4nIZRyP50PUg?pwd=emx4
                       提取码: emx4
        Ⅱ.
            1. 下载好的图片集文件夹 Attachment2.zip 里的内容顶替 "./datasets" 里的内容
            2. 下载好的 Attachment3 文件夹内容顶替 "./forecastSets" 文件夹内容
            3. Illusionna 已经训练好的 GoodWeight.pt 权重文件可以直接用于 classify.py 分类任务
            4. "./Register/encoding.txt" 由 process.py 生成，训练自己的图片集直接删除整个 "./Register" 文件夹即可，不必手动配置

    环境配置:
        Ⅰ. 操作系统: 建议 Windows10 + Python 3.8
        Ⅱ. 第三方库:
            1. torch、torchvision、torchaudio
            2. linformer、einops
            3. scikit-learn
            4. matplotlib、numpy、pillow、SciencePlots
            5. 以及我也不确定还需要什么库...
        Ⅲ. Conda 支持:
            conda create -n CV python==3.8.0
            conda activate CV
            pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
            pip install -i https://pypi.tuna.tsinghua.edu.cn/simple linformer
            pip install -i https://pypi.tuna.tsinghua.edu.cn/simple einops
            pip install -i https://pypi.tuna.tsinghua.edu.cn/simple scikit-learn
            pip install -i https://pypi.tuna.tsinghua.edu.cn/simple matplotlib
            pip install -i https://pypi.tuna.tsinghua.edu.cn/simple pillow
            [pip install git+https://github.com/garrettj403/SciencePlots.git 插图 illustrate.py 额外调用包，视情况安装]

    执行程序:
        Step 1: 执行 process.py 处理文件，自动生成 "./Register" 文件夹
        Step 2: 修改好 train.py 参数后，执行训练文件，生成权重文件保存至 "./Register/logs" 文件夹
        (Step 3): [可执行 illustrate.py 插图文件绘图]
        (Step 4): [修改好 predict.py 参数后，可执行预测文件，用于查看模型效果如何]
        Step 5: 调整好参数后，执行 classify.py 分类文件，若模型效果良好，则可以应用

    代码仓库: https://github.com/Illusionna/Transformer

    注意事项: 阅读 "./README.md" 文件
    """
    os.system('cls')
    @abstractmethod
    def __init__(self) -> None:
        pass