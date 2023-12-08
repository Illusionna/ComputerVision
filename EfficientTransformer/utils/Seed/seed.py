'''
# System --> Windows & Python3.8.0
# Author --> Illusionna
'''
# -*- Encoding: UTF-8 -*-


import os
import torch
import random
import numpy as np
import torch.backends.cudnn


class SEED:
    """
    随机种子播种类，确保程序执行的一致性.
    """
    def __init__(self, seed:int=42) -> None:
        self.seed = seed
        SEED.SeedEverything(self)

    def SeedEverything(self) -> None:
        random.seed(self.seed)
        os.environ['PYTHONHASHSEED'] = str(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True