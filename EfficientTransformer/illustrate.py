import os
from utils.Preprocess.Illustrate import ILLUSTRATE

os.system('cls')

chart = ILLUSTRATE(jsonIO='./Register/trainResults.json')

chart.Illustrate(
    style = [2, 10, 13]     # 选择你的风格 >>> print(ILLUSTRATE.Illustrate.__doc__).
)