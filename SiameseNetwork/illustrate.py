'''
# System --> Windows & Python3.8.0
# File ----> illustrate.py
# Author --> Illusionna
# Create --> 2024/01/13 22:20:17
'''
# -*- Encoding: UTF-8 -*-


from utils.illustrate_interface import ILLUSTRATE

instance = ILLUSTRATE(
    # 选择需要绘图的结果文件路径.
    jsonDataPath = './cache/testDissimilarityResult.json'
)

# 实列对象开始插图, 注意字面量选择与结果文件对应.
instance.Illustrate(select='predict')