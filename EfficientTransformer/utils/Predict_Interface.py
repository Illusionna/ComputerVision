'''
# System --> Windows & Python3.8.0
# File ----> Predict_Interface.py
# Author --> Illusionna
# Create --> 2023/11/23 21:35:34
'''
# -*- Encoding: UTF-8 -*-


import time
from torch import load
from torch.utils.data import DataLoader
from utils.Preprocess.Custom_Data import CUSTOM_DATA


class PREDICT_INTERFACE:
    def __init__(
            self,
            *args,
            testList:list,
            log:str,
            composeList:list,
            parameterDictionary:dict,
            **kwargs
    ) -> None:
        stamp = time.time()
        self.testList = testList
        self.composeList = composeList[-1]
        self.parameters = parameterDictionary
        self.ViT = load(log).to(self.parameters['device'])
        self.ViT.eval()
        PREDICT_INTERFACE.__Predict(self)
        now = time.strftime("%H:%M:%S", time.gmtime(time.time()-stamp))
        print(f'<<-----------------------------Cost: {now}----------------------------->>')

    def __Predict(self) -> None:
        PREDICT_INTERFACE.__Prepare(self)
        print('<<--------------------------------Predict.-------------------------------->>')
        accumulativeAccuracy = 0
        for (tensorX, tensorY) in self.testLoader:
            tensorX = tensorX.to(self.parameters['device'])
            tensorY = tensorY.to(self.parameters['device'])
            output = self.ViT(tensorX)
            accuracy = ((output.argmax(dim=1)) == tensorY).float().mean()
            accumulativeAccuracy = accumulativeAccuracy + accuracy / len(self.testLoader)
        del tensorX
        del tensorY
        print(f'Test Accuracy: {accumulativeAccuracy:.5f}')
        PREDICT_INTERFACE.__Save(float(accumulativeAccuracy))
        print('<<-----------------Test Accuracy is saved to "./Register"----------------->>')

    def __Prepare(self) -> None:
        PREDICT_INTERFACE.__GetClassification(self)
        PREDICT_INTERFACE.__Decode(self)
        PREDICT_INTERFACE.__Digitize(self)
        testData = CUSTOM_DATA(
            IOList = self.testList,
            labelsList = self.testLabels,
            transforms = self.composeList
        )
        self.testLoader = DataLoader(
            dataset = testData,
            batch_size = self.parameters['batchSize'],
            shuffle = True
        )

    def __GetClassification(self) -> None:
        self.testLabels = []
        for i in range(0, len(self.testList), 1):
            tempString = self.testList[i]
            subString = tempString[(tempString.find('datasets')+9):]
            label = subString[:(subString.find('/'))]
            self.testLabels.append(label)
        del tempString
        del subString
        del label

    def __Decode(self) -> None:
        decodeList = []
        f = open('./Register/encoding.txt', mode='r')
        while True:
            line = f.readline()
            if not line:
                break
            line = line.strip()
            decodeList.append(line)
        f.close()
        encodingList = []
        categories = []
        for n in range(0, len(decodeList), 1):
            tempString = decodeList[n]
            index = tempString.find(':')
            encodingList.append(int(tempString[(index+1):]))
            categories.append(tempString[:index])
        self.encodingDictionary = dict(zip(categories, encodingList))
        del tempString

    def __Digitize(self) -> None:
        for j in range(0, len(self.testLabels), 1):
            value = self.encodingDictionary[self.testLabels[j]]
            self.testLabels[j] = value

    def __Save(accumulativeAccuracy:float) -> None:
        f = open('./Register/testResult.txt', mode='w', encoding='UTF-8')
        f.write('Test Accuracy: ' + str(accumulativeAccuracy))
        f.close()