'''
# System --> Windows & Python3.8.0
# Author --> Illusionna
'''
# -*- Encoding: UTF-8 -*-


import time
import json
from torch import (no_grad, save)
from torch.utils.data import DataLoader
from utils.Preprocess.Custom_Data import CUSTOM_DATA


class TRAIN_INTERFACE:
    def __init__(
            self,
            *args,
            trainList:list,
            validationList:list,
            composeList:list,
            parameterDictionary:dict,
            **kwargs
    ) -> None:
        self.trainList = trainList
        self.validationList = validationList
        self.composeList = composeList[:2]
        self.parameters = parameterDictionary
        start = time.time()
        TRAIN_INTERFACE.__Train(self)
        terminal = time.strftime("%H:%M:%S", time.gmtime(time.time()-start))
        print(f'<<--------------------------Total Cost: {terminal} -------------------------->>')

    def __Prepare(self) -> None:
        TRAIN_INTERFACE.__GetClassification(self)
        TRAIN_INTERFACE.__Decode(self)
        TRAIN_INTERFACE.__Digitize(self)
        trainData = CUSTOM_DATA(
            IOList = self.trainList,
            labelsList = self.trainLabels,
            transforms = self.composeList[0]
        )
        validationData = CUSTOM_DATA(
            IOList = self.validationList,
            labelsList = self.validationLabels,
            transforms = self.composeList[1]
        )
        self.trainLoader = DataLoader(
            dataset = trainData,
            batch_size = self.parameters['batchSize'],
            shuffle = True
        )
        self.validationLoader = DataLoader(
            dataset = validationData,
            batch_size = self.parameters['batchSize'],
            shuffle = True
        )

    def __Train(self) -> None:
        TRAIN_INTERFACE.__Prepare(self)
        recordDictionary = {
            'epoch': [],
            'trainLoss': [],
            'trainAccuracy': [],
            'validationLoss': [],
            'validationAccuracy': []
        }
        print('<<--------------------------------Training-------------------------------->>')
        for iteration in range(0, self.parameters['epochs'], 1):
            stamp = time.time()
            epochLoss = 0
            epochAccuracy = 0
            for (tensorX, tensorY) in self.trainLoader:
                tensorX = tensorX.to(self.parameters['device'])
                tensorY = tensorY.to(self.parameters['device'])
                output = self.parameters['model'](tensorX)
                loss = self.parameters['criterion'](output, tensorY)
                self.parameters['optimizer'].zero_grad()
                loss.backward()
                self.parameters['optimizer'].step()
                accuracy = ((output.argmax(dim=1)) == tensorY).float().mean()
                # --------------------------------------------------------------------
                # # correct = ((torch.max(output.data, 1)[1] == tensorY).sum().item())
                # # accuracy = correct / len(tensorY)
                # --------------------------------------------------------------------
                epochAccuracy = epochAccuracy + accuracy / len(self.trainLoader)
                epochLoss = epochLoss + loss / len(self.trainLoader)
            del tensorX
            del tensorY
            with no_grad():
                epochValidationAccuracy = 0
                epochValidationLoss = 0
                for (tensorX, tensorY) in self.validationLoader:
                    tensorX = tensorX.to(self.parameters['device'])
                    tensorY = tensorY.to(self.parameters['device'])
                    validationOutput = self.parameters['model'](tensorX)
                    validationLoss = self.parameters['criterion'](validationOutput, tensorY)
                    accuracy = ((validationOutput.argmax(dim=1)) == tensorY).float().mean()
                    # --------------------------------------------------------------------
                    # # correct = ((torch.max(validationOutput.data, 1)[1] == tensorY).sum().item())
                    # # accuracy = correct / len(tensorY)
                    # --------------------------------------------------------------------
                    epochValidationAccuracy = epochValidationAccuracy + accuracy / len(self.validationLoader)
                    epochValidationLoss = epochValidationLoss + validationLoss / len(self.validationLoader)
                del tensorX
                del tensorY
            print(f'Epoch: {iteration+1}/{self.parameters["epochs"]} -- Loss: {epochLoss:.5f} -- Accuracy: {epochAccuracy:.5f} -- Validation Loss: {epochValidationLoss:.5f} -- Validation Accuracy: {epochValidationAccuracy:.5f}')
            recordDictionary['epoch'].append(iteration+1)
            recordDictionary['trainLoss'].append(float(epochLoss))
            recordDictionary['trainAccuracy'].append(float(epochAccuracy))
            recordDictionary['validationLoss'].append(float(epochValidationLoss))
            recordDictionary['validationAccuracy'].append(float(epochValidationAccuracy))
            save(self.parameters['model'], f'./Register/logs/Loss{epochLoss:.5f}ValLoss{epochValidationLoss:.5f}.pt')
            now = time.strftime("%H:%M:%S", time.gmtime(time.time()-stamp))
            print(f'Cost: {now}')
        TRAIN_INTERFACE.__SaveJson(recordDictionary)
        print('<<-------------------Model is saved to "./Register/logs"------------------->>')
        print('<<---------------------Result is saved to "./Register"--------------------->>')

    def __GetClassification(self) -> None:
        self.trainLabels = []
        self.validationLabels = []
        for i in range(0, len(self.trainList), 1):
            tempString = self.trainList[i]
            subString = tempString[(tempString.find('datasets')+9):]
            label = subString[:(subString.find('/'))]
            self.trainLabels.append(label)
        for j in range(0, len(self.validationList), 1):
            tempString = self.validationList[j]
            subString = tempString[(tempString.find('datasets')+9):]
            label = subString[:(subString.find('/'))]
            self.validationLabels.append(label)
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
        for i in range(0, len(self.trainLabels), 1):
            value = self.encodingDictionary[self.trainLabels[i]]
            self.trainLabels[i] = value
        for j in range(0, len(self.validationLabels), 1):
            value = self.encodingDictionary[self.validationLabels[j]]
            self.validationLabels[j] = value

    def __SaveJson(D:dict) -> None:
        with open('./Register/trainResults.json', 'w') as f:
            f.write(
                json.dumps(
                    D,
                    ensure_ascii = False,
                    indent = 4,
                    separators = (',', ': ')
                )
            )