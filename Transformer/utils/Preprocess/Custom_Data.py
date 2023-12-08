'''
# System --> Windows & Python3.8.0
# Author --> Illusionna
'''
# -*- Encoding: UTF-8 -*-


from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset


class CUSTOM_DATA(Dataset):
    """
    自定义数据集类.
    """
    def __init__(
            self,
            IOList:list,
            labelsList:list,
            transforms:transforms.Compose
    ) -> None:
        self.IOList = IOList
        self.labelsList = labelsList
        self.transforms = transforms

    def __getitem__(self, idx:int) -> tuple:
        imagePath = self.IOList[idx]
        image = Image.open(imagePath)
        transformImage = self.transforms(image)
        # ------------------------------------------
        # chart = transforms.ToPILImage()
        # picture = chart(transformImage)
        # picture.show()
        # ------------------------------------------
        label = self.labelsList[idx]
        return (transformImage, label)

    def __len__(self) -> int:
        return len(self.IOList)