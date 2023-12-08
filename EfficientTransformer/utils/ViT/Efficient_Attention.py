'''
# System --> Windows & Python3.8.0
# Author --> Illusionna
'''
# -*- Encoding: UTF-8 -*-


from torch import device
from linformer import Linformer
from utils.ViT.vit_pytorch.efficient import ViT


class TRANSFORMER:
    """
    Visual Transformer: Efficient Attention with Linformer.
    """
    def __init__(
            self,
            *args,
            device:device,
            classNumbers:int,
            channelNumbers:int,
            **kwargs
    ) -> None:
        self.device = device
        self.classNumbers = classNumbers
        self.channelNumbers = channelNumbers

    def Initialize(self) -> ViT:
        efficientTransformer = Linformer(
            dim = 128,
            depth = 12,
            heads = 8,
            seq_len = 7 * 7 + 1,  # 7x7 patches + 1 cls-token.
            k = 64
        )
        model = ViT(
            dim = 128,
            image_size = 224,
            patch_size = 32,
            num_classes = self.classNumbers,
            transformer = efficientTransformer,
            channels = self.channelNumbers,
        ).to(self.device)
        return model