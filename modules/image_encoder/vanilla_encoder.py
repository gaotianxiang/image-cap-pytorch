import torch.nn as nn
import torch.nn.functional as F
from .inception import inception_v3


class CNN(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.inception = inception_v3(pretrained=True, no_fc=True, attention=False)

    def forward(self, imgs):
        """

        Args:
            imgs: batch of images N * H * W * C

        Returns:
            x: batch of feature vectors N * 2048
        """
        x = imgs
        x = self.inception(x)
        return x
