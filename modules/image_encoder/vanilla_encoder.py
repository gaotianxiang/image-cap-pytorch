import torch.nn as nn
from .vgg import vgg16


class CNN(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.vgg = vgg16(pretrained=True, attention=False)

    def forward(self, imgs):
        """

        Args:
            imgs: batch of images N * H * W * C

        Returns:
            x: batch of feature vectors N * 2048
        """
        x = imgs
        x = self.vgg(x)
        return x
