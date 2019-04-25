from .vgg import vgg16
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.vgg = vgg16(pretrained=True, attention=True)
        self.hidden_size = hps.hidden_size

    def forward(self, imgs):
        x = imgs
        x = self.vgg(x)
        return x
