import torchvision.models as models
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.inception = models.inception_v3(pretrained=True, no_fc=True, attention=True)
        self.hidden_size = hps.hidden_size

    def forward(self, imgs):
        x = imgs
        x = self.inception(x)
        x = x.view(-1, 2048 * 8 * 8 / self.hidden_size, self.hidden_size)
        return x
