import torch.nn as nn
import torchvision.models as models


class CNN(nn.Module):
    def __init__(self, load_pretrained=False):
        super().__init__()
        self.inception = models.inception_v3(pretrained=load_pretrained)

    def forward(self, imgs):
        x = imgs
        x = self.inception(x)
        return x
