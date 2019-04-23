import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F


class CNN(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.inception = models.inception_v3(pretrained=True)
        self.features = nn.Sequential(*list(self.inception.children())[0:-1])
        self.dropout = nn.Dropout()

    def forward(self, imgs):
        """

        Args:
            imgs: batch of images N * H * W * C

        Returns:
            x: batch of feature vectors N * 2048
        """
        x = imgs
        x = self.features(x)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = self.dropout(x)
        x = x.view(x.size(0), -1)
        return x
