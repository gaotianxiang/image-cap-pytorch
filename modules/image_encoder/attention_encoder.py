from .inception import inception_v3
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.inception = inception_v3(pretrained=True, no_fc=True, attention=True)
        self.hidden_size = hps.hidden_size
        self.num_ctx = int(2048 * 8 * 8 / self.hidden_size)

    def forward(self, imgs):
        x = imgs
        x = self.inception(x)
        # print(x.size())
        x = x.view(-1, self.num_ctx, self.hidden_size)
        return x
