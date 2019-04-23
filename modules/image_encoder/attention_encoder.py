import torchvision.models as models
import torch.nn as nn


class CNN(nn.Module):
    def __init__(self, hidden_size, load_pretrained=False):
        super().__init__()
        self.inception = models.inception_v3(pretrained=load_pretrained)
        # print(list(self.inception.children())[-2])
        self.features = nn.Sequential(*list(self.inception.children()[0:-1]))
        self.hidden_size = hidden_size
        # self.fcn = nn.Linear(in_features=self.inception.fc.out_features, out_features=hidden_size)

    def forward(self, imgs):
        x = imgs
        x = self.features(x)
        x = x.view(-1, 2048 * 8 * 8 / self.hidden_size, self.hidden_size)
        return x
