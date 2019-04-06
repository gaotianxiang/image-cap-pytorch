import torch
import torch.nn as nn
import torchvision.models as models
import os


class CNN(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.inception = models.inception_v3(pretrained=False)
        self.fcn = nn.Linear(in_features=self.inception.fc.out_features, out_features=hidden_size)

    def forward(self, imgs):
        x = imgs
        x = self.inception(x)
        x = self.fcn(x)
        return x


def cnn_test():
    os.makedirs('./cnn_pretrained_weights', exist_ok=True)
    os.environ['TORCH_MODEL_ZOO'] = './cnn_pretrained_weights'
    cnn = CNN(256)
    print(cnn)


if __name__ == '__main__':
    cnn_test()
