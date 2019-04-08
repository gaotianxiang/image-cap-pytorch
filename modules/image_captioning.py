import torch
import torch.nn as nn
import torchvision.models as models
import os
import torch.nn.functional as F
from .data_loader import SpecialTokens
import random


class CNN(nn.Module):
    def __init__(self, hidden_size, load_pretrained=False):
        super().__init__()
        self.inception = models.inception_v3(pretrained=load_pretrained)
        self.fcn = nn.Linear(in_features=self.inception.fc.out_features, out_features=hidden_size)

    def forward(self, imgs):
        x = imgs
        x = self.inception(x)
        x = self.fcn(x)
        return x


class LanguageDecoder(nn.Module):
    def __init__(self, vocabulary_size, hidden_size):
        super().__init__()
        self.vocabulary_size = vocabulary_size
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(vocabulary_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
        self.out = nn.Linear(in_features=hidden_size, out_features=vocabulary_size)

    def forward(self, input, hidden):
        output = self.embedding(input).view(1, -1, self.hidden_size)
        output = F.relu(output)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=-1)
        return output, hidden


class ImageCaptioning(nn.Module):
    def __init__(self, hidden_size, vocabulary_size, device, max_length, teacher_forcing_ratio,
                 cnn_load_pretrained=False):
        super().__init__()
        self.image_encoder = CNN(hidden_size, load_pretrained=cnn_load_pretrained)
        self.language_decoder = LanguageDecoder(vocabulary_size, hidden_size)
        self.device = device
        self.max_length = max_length
        self.hidden_size = hidden_size
        self.teacher_forcing_ratio = teacher_forcing_ratio

    def forward(self, imgs, true_sentences):
        bs = imgs.size(0)
        true_sentences = true_sentences.transpose(0, 1)
        img_fvs = self.image_encoder(imgs)
        decoder_input = torch.tensor([SpecialTokens.SOS_token] * bs, device=self.device)
        decoder_hidden = img_fvs.view(1, bs, self.hidden_size)
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        langage_output = []
        if use_teacher_forcing:
            for di in range(self.max_length):
                decoder_output, decoder_hidden = self.language_decoder(decoder_input, decoder_hidden)
                langage_output.append(decoder_output)
                decoder_input = true_sentences[di]
        else:
            for di in range(self.max_length):
                decoder_output, decoder_hidden = self.language_decoder(decoder_input, decoder_hidden)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                langage_output.append(decoder_output)
        return langage_output


def cnn_test():
    os.makedirs('./cnn_pretrained_weights', exist_ok=True)
    os.environ['TORCH_MODEL_ZOO'] = './cnn_pretrained_weights'
    cnn = CNN(256)
    print(cnn)


if __name__ == '__main__':
    cnn_test()
