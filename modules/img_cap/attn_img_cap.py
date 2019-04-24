import torch
import torch.nn as nn
import random
from modules.image_encoder.get_img_encoder import get_img_encoder
from modules.text_decoder.get_text_decoder import get_language_decoder


class ImageCaptioning(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.image_encoder = get_img_encoder(hps)
        self.language_decoder = get_language_decoder(hps)
        self.device = hps.device
        self.max_length = hps.max_caption_length
        self.hidden_size = hps.hidden_size
        self.teacher_forcing_ratio = hps.teacher_forcing_ratio

    def forward(self, imgs, true_sentences):
        bs = imgs.size(0)
        true_sentences = true_sentences.transpose(0, 1)
        img_fvs = self.image_encoder(imgs)
        # print(img_fvs.size())
        # img_fvs = self.img_fvs_to_hs(img_fvs)
        decoder_input = torch.tensor([0] * bs, device=self.device)
        decoder_hidden = img_fvs.mean(1).view(1, bs, self.hidden_size)
        use_teacher_forcing = True if random.random() < self.teacher_forcing_ratio else False
        langage_output = []
        if use_teacher_forcing:
            for di in range(self.max_length):
                decoder_output, decoder_hidden = self.language_decoder(decoder_input, decoder_hidden, img_fvs)
                langage_output.append(decoder_output)
                decoder_input = true_sentences[di]
        else:
            for di in range(self.max_length):
                decoder_output, decoder_hidden = self.language_decoder(decoder_input, decoder_hidden, img_fvs)
                topv, topi = decoder_output.topk(1)
                decoder_input = topi.squeeze().detach()
                langage_output.append(decoder_output)
        return langage_output
