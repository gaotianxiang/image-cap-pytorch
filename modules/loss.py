import torch
import torch.nn as nn


class SparseCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.nll = nn.NLLLoss(reduce=False)

    def forward(self, true_sentences, generated_sentences, masks):
        true_sentences = true_sentences.transpose(0, 1)
        generated_sentences = torch.cat(generated_sentences).transpose(0, 1)
        masks = masks.transpose(0, 1)
        loss = self.nll(generated_sentences, true_sentences) * masks
        loss = loss.sum() / masks.sum()
        return loss
