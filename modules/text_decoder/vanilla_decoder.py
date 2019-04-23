import torch.nn as nn
import torch.nn.functional as F


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
