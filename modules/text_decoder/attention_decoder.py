import torch.nn as nn
import torch.nn.functional as F
import torch


class LanguageDecoder(nn.Module):
    def __init__(self, hps):
        super().__init__()
        self.hps = hps
        self.vocabulary_size = hps.vocabulary_size
        self.hidden_size = hps.hidden_size
        self.embedding = nn.Embedding(hps.vocabulary_size, hps.hidden_size)
        self.gru = nn.GRU(hps.hidden_size, hps.hidden_size)
        self.out = nn.Linear(in_features=hps.hidden_size, out_features=hps.vocabulary_size)
        self.dropout = nn.Dropout(p=hps.dropout_rate)

        self.attn = Attention(256, hps.hidden_size)

    def forward(self, input, hidden, context):
        input_embedding = self.embedding(input).view(1, -1, self.hidden_size)
        input_embedding = self.dropout(input_embedding)
        output = self.attn(input_embedding, hidden, context)
        output, hidden = self.gru(output, hidden)
        output = F.log_softmax(self.out(output[0]), dim=-1)
        return output, hidden


class Attention(nn.Module):
    def __init__(self, num_ctx_vec, hidden_size):
        super(Attention, self).__init__()
        self.num_ctx_vec = num_ctx_vec
        self.hidden_size = hidden_size
        self.attn = nn.Linear(hidden_size * 2, num_ctx_vec)
        self.attn_combine = nn.Linear(hidden_size * 2, hidden_size)

    def forward(self, input_embedding, prev_hidden, context):
        attn_weights = F.softmax(self.attn(torch.cat((input_embedding[0], prev_hidden[0]), dim=1)))
        attn_applied = torch.bmm(attn_weights.unsqueeze(1),
                                 context).squeeze(1)
        input_ctx_combine = torch.cat((input_embedding[0], attn_applied), dim=1)
        output = self.attn_combine(input_ctx_combine).unsqueeze(0)
        output = F.relu(output)
        return output

