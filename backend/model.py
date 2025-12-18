import torch.nn as nn
from torchvision import ops
import json


class Encoder(nn.Module):
    def __init__(self, vocab_dim, emb_dim, hidden_dim, num_layers, bidirectional):
        super().__init__()
        self.embedding = nn.Embedding(vocab_dim, emb_dim)
        self.encoder = nn.LSTM(
            emb_dim, hidden_dim, num_layers, bidirectional=bidirectional
        )

    def forward(self, input):
        # input: [batch_size, seq_len]
        x = self.embedding(input)  # -> [batch, seq, emb]

        # LSTM expects [seq_len, batch, emb]
        x = x.transpose(0, 1)

        d, (h, c) = self.encoder(x)
        # d: [seq_len, batch, hidden * dirs]

        return d, h, c


class NERClassifier(nn.Module):
    def __init__(
        self,
        vocab_dim=2000004,
        emb_dim=300,
        hidden_dim=300,
        num_layers=3,
        bidirectional=True,
        number_of_classes=8,
        dropout=0,
    ):
        super().__init__()

        self.encoder = Encoder(
            vocab_dim, emb_dim, hidden_dim, num_layers, bidirectional
        )

        bidirect_mult = 2 if bidirectional else 1
        mlp_input_dim = hidden_dim * bidirect_mult

        # MLP: [batch, ..., input_dim] -> [batch, ..., num_classes]
        self.MLP = ops.MLP(
            in_channels=mlp_input_dim,
            hidden_channels=[mlp_input_dim // 2, mlp_input_dim // 4, number_of_classes],
            dropout=dropout,
        )

        with open("data/idx2tag.json", "r", encoding="utf-8") as f:
            loaded_dict = json.load(f)
            self.idx2tag = {int(k): v for k, v in loaded_dict.items()}

    def forward(self, input_ids):
        # 1. Encode (Returns [seq_len, batch, hidden])
        d, _, _ = self.encoder(input_ids)

        # 2. Transpose to [batch, seq_len, hidden] for the MLP
        d = d.transpose(0, 1)

        # 3. Classify
        logits = self.MLP(d)
        return logits
