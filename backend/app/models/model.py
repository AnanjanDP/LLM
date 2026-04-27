import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, vocab_size, d_model=128, nhead=4, num_layers=2):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            batch_first=True  # ✅ Fix dimension issue
        )

        self.transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers
        )

        self.fc = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: [batch, seq]
        x = self.embedding(x)          # [batch, seq, d_model]
        x = self.transformer(x)        # [batch, seq, d_model]
        x = self.fc(x)                 # [batch, seq, vocab_size]
        return x