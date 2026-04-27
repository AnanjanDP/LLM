import torch
import torch.nn as nn


class TransformerModel(nn.Module):
    def __init__(self, vocab_size=10000, d_model=512):
        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=8,
            num_encoder_layers=2,
            num_decoder_layers=2
        )

        self.fc_out = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        # x: (batch, seq)
        x = self.embedding(x)  # (batch, seq, d_model)

        x = x.permute(1, 0, 2)  # (seq, batch, d_model)

        output = self.transformer(x, x)

        output = output.permute(1, 0, 2)  # (batch, seq, d_model)

        logits = self.fc_out(output)

        return logits