import math

import torch
from torch import nn
from torch.nn import functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_position: int = 514):
        super(PositionalEncoding, self).__init__()

        pe = torch.zeros(max_position, d_model)
        position = torch.arange(0, max_position, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class AfroLIDModel(nn.Transformer):
    def __init__(
        self,
        n_encoder_layers: int = 12,
        n_decoder_layers: int = 12,
        vocab_size: int = 64001,
        num_classes: int = 528,
        d_model: int = 768,
        nhead: int = 12,
        d_ff: int = 3072,
        max_position_embeddings: int = 1026,
        dropout: int = 0.1,
        activation: str = 'relu'
    ) -> None:
        super(AfroLIDModel, self).__init__(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_ff,
            num_encoder_layers=n_encoder_layers,
            num_decoder_layers=n_decoder_layers,
            dropout=dropout,
            activation=activation,
            norm_first=False,
            batch_first=True,
            layer_norm_eps=1e-5
        )
        self.scale_factor = math.sqrt(d_model)
        self.pos_encoder = PositionalEncoding(d_model, max_position_embeddings)

        self.encoder_emb = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.decoder_emb = nn.Embedding(num_classes, d_model, padding_idx=1)
        
        self.output_projection = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor = None, attention_mask=None) -> torch.Tensor:
        encoder_embeds = self.pos_encoder(self.encoder_emb(input_ids) * self.scale_factor)
        encoder_output = self.encoder(encoder_embeds)
        
        batch_size = input_ids.size(0)
        if decoder_input_ids is None:
            # We prefix with 2 to indicate the start of the sequence
            decoder_input_ids = torch.full((batch_size, 1), fill_value=2, dtype=torch.long, device=input_ids.device)
        
        decoder_embeds = self.pos_encoder(self.decoder_emb(decoder_input_ids) * self.scale_factor)
        decoder_output = self.decoder(decoder_embeds, encoder_output)
        
        logits = self.output_projection(decoder_output)

        return F.softmax(logits, dim=-1)
