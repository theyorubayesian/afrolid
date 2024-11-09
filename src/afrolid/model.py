import math
from typing import Any, Optional

import torch
from torch import nn
from torch.nn import functional as F


class SinusoidalPositionalEmbedding(nn.Module):
    """Copied from Fairseq's SinusoidalPositionalEmbedding module.
    https://github.com/facebookresearch/fairseq/blob/main/fairseq/modules/sinusoidal_positional_embedding.py

    Differs slightly from the Vaswani et al. version
    """
    def __init__(self, embedding_dim, padding_idx, init_size=1024, auto_expand=True):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.padding_idx = padding_idx if padding_idx is not None else 0
        self.register_buffer(
            "weights",
            SinusoidalPositionalEmbedding.get_embedding(
                init_size, embedding_dim, padding_idx
            ),
            persistent=False,
        )
        self.max_positions = int(1e5)
        self.auto_expand = auto_expand

    @staticmethod
    def get_embedding(
        num_embeddings: int, embedding_dim: int, padding_idx: Optional[int] = None
    ):
        """Build sinusoidal embeddings.

        This matches the implementation in tensor2tensor, but differs slightly
        from the description in Section 3.5 of "Attention Is All You Need".
        """
        half_dim = embedding_dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float) * -emb)
        emb = torch.arange(num_embeddings, dtype=torch.float).unsqueeze(
            1
        ) * emb.unsqueeze(0)
        emb = torch.cat([torch.sin(emb), torch.cos(emb)], dim=1).view(
            num_embeddings, -1
        )
        if embedding_dim % 2 == 1:
            # zero pad
            emb = torch.cat([emb, torch.zeros(num_embeddings, 1)], dim=1)
        if padding_idx is not None:
            emb[padding_idx, :] = 0
        return emb

    @staticmethod
    def create_position_ids_from_input_ids(input_ids: torch.Tensor, padding_idx: int = 1, past_key_values_length=0) -> torch.Tensor:
        """
        Copied from the HuggingFace's `transformers` library.
        
        Replace non-padding symbols with their position numbers. Position numbers begin at padding_idx+1. Padding symbols
        are ignored. This is modified from fairseq's `utils.make_positions`.

        Args:
            x: torch.Tensor x:

        Returns: torch.Tensor
        """
        # The series of casts and type-conversions here are carefully balanced to both work with ONNX export and XLA.
        mask = input_ids.ne(padding_idx).int()
        incremental_indices = (torch.cumsum(mask, dim=1).type_as(mask) + past_key_values_length) * mask
        return incremental_indices.long() + padding_idx
    
    def forward(
        self,
        input: torch.Tensor,
        incremental_state: Optional[Any] = None,
        timestep: Optional[torch.Tensor] = None,
        positions: Optional[Any] = None,
    ):
        """Input is expected to be of size [bsz x seqlen]."""
        bsz, seq_len = input.size(0), input.size(1)
        max_pos = self.padding_idx + 1 + seq_len
        weights = self.weights

        if max_pos > self.weights.size(0):
            # If the input is longer than the number of pre-computed embeddings,
            # compute the extra embeddings on the fly.
            # Only store the expanded embeddings if auto_expand=True.
            # In multithreading environments, mutating the weights of a module
            # may cause trouble. Set auto_expand=False if this happens.
            weights = SinusoidalPositionalEmbedding.get_embedding(
                max_pos, self.embedding_dim, self.padding_idx
            ).to(self.weights)
            if self.auto_expand:
                self.weights = weights

        if incremental_state is not None:
            # positions is the same for every token when decoding a single step
            pos = timestep.view(-1)[0] + 1 if timestep is not None else seq_len
            return weights[self.padding_idx + pos, :].expand(bsz, 1, -1)
        
        positions = self.create_position_ids_from_input_ids(input, self.padding_idx)
        return (
            weights.index_select(0, positions.view(-1)).view(bsz, seq_len, -1).detach()
        )


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
        self.pos_encoder = SinusoidalPositionalEmbedding(d_model, padding_idx=1, init_size=max_position_embeddings)

        self.encoder_emb = nn.Embedding(vocab_size, d_model, padding_idx=1)
        self.decoder_emb = nn.Embedding(num_classes, d_model, padding_idx=1)
        
        self.output_projection = nn.Linear(d_model, num_classes, bias=False)

    def forward(self, input_ids: torch.Tensor, decoder_input_ids: torch.Tensor = None) -> torch.Tensor:        
        encoder_embeds = self.encoder_emb(input_ids) * self.scale_factor
        encoder_embeds += self.pos_encoder(input_ids)
        encoder_output = self.encoder(encoder_embeds)
        
        batch_size = input_ids.size(0)
        if decoder_input_ids is None:
            # We prefix with 2 to indicate the start of the sequence
            decoder_input_ids = torch.full((batch_size, 1), fill_value=2, dtype=torch.long, device=input_ids.device)
        
        decoder_embeds = self.decoder_emb(decoder_input_ids) * self.scale_factor
        decoder_embeds += self.pos_encoder(decoder_input_ids, incremental_state=True)
        decoder_output = self.decoder(decoder_embeds, encoder_output)

        # Output projection
        logits = self.output_projection(decoder_output)
        return F.softmax(logits, dim=-1)
