"""
CADGenesis-Mini model.

A small, real, trainable encoder-decoder Transformer:
  - Encoder reads LANGUAGE tokens (the design request).
  - Decoder generates GEOMETRY/FEATURE tokens (the CAD construction sequence),
    cross-attending to the language encoding -- this is the working, scaled
    version of "Geometry Attention" from the Phase 1 architecture: geometry
    generation is literally conditioned on language context via attention.
  - CAD tokens get an extra TYPE embedding (special / primitive / parameter),
    a small real instance of the "token hierarchy" idea: the model gets an
    explicit signal about what *kind* of token it's producing next, not just
    which token.

This is intentionally small enough to train on a single Colab GPU (or even
CPU) in minutes, not a claim to be the full CADGenesis-LM architecture.
"""

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=256):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, : x.size(1)]


class CADGenesisMini(nn.Module):
    def __init__(
        self,
        lang_vocab_size,
        cad_vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=256,
        dropout=0.1,
        max_len=64,
    ):
        super().__init__()
        self.d_model = d_model

        # --- Language ("text") token embedding ---
        self.lang_embed = nn.Embedding(lang_vocab_size, d_model, padding_idx=0)

        # --- CAD/geometry token embedding + type (hierarchy) embedding ---
        self.cad_embed = nn.Embedding(cad_vocab_size, d_model, padding_idx=0)
        self.type_embed = nn.Embedding(3, d_model)  # 0=special 1=primitive 2=parameter

        self.pos_enc = PositionalEncoding(d_model, max_len)

        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )

        self.out_proj = nn.Linear(d_model, cad_vocab_size)

    def forward(
        self,
        src_ids,
        tgt_in_ids,
        tgt_type_ids,
        src_key_padding_mask=None,
        tgt_key_padding_mask=None,
    ):
        """
        src_ids:        (B, S)  language token ids
        tgt_in_ids:      (B, T)  CAD token ids, decoder input (shifted right)
        tgt_type_ids:    (B, T)  CAD token *type* ids for the same positions
        """
        src = self.pos_enc(self.lang_embed(src_ids) * math.sqrt(self.d_model))
        tgt = self.cad_embed(tgt_in_ids) + self.type_embed(tgt_type_ids)
        tgt = self.pos_enc(tgt * math.sqrt(self.d_model))

        T = tgt_in_ids.size(1)
        # bool causal mask (kept the same dtype as the padding masks to avoid
        # PyTorch's mismatched-mask-type warning/slow path)
        causal_mask = torch.triu(torch.ones(T, T, dtype=torch.bool, device=tgt.device), diagonal=1)

        out = self.transformer(
            src,
            tgt,
            tgt_mask=causal_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask,
            memory_key_padding_mask=src_key_padding_mask,
        )
        return self.out_proj(out)  # (B, T, cad_vocab_size)
