"""A tiny, self-contained transformer block used only to demonstrate and test
the LoRA and quantisation modules.

This is NOT the CS 224N GPT-2 model and contains none of the course solution.
It is a minimal stack of standard ``nn.Linear`` layers whose attention
projections are named ``query`` / ``key`` / ``value`` so that the generic
``apply_lora`` helper has something realistic to target, and so the whole thing
can be int8-quantised. It downloads nothing and runs in milliseconds on CPU.
"""

from __future__ import annotations

import torch
from torch import nn


class ToyAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        assert d_model % n_head == 0
        self.n_head = n_head
        self.d_head = d_model // n_head
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.out = nn.Linear(d_model, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape
        q = self.query(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        k = self.key(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        v = self.value(x).view(B, T, self.n_head, self.d_head).transpose(1, 2)
        att = (q @ k.transpose(-2, -1)) / (self.d_head ** 0.5)
        att = att.softmax(dim=-1)
        y = (att @ v).transpose(1, 2).contiguous().view(B, T, C)
        return self.out(y)


class ToyBlock(nn.Module):
    def __init__(self, d_model: int, n_head: int):
        super().__init__()
        self.self_attention = ToyAttention(d_model, n_head)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.act = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.self_attention(self.ln1(x))
        x = x + self.fc2(self.act(self.fc1(self.ln2(x))))
        return x


class ToyTransformer(nn.Module):
    """A small multi-layer toy transformer over a fixed vocabulary."""

    def __init__(self, vocab: int = 256, d_model: int = 128, n_head: int = 4,
                 n_layer: int = 3, max_len: int = 64):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        self.layers = nn.ModuleList(ToyBlock(d_model, n_head) for _ in range(n_layer))
        self.head = nn.Linear(d_model, vocab)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, T = idx.shape
        pos = torch.arange(T, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for layer in self.layers:
            x = layer(x)
        return self.head(x)
