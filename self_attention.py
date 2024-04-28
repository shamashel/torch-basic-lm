import torch
from torch import nn


class MultiHead(nn.Module):
    def __init__(self, num_heads: int, block_size: int, n_embd: int, head_size: int, dropout: float):
        super().__init__()
        self.heads = nn.ModuleList(
            [Head(block_size, n_embd, head_size, dropout) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        out = torch.cat([head(x) for head in self.heads], dim=-1)
        out = self.proj(out)
        return self.drop(out)


class Head(nn.Module):
    def __init__(self, block_size: int, n_embd: int, head_size: int, dropout: float):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(
            torch.ones(block_size, block_size)))
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor):
        # From Attention is All You Need
        # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
        # In this case, Q K and V are all the same. d_k is head_size, and T is basically 1
        # not transposing since we just want the queries
        q: torch.Tensor = self.query(x)
        # transposing so we can get the keys (vocab) from the last dimension
        k: torch.Tensor = self.key(x).transpose(-2, -1)
        v = self.value(x)
        # Compute attention scores
        B, T, C = x.shape
        wei = q @ k  # Q * K^T
        # wei / sqrt(d_k), normalize weights
        wei: torch.Tensor = wei * (C**-0.5)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float(
            '-inf'))  # Ignore future tokens
        # aggregate values
        wei = torch.softmax(wei, dim=-1)
        wei = self.drop(wei)
        out: torch.Tensor = wei @ v
        return out
