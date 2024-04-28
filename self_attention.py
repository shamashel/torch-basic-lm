import torch
from torch import nn

class MultiHead(nn.Module):
  def __init__(self, num_heads: int, block_size: int, n_embd: int, head_size: int):
    super().__init__()
    self.heads = nn.ModuleList([
      Head(block_size, n_embd, head_size) for _ in range(num_heads)
    ])
  def forward(self, x: torch.Tensor):
    return torch.cat([head(x) for head in self.heads], dim=-1)

class Head(nn.Module):
  def __init__(self, block_size: int, n_embd: int, head_size: int):
    super().__init__()
    self.key = nn.Linear(n_embd, head_size, bias=False)
    self.query = nn.Linear(n_embd, head_size, bias=False)
    self.value = nn.Linear(n_embd, head_size, bias=False)
    self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))

  def forward(self, x: torch.Tensor):
    # From Attention is All You Need
    # Attention(Q, K, V) = softmax(QK^T / sqrt(d_k))V
    # In this case, Q K and V are all the same. d_k is head_size, and T is basically 1
    q: torch.Tensor = self.query(x) # not transposing since we just want the queries
    k: torch.Tensor = self.key(x).transpose(-2,-1) # transposing so we can get the keys (vocab) from the last dimension
    v = self.value(x)
    # Compute attention scores
    B,T,C = x.shape
    wei = q @ k # Q * K^T
    wei: torch.Tensor = wei * (C**-0.5) # wei / sqrt(d_k), normalize weights
    wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # Ignore future tokens
    # aggregate values
    wei = torch.softmax(wei, dim=-1)
    out: torch.Tensor = wei @ v
    return out