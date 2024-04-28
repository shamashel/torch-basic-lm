from typing import Literal
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from encoder import encode, decode
from self_attention import Head, MultiHead

class Batcher():
  def __init__(self, device: Literal['cuda', 'cpu'], batch_size: int, block_size: int):
    self.device = device
    self.batch_size = batch_size
    self.block_size = block_size
    with open('input.txt', 'r', encoding='utf-8') as f:
      text = f.read()
      my_tensors = torch.tensor(encode(text), dtype=torch.long)
      n = int(0.9*len(my_tensors))
      self.train_data = my_tensors[:n]
      self.val_data = my_tensors[n:]
      self.vocab = set(text)

  def get_batch(self, split: str = 'val'):
    data = self.train_data if split == 'train' else self.val_data
    random_indexes = torch.randint(len(data) - self.block_size, (self.batch_size,)).to(self.device)
    context_stack = torch.stack([data[i:i+self.block_size] for i in random_indexes]).to(self.device)
    answer_stack = torch.stack([data[i+1:i+self.block_size+1] for i in random_indexes])
    return context_stack, answer_stack 

class FeedForward(nn.Module):
  def __init__(self, n_embd: int):
    super().__init__()
    self.net = nn.Sequential(
      nn.Linear(n_embd, n_embd * 4),
      nn.ReLU(),
      nn.Linear(n_embd * 4, n_embd)
    )

  def forward(self, x: torch.Tensor):
    return self.net(x)

class Block(nn.Module):
  def __init__(self, n_embd: int, block_size: int, n_head: int):
    super().__init__()
    head_size = n_embd // n_head
    self.sa_head = MultiHead(n_head, block_size, n_embd, head_size)
    self.ffwd = FeedForward(n_embd)

  def forward(self, x: torch.Tensor):
    x = x + self.sa_head(x)
    x = x + self.ffwd(x)
    return x

class BigramLanguageModel(nn.Module):
  def __init__(self, device: Literal['cuda', 'cpu'], block_size: int, vocab_size: int, n_embd: int):
    super().__init__()
    self.block_size = block_size
    self.vocab_size = vocab_size
    self.n_embd = n_embd
    self.device = device
    # Create a table to embed both token and position
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    self.expected_loss: np.float64 = np.log(1/vocab_size) * -1
    self.blocks = nn.Sequential(
      Block(n_embd, block_size=block_size, n_head=4),
      Block(n_embd, block_size=block_size, n_head=4),
      Block(n_embd, block_size=block_size, n_head=4),
    )

  def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
    # Predict next tokens
    B, T = idx.shape
    tok_emb: torch.Tensor = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T, device=self.device))
    x: torch.Tensor = tok_emb + pos_emb
    x = self.blocks(x)
    logits: torch.Tensor = self.lm_head(x)
    if targets is None:
      loss = 0
    else:
      batch, block, vocab = logits.shape
      # Reformat logits and targets so each entry can be compared
      logits = logits.view(batch * block, vocab)
      targets = targets.view(batch * block)
      # Compare predicted tokens to actual
      loss = F.cross_entropy(logits, targets)
    return logits, loss
  
  # Given a 2d matrix of dimensions token and sentence
  # generate new tokens in the next sentence
  def generate(self, idx: torch.Tensor, max_new_tokens: int):
    for _ in range(max_new_tokens):
      cropped_idx = idx[:, -self.block_size:] # Crop out the last block_size tokens
      logits, _ = self(cropped_idx)
      # Logits has dimensions token, sentence, token_list
      # We want to make a new sentence, so only look at the last sentence
      logits = logits[:, -1, :]
      # Get possible next tokens and select one
      probabilities = F.softmax(logits, dim=-1)
      idx_next = torch.multinomial(probabilities, num_samples=1)
      # Add the new token to the end of the tensor
      idx = torch.cat((idx, idx_next), dim=1)
    return idx

@torch.no_grad()
def estimate_loss(model: nn.Module, batcher: Batcher, eval_interval: int, device: Literal['cuda', 'cpu'] = 'cuda'):
  out = {}
  model.eval() # set to eval phase
  for split in ['train', 'val']:
    losses = torch.zeros(eval_interval)
    for k in range(eval_interval):
      x, y = batcher.get_batch(split=split)
      logits, loss = model(x.to(device), y.to(device))
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train() # set back to training phase
  return out