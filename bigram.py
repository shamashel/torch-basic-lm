from typing import Literal
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from encoder import encode

# HYPERPARAMETERS #
BATCH_SIZE = 32 # how many sequences of tokens will we process in parallel
BLOCK_SIZE = 8 # how long is a single token sequence (context length)
MAX_ITERS = 3000
EVAL_INTERVAL = 300
LEARNING_RATE = 1e-2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EMBEDDING_DIMENSIONS = 32
# --------------- #

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

  def get_batch(self, split: str = 'validate'):
    data = self.train_data if split == 'train' else self.val_data
    random_indexes = torch.randint(len(data) - self.block_size, (self.batch_size,)).to(self.device)
    context_stack = torch.stack([data[i:i+self.block_size] for i in random_indexes]).to(self.device)
    answer_stack = torch.stack([data[i+1:i+self.block_size+1] for i in random_indexes])
    return context_stack, answer_stack 

class BigramLanguageModel(nn.Module):
  def __init__(self, block_size: int, vocab_size: int, n_embd: int):
    super().__init__()
    # Create a table to embed both token and position
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    self.expected_loss: np.float64 = np.log(1/vocab_size) * -1

  def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
    # Predict next tokens
    tok_emb: torch.Tensor = self.token_embedding_table(idx)
    logits = self.lm_head(tok_emb)
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
      logits, _ = self(idx)
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
def estimate_loss(model: nn.Module, batcher: Batcher):
  out = {}
  model.eval() # set to eval phase
  for split in ['train', 'val']:
    losses = torch.zeros(EVAL_INTERVAL)
    for k in range(EVAL_INTERVAL):
      x, y = batcher.get_batch(split='train')
      logits, loss = model(x.to(DEVICE), y.to(DEVICE))
      losses[k] = loss.item()
    out[split] = losses.mean()
  model.train() # set back to training phase
  return out

def train(model: nn.Module, batcher: Batcher, iterations=MAX_ITERS, lr=LEARNING_RATE):
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  for i in range(iterations):
    if i % EVAL_INTERVAL == 0:
      losses = estimate_loss(model, batcher)
      print(f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
    context_stack, answer_stack = batcher.get_batch(split='train')
    _, loss = model(context_stack.to(DEVICE), answer_stack.to(DEVICE))
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

b = Batcher(DEVICE, BATCH_SIZE, BLOCK_SIZE)
m = BigramLanguageModel(BLOCK_SIZE, len(b.vocab), NUM_EMBEDDING_DIMENSIONS).to(DEVICE)

train(m, b)
