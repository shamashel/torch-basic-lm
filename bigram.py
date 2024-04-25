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
# --------------- #

class Batcher():
  def __init__(self):
    self.device = DEVICE
    with open('input.txt', 'r', encoding='utf-8') as f:
      text = f.read()
      self.vocab = set(text)
      my_tensors = torch.tensor(encode(text), dtype=torch.long)
      n = int(0.9*len(my_tensors))
      self.train_data = my_tensors[:n]
      self.val_data = my_tensors[n:]

  def get_batch(self, batch_size: int = BATCH_SIZE, block_size: int = BLOCK_SIZE, split: str = 'validate'):
    data = self.train_data if split == 'train' else self.val_data
    random_indexes = torch.randint(len(data) - block_size, (batch_size,)).to(self.device)
    context_stack = torch.stack([data[i:i+block_size] for i in random_indexes]).to(self.device)
    answer_stack = torch.stack([data[i+1:i+block_size+1] for i in random_indexes])
    return context_stack, answer_stack 

class BigramLanguageModel(nn.Module):
  def __init__(self, vocab_size: int):
    super().__init__()
    self.device = DEVICE
    self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)
    self.expected_loss: np.float64 = np.log(1/vocab_size) * -1
    self.to(self.device)

  def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
    # Predict next tokens
    logits: torch.Tensor = self.token_embedding_table(idx)
    batch, block, vocab = logits.shape
    # Reformat logits and targets so each entry can be compared
    logits = logits.view(batch * block, vocab)
    targets = targets.view(batch * block)
    if targets is None:
      loss = 0
    else:
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
    
def train_bag_of_words(batcher: Batcher):
  context_stack, answer_stack = batcher.get_batch(split='train')
  tril = torch.tril(torch.ones(BLOCK_SIZE, BLOCK_SIZE))
  wei = torch.zeros((BLOCK_SIZE, BLOCK_SIZE))
  wei = wei.masked_fill(tril == 0, float('-inf'))
  wei = F.softmax(wei, dim=-1)
  xbow = wei @ context_stack
  return xbow, answer_stack


b = Batcher()
m = BigramLanguageModel(vocab_size=len(b.vocab)).to(DEVICE)

train(m, b)
