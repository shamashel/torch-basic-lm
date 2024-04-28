from typing import Literal
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from encoder import encode, decode
from self_attention import Head, MultiHead

# HYPERPARAMETERS #
BATCH_SIZE = 32 # how many sequences of tokens will we process in parallel
BLOCK_SIZE = 8 # how long is a single token sequence (context length)
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 1e-3
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

  def get_batch(self, split: str = 'val'):
    data = self.train_data if split == 'train' else self.val_data
    random_indexes = torch.randint(len(data) - self.block_size, (self.batch_size,)).to(self.device)
    context_stack = torch.stack([data[i:i+self.block_size] for i in random_indexes]).to(self.device)
    answer_stack = torch.stack([data[i+1:i+self.block_size+1] for i in random_indexes])
    return context_stack, answer_stack 

class BigramLanguageModel(nn.Module):
  def __init__(self, block_size: int, vocab_size: int, n_embd: int):
    super().__init__()
    self.block_size = block_size
    self.vocab_size = vocab_size
    self.n_embd = n_embd
    # Create a table to embed both token and position
    self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
    self.position_embedding_table = nn.Embedding(block_size, n_embd)
    self.sa_head = MultiHead(4, block_size, n_embd, n_embd//4)
    self.lm_head = nn.Linear(n_embd, vocab_size)
    self.expected_loss: np.float64 = np.log(1/vocab_size) * -1

  def forward(self, idx: torch.Tensor, targets: torch.Tensor = None):
    # Predict next tokens
    B, T = idx.shape
    tok_emb: torch.Tensor = self.token_embedding_table(idx)
    pos_emb = self.position_embedding_table(torch.arange(T, device=DEVICE))
    x: torch.Tensor = tok_emb + pos_emb
    x = self.sa_head(x)
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
def estimate_loss(model: nn.Module, batcher: Batcher):
  out = {}
  model.eval() # set to eval phase
  for split in ['train', 'val']:
    losses = torch.zeros(EVAL_INTERVAL)
    for k in range(EVAL_INTERVAL):
      x, y = batcher.get_batch(split=split)
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

b = Batcher(
  device=DEVICE,
  batch_size=BATCH_SIZE, 
  block_size=BLOCK_SIZE
)
m = BigramLanguageModel(
  block_size=BLOCK_SIZE,
  vocab_size=len(b.vocab),
  n_embd=NUM_EMBEDDING_DIMENSIONS
).to(DEVICE)

def run_model(text: str, response_size: int = BLOCK_SIZE):
  data = torch.tensor(encode(text), dtype=torch.long)
  random_indexes = torch.randint(len(data) - BLOCK_SIZE, (BLOCK_SIZE,)).to(DEVICE)
  context_stack = torch.stack([data[i:i+BLOCK_SIZE] for i in random_indexes]).to(DEVICE)
  encoded = m.generate(idx = context_stack, max_new_tokens=response_size)[0]
  return decode(encoded.tolist())
  

train(m, b)
resp = run_model('wherefore art thou', 100)
print("Prompt: 'wherefore art thou'")
print("Response:", 'wherefore art thou' + resp) # I wonder if "why are you" would work too?