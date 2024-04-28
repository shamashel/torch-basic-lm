from typing import Literal
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np

from encoder import encode, decode
from bigram import BigramLanguageModel, Batcher, estimate_loss

# HYPERPARAMETERS #
BATCH_SIZE = 32 # how many sequences of tokens will we process in parallel
BLOCK_SIZE = 8 # how long is a single token sequence (context length)
MAX_ITERS = 5000
EVAL_INTERVAL = 500
LEARNING_RATE = 1e-3
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
NUM_EMBEDDING_DIMENSIONS = 32
# --------------- #

def train_model(model: nn.Module, batcher: Batcher, iterations=MAX_ITERS, lr=LEARNING_RATE):
  optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
  for i in range(iterations):
    if i % EVAL_INTERVAL == 0:
      losses = estimate_loss(model, batcher, EVAL_INTERVAL, DEVICE)
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
  device=DEVICE,
  block_size=BLOCK_SIZE,
  vocab_size=len(b.vocab),
  n_embd=NUM_EMBEDDING_DIMENSIONS
).to(DEVICE)

def run_model(model: nn.Module, text: str, response_size: int = BLOCK_SIZE):
  data = torch.tensor(encode(text), dtype=torch.long)
  random_indexes = torch.randint(len(data) - BLOCK_SIZE, (BLOCK_SIZE,)).to(DEVICE)
  context_stack = torch.stack([data[i:i+BLOCK_SIZE] for i in random_indexes]).to(DEVICE)
  encoded = model.generate(idx = context_stack, max_new_tokens=response_size)[0]
  return decode(encoded.tolist())
  

print("Training model...")
train_model(m, b)
print("Training complete! Generating response...\n")
resp = run_model(m, 'wherefore art thou', 100)
print("Prompt: wherefore art thou") # I wonder if "why are you" would work too?
print("Response:", resp)