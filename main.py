from typing import Literal
import torch
import torch.nn as nn
from torch.nn import functional as F
import numpy as np
import os

from encoder import encode, decode
from bigram import BigramLanguageModel, Batcher, estimate_loss

# HYPERPARAMETERS #
### Impacts performance ###
BATCH_SIZE = 64  # how many sequences of tokens will we process in parallel
BLOCK_SIZE = 256  # how long is a single token sequence (context length)
LEARNING_RATE = 1e-4
NUM_EMBEDDING_DIMENSIONS = 384
NUM_HEADS = 6
NUM_LAYERS = 6
MAX_ITERS = 5000
### Others ###
EVAL_INTERVAL = 500
DROPOUT_RATE = 0.2
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
# --------------- #


def train_model(model: nn.Module, batcher: Batcher, iterations=MAX_ITERS, lr=LEARNING_RATE):
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    for i in range(iterations):
        if i % EVAL_INTERVAL == 0:
            losses = estimate_loss(model, batcher, EVAL_INTERVAL, DEVICE)
            print(
                f"step {i}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")
        context_stack, answer_stack = batcher.get_batch(split='train')
        _, loss = model(context_stack.to(DEVICE), answer_stack.to(DEVICE))
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
    return optimizer


b = Batcher(
    device=DEVICE,
    batch_size=BATCH_SIZE,
    block_size=BLOCK_SIZE
)
m = BigramLanguageModel(
    device=DEVICE,
    block_size=BLOCK_SIZE,
    vocab_size=len(b.vocab),
    n_embd=NUM_EMBEDDING_DIMENSIONS,
    n_head=NUM_HEADS,
    n_layers=NUM_LAYERS,
    dropout=DROPOUT_RATE
).to(DEVICE)


def run_model(model: nn.Module, response_size: int = BLOCK_SIZE):
    context = torch.zeros((1, 1), dtype=torch.long, device=DEVICE)
    encoded = model.generate(
        idx=context, max_new_tokens=response_size)[0]
    return decode(encoded.tolist())


if os.path.exists('model.pth'):
    print("Loading model from file...")
    checkpoint = torch.load('model.pth')
    m.load_state_dict(checkpoint['model_state_dict'])
    print("Model loaded!")
else:
    print("Training model...")
    optimizer = train_model(m, b)
    torch.save({
        'model_state_dict': m.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, 'model.pth')
    print("Training complete!")
print("Generating response...\n")
resp = run_model(m, 256)
print("Response:", resp)
