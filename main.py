import torch

from encoder import encode

with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

my_tensors = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(my_tensors))
train_data = my_tensors[:n]
val_data = my_tensors[n:]

block_size = 8
batch_size = 4

def get_batch(split: str):
    data = train_data if split == 'train' else val_data
    random_indexes = torch.randint(len(data) - block_size, (batch_size,))
    context_stack = torch.stack([data[i:i+block_size] for i in random_indexes])
    answer_stack = torch.stack([data[i+1:i+block_size+1] for i in random_indexes])
    return context_stack, answer_stack

print(my_tensors)