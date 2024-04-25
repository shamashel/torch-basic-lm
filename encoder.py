with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }

def encode(s: str):
    return [stoi[c] for c in s]

def decode(l: list[int]):
    return ''.join([itos[i] for i in l])