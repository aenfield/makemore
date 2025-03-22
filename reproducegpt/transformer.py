import torch
import torch.nn as nn
from torch.nn import functional as F

# Based on Andrej Karpathy "Let's build GPT" at https://www.youtube.com/watch?v=kCc8FmEb1nY&t=528s
# mostly his code; for a lot of the initial stuff I have more notes about the code, that I took, in gpt-dev.ipynb

# I created this file by starting w/ the bigram.py code when he started doing transformer
# stuff in the lecture, at around 58m.

# hyper params
batch_size = 32 # how many independent seqs we process in parallel
block_size = 8 # what's the max context length for prediction?
max_iters = 500
eval_interval = 300 # eval and output perf measure how often? per this number of iters
learning_rate = 1e-2
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available else 'cpu'
(print(f"Training on {device}."))
eval_iters = 200 # how many batches to average over when estimating loss/performance
n_embd = 32
# --------

torch.manual_seed(1337)

# !wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# all the unique chars - our vocab - that exist in the training text
chars = sorted(list(set(text)))
vocab_size = len(chars)
# mapping between chars and ints
stoi = { ch:i for i, ch in enumerate(chars) }
itos = { i:ch for i, ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encode a str (list of chars) and output a list of ints
decode = lambda l: ''.join([itos[i] for i in l]) # decode a list of ints and output a str (list of chars)

# train and test splits
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data)) # train on first 90% of text, keep rest for val
train_data = data[:n]
val_data = data[n:]

# load data
def get_batch(split):
    # gen a small batch of data of data of inputs x and targets y - see gpt-dev for details
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    # get a bunch of batches, calc the loss for each, and return the mean
    out = {}
    model.eval() # switch model to mode suitable for eval (turn off random dropout and use running estimates for batch norm)
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train() # and turn back on training mode
    return out

# very simple bigram model
class TransformerLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # read the logits for the next token directly from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # each position in the context/block gets its own embedding vector
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C) # C here is n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B, T, C) # combine emb for tokens and position, due to broadcasting (T,C) across the batches
        logits = self.lm_head(x) # (B,T,C) # C is here vocab_size

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        # idx is a (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # predict for every location in every sequence
            logits, loss = self(idx)
            # focus on the last item in each sequence
            logits = logits[:, -1, :] # (B, C)
            # apply softmax op to get probs from non-normalized logits
            probs = F.softmax(logits, dim=-1) # (B, C)
            # and sample to get a token
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx
    
model = TransformerLanguageModel()
m = model.to(device)

optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# and train
for iter in range(max_iters):
    # periodically eval the loss on train and val sets
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a bunch of data
    xb, yb = get_batch('train')

    # eval the loss on the sampled data
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# and when done, generate from the trained model, starting with a single newline - token w/ idx 0 - as context
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=200)[0].tolist()))


