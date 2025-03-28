import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.amp import autocast, GradScaler # try out AMP (automatic mixed precision)

import time
import argparse

# Based on Andrej Karpathy "Let's build GPT" at https://www.youtube.com/watch?v=kCc8FmEb1nY&t=528s
# mostly his code; for a lot of the initial stuff I have more notes about the code, that I took, in gpt-dev.ipynb

# I created this file by starting w/ the bigram.py code when he started doing transformer
# stuff in the lecture, at around 58m.

# hyper params
batch_size = 32 # how many independent seqs we process in parallel
block_size = 8 # what's the max context length for prediction?
# max_iters = 500
max_iters = 5000 # higher than orig ~3K because self-attention has a lower learning rate
eval_interval = 300 # eval and output perf measure how often? per this number of iters
#learning_rate = 1e-2
learning_rate = 1e-3 # decreased a bit because self-attention needs to go a bit slower
device = torch.accelerator.current_accelerator() if torch.accelerator.is_available else 'cpu'
(print(f"Using device: {device}."))
eval_iters = 200 # how many batches to average over when estimating loss/performance
n_embd = 32
n_head = 4
n_layer = 4
dropout = 0.0

# redefine some params for scale-up - I'll leave the ones above alive and just overwrite
# them here for ease of (un)commenting out. These are the ones that Karpathy suggested and that took
# his A100 ~15m to train on (at the time of the lecture, in early 2023). My MacBook Pro M3 18GB took
# 1hr15m to train on these, and my 5080 on WSL/Ubuntu only ~7.5m. Then, when I added AMP it reduced
# the amt of training time by ~18% to ~6m. Each of these achieved a validation loss of ~1.48 or so 
# with 5000 iterations. 
batch_size = 64
block_size = 256
eval_interval = 500
learning_rate = 3e-4
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

# # now, given the quick time noted above and the hopeful additional memory headroom enabled by AMP, I'll
# # try scaling up further using the settings recommended by ChatGPT when I had it take a look at the code.
n_embd = 768
n_head = 12
n_layer = 12
# w/ the above 64 and also 48 batch sizes I had GPU memory usage at ~95% when ChatGPT says I want it 80-90%; 
# 32 gave ~72%, and ChatGPT says I don't need batch size to be multiples of 2 and making sure to use all the memory
# I can is most important, so I'll try 40, which gives me right at ~90%.
batch_size = 40 

# # With AMP the training kept hanging w/ no GPU activity, but w/ AMP turned off it'd run to completion, and w/ less
# # GPU memory usage (the opposite of what I'd expect?), so I'll try w/ a bigger batch size and no AMP. 56 gives 
# # ~88% GPU memory usage w/o AMP. And... the model crazy overfits pretty quickly... I get to a loss of 1.53 at 3000
# # iterations and then it heads back up pretty quickly and by 10K iterations I'm at 3.26. The generated text even w/
# # the overfit model seems good - maybe better than I saw before? - so now I'm curious about what the text from the
# # model w/ the best val loss would be, which means I need to manually tune params so it ends before it overfits, or
# # just implement simple early stopping. I'll try the latter.
batch_size = 56
max_iters = 10000

# now I'll try again w/ the crazy complicated/deep network
eval_interval = 250 
# --------

# torch.manual_seed(1337)

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
    # gen a batch of data of data of inputs x and targets y - see gpt-dev for details
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i + block_size] for i in ix])
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

class Head(nn.Module):
    """ One self-attention head. """

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)   # (B, T, C)
        q = self.query(x) # (B, T, C)
        # and compute attention scores - 'affinities'
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei) # randomly prevent some of the attention nodes from communicating 
        # do weighted agg of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) -> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ Multiple self-attention heads in parallel. """

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd) # to rejoin residual pathway
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ Simple linear layer followed by a non-linearity. """
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd), # to rejoin, like the proj in MultiHeadAttention
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication with self-attention followed by computation with feed-forward. """

    def __init__(self, n_embd, n_head):
        # n_embd: embedding dimension, n_head: number of heads we'd like
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # x = self.sa(x)
        # x = self.ffwd(X)
        # do w/ residual connections to help w/ optimization/training since the network's becoming pretty deep
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x)) 
        return x


# very simple transformer model
class TransformerLanguageModel(nn.Module):

    def __init__(self):
        super().__init__()
        # read the logits for the next token directly from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # each position in the context/block gets its own embedding vector
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm after all blocks
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        # idx and targets are both (B,T) tensors of integers

        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx) # (B,T,C) # C here is n_embd
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T,C)
        x = tok_emb + pos_emb # (B, T, C) # combine emb for tokens and position, due to broadcasting (T,C) across the batches
        x = self.blocks(x) # (B, T, C)
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
        self.eval()
        with torch.no_grad():
            # idx is a (B, T) array of indices in the current context
            for _ in range(max_new_tokens):
                # crop idx to the last block_size tokens - position embedding's only set up for block_size tokens, we only work w/ that many 
                idx_cond = idx[:, -block_size:]
                # predict for every location in every sequence
                logits, loss = self(idx_cond)
                # focus on the last item in each sequence
                logits = logits[:, -1, :] # (B, C)
                # apply softmax op to get probs from non-normalized logits
                probs = F.softmax(logits, dim=-1) # (B, C)
                # and sample to get a token
                idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
                # append sampled index to the running sequence
                idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        self.train()

        return idx
    
    @torch.no_grad()
    def estimate_loss(self):
        # get a bunch of batches, calc the loss for each, and return the mean
        out = {}
        self.eval() # switch model to mode suitable for eval (turn off random dropout and use running estimates for batch norm)
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X, Y = get_batch(split)
                logits, loss = self(X, Y)
                losses[k] = loss.item()
            out[split] = losses.mean()
        self.train() # and turn back on training mode
        return out


# these next little funcs could all be model instance methods but I'll leave them as they cam from Karpathy/as they resulted from simple DRY
def count_parameters(model):
    return torch.nn.utils.parameters_to_vector(model.parameters()).numel()

def save_model_weights(model, iter, in_training=False):
    torch.save(model.state_dict(), f'model_weights_{count_parameters(model)}_{batch_size}batch_{in_training}intraining.pth')

def generate_tokens(model, token_count_to_generate):
    # and when done, generate from the trained model, starting with a single newline - token w/ idx 0 - as context
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    return decode(model.generate(context, max_new_tokens=token_count_to_generate)[0].tolist())


def train():
    best_val_loss = float('inf')
    max_patience = 3
    patience_counter = 0 

    model_create = TransformerLanguageModel()
    model = model_create.to(device)
    print(f'Parameter count: {count_parameters(model)}.')

    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    use_amp = device == torch.device('cuda') # AMP only works on CUDA devices (some of it might work on MPS but I won't figure that out for now since I'll do bigger training w/ NVidia hardware, likely)
    use_amp = False
    print(f'Using AMP? {use_amp}.')
    scaler = GradScaler(device.type)

    start_time = time.perf_counter()

    # and train
    for iter in range(max_iters):
        # periodically eval the loss on train and val sets and consider stopping early
        if iter % eval_interval == 0:
            losses = model.estimate_loss()
            print(f"Step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}.")

            if losses['val'] < best_val_loss:
                best_val_loss = losses['val']
                patience_counter = 0
                save_model_weights(model, iter, in_training=True)
            else:
                patience_counter += 1
                print(f'No val loss improvement - patience: {patience_counter}/{max_patience}.')
                if patience_counter >= max_patience:
                    print('Early stopping triggered - stopping.')
                    break

        xb, yb = get_batch('train')

        optimizer.zero_grad(set_to_none=True)

        if use_amp:
            with autocast(device_type=device.type):
                logits, loss = model(xb, yb)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, loss = model(xb, yb)
            loss.backward()
            optimizer.step()

    end_time = time.perf_counter()

    seconds_elapsed = end_time - start_time
    print(f'Training time: {seconds_elapsed:.3f} seconds, {seconds_elapsed / max_iters:.4f}s/step.')

    save_model_weights(model, iter)

    print(generate_tokens(model, 200))

def generate(weights_filename, token_count_to_generate):
    # note that this'll only work when the model defined above is the same as what was used to train and get the 
    # specified weights; since the Karpathy design hardcodes the config params (like number of heads, etc.) those have
    # to be set correctly at the top of the file in order for this to work and to avoid "Error(s) in loading state_dict 
    # for TransformerLanguageModel" errors; a better design for this kind of further expansion could be to persist/load 
    # the model defn in some fashion probably combined w/ parameterizing model creation rather than making the config 
    # params global vars as he does  
    model = TransformerLanguageModel().to(device)
    model.load_state_dict(torch.load(weights_filename, map_location=device))

    print(generate_tokens(model, token_count_to_generate))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('mode', choices=['train','generate'], nargs='?', default='train')
    parser.add_argument('weights_filename', nargs='?')
    parser.add_argument('--tokens', type=int, default=500)
    args = parser.parse_args()

    if args.mode == 'train':
        train()
    elif args.mode == 'generate':
        if not args.weights_filename:
            print('Error: need filename from which to load weights')
            exit(1)
        generate(args.weights_filename, args.tokens)
