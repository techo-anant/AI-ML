# The Bigram model we built, with decoder-only transformers (self-attention)

import torch 
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters 
batch_size = 64 # how many independent sequences will we process in parallel?
block_size = 256 # what is the maximum context length for predictions?
max_iters = 5000
eval_interval = 500
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384

n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# wget https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# creating vocabulary/tokens
chars = sorted(list(set(text)))
vocab_size = len(chars)

# create a mapping from characters to integers
stoi = {ch:i for i, ch in enumerate(chars)}
itos = {i:ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l])  # decoder: take a list of int, output a string

# train and test splits (tr: 90%, test:10%)
data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

# data loading

# how we train a transformer:
    # To compute heavy to do it all at once, so we sample random chunks and train iterativily on them
    # max len of chunks ( block_size )

# how transformers work? unlike RNNs and CNNs, transformers train the block_size in parallel not iterative
# So, a block size of 8 means, training 8 threads starting with a pair (1 prev predicts next) then (2 prev predicts
# next) ... total 8

def get_batch(split):
    # generate a small batch of data of inputs x and target y
    data = train_data if split =='train' else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i: i+block_size] for i in ix])
    y = torch.stack([data[i+1: i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):
    """one head of self-attention : Scaled Dot-Porduct Attention from paper (Attention is all you need)"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size))) # now accessible as self.tril, but won't be in model.parameters() (so not trained)

        self.dropout = nn.Dropout(dropout)  # to prevent over-fitting, while scaling up

    def forward(self, x): # decoder block

        B, T, C = x.shape
        k = self.key(x) # (B, T, C) as head_size = n_embd
        q = self.query(x) # (B, T, C)

        # compute attention score ("affinities")
        wei = q @ k.transpose(-2, -1) * C**-0.5 # (B, T, C) @ (B, C, T) --> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
        wei = self.dropout(wei)

        # perform the weighted aggregation of the values
        v = self.value(x) # (B, T, C)
        out = wei @ v # (B, T, T) @ (B, T, C) --> (B, T, C)
        return out

class MultiHeadAttention(nn.Module):
    """ multiple head of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(num_heads * head_size, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    """ a simple linear layer followed by a non-linearity : acct to the paper ( a MLP )"""

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            # multipling per Attention is all you need paper
            nn.Linear(n_embd, 4 * n_embd),  # linear
            nn.ReLU(), # non-linearity
            nn.Linear(4 * n_embd, n_embd),  # linear projection layer back into residual pathway
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    """ Transformer block: communication followed by computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd//n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):  # using skip/residual connection : fork off and so some compute, then come back
        x = x + self.sa(self.ln1(x))  # little diff then the paper: we are doing per_norm not post_norm (so normalize the input instead of the output)
        x = x + self.ffwd(self.ln2(x))
        return x

class BigramLanguageModel(nn.Module): # BigamLanaguageModel is a subclass of nn.Module

    def __init__(self):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)  # creates a wrapper aroung tensor of given size
        self.position_embedding_table = nn.Embedding(block_size, n_embd) # embedding for the positions in the context of transfromer

        # self.sa_heads = MultiHeadAttention(num_heads, n_embd//num_heads) # i.e. 4 heads of 8-dimensional self-attention
        # self.ffwd = FeedForward(n_embd, n_heads)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd) # final layer norm
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None): # method over-ridding ( defining how forward pass will happen)

        B, T = idx.shape
        # idx ad targets are both (B, T) tensor of intergets
        tok_emb = self.token_embedding_table(idx) # (B, T, C) (batch = batch_size, time = block_size, channels = n_embd)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        x = tok_emb + pos_emb # (B, T, C)
        # what deos x hold - the tokens identity + the position at which token occurs

        # x = self.sa_heads(x) # apply one head of self-attention. (B, T, C)
        # x = self.ffwd(x) # (B, T, C)
        x = self.blocks(x) # (B, T, C)
        x = self.ln_f(x) # (B, T, C)
        logits = self.lm_head(x) # (B, T, vocab_size)


        if targets is None:
            loss = None
        else:
            # how to pass logits to F.cross_entropy? we have (B, T, C), F.cross_entropy expects C to be 2nd dim
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # corp idx to the last block_size tokens
            idx_cond = idx[:, -block_size:]  # as positinal embeddings can only take up to blocksize (any more and we break the context limit)
            # get the predictions
            logits, loss = self(idx_cond) # perfroms __call__ on model
            # focus only on last time step
            logits = logits[:, -1, :] # becomes (B, C) : to keep it bigram (only look at just previous not the whole history)
            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled idx to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel()
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # what we used till now = SGT Schotastic gradient descent, now AdamW

print("PyTorch version: ", torch.__version__)
print("PyTorch with CUDA available: ", torch.version.cuda)
print("CUDA available: ", torch.cuda.is_available())
print(f"Running on: {device}\n-----------")
print(sum(p.numel() for p in model.parameters()) / 1e6, 'M parameters')

for iter in range(max_iters):

    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of date
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb) # forward pass
    optimizer.zero_grad(set_to_none=True) # gards to None
    loss.backward() # backward pass
    optimizer.step() # update the grads

# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# result = decode(m.generate(context, max_new_tokens=20000)[0].tolist())

model.eval()   # turn dropout off
with torch.no_grad():
    context = torch.zeros((1, 1), dtype=torch.long, device=device)
    result = decode(m.generate(context, max_new_tokens=20000)[0].tolist())

with open('result_v2.txt', 'w', encoding='utf8') as f:
    f.write(result)

"""
result: (GPU necessary: if not, lower the hyperparameters)

PyTorch version: 2.14.1+cu130
PyTorch with CUDA available: 13.0
CUDA available: True
Running on: running on: cuda
-----------

10.79M parameters

step 0: train loss 4.2849, val loss 4.2827
step 500: train loss 1.9969, val loss 2.0803
step 1000: train loss 1.5900, val loss 1.7641
step 1500: train loss 1.4325, val loss 1.6359
step 2000: train loss 1.3441, val loss 1.5721
step 2500: train loss 1.2747, val loss 1.5264
step 3000: train loss 1.2271, val loss 1.5074
step 3500: train loss 1.1823, val loss 1.4857
step 4000: train loss 1.1447, val loss 1.4863
step 4500: train loss 1.1092, val loss 1.4875

Maketching him right foe
And will bear and read in hand, answer you,
If nothing perhapping me, as she hath done
Forberred him enought follow'd. But do you find
Hermione of his true-current of his hoyalt grieves
With that waste mean royal things,
You heard parts the shames, and her pursants befort
And send further; therefoot were praised
No wake and by at cies the poison
Air, whose that shall curse and so bloves,--
That please they to clame for thus,--corth, they are share,
That plehes in our wom
"""