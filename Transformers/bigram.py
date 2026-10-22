# The Bigram model we built

import torch   
import torch.nn as nn
from torch.nn import functional as F


# hyperparameters 
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 8 # what is the maximum context length for predictions?
max_iters = 3000
eval_interval = 300
learning_rate = 1e-2
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200


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

# simple bigram model

class BigramLanguageModel(nn.Module): # BigamLanaguageModel is a subclass of nn.Module

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, vocab_size)  # creates a wrapper aroung tensor of given size

    def forward(self, idx, targets=None): # method over-ridding ( defining how forward pass will happen)

        # idx ad targets are both (B, T) tensor of intergets
        logits = self.token_embedding_table(idx) # (B, T, C) (batch = batch_size, time = block_size, channels = vocab_size)

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
            # get the predictions
            logits, loss = self(idx) # perfroms __call__ on model
            # focus only on last time step
            logits = logits[:, -1, :] # becomes (B, C) : to keep it bigram (only look at just previous not the whole history)
            # apply softmax to get probs
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled idx to the running sequence
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

model = BigramLanguageModel(vocab_size)
m = model.to(device)

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # what we used till now = SGT Schotastic gradient descent, now AdamW

print(torch.__version__)
print(torch.version.cuda)
print(torch.cuda.is_available())
print(f"running on: {device}\n-----------")

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

context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))