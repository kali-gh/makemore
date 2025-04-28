import torch
import torch.nn as nn

from torch.nn import functional as F

torch.manual_seed(1337)

# This code base is mostly a working copy from Andrej Karpathy's makemore series.
SMALL = True

if SMALL:
    batch_size = 32
    block_size = 8
    max_iters = 5000
    eval_interval = 500
    learning_rate = 1e-2
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 32
    n_head = 4
    n_layer = 3
    dropout = 0.2
else:
    batch_size = 64
    block_size = 256
    max_iters = 5000
    eval_interval = 500
    learning_rate = 3e-4
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    eval_iters = 200
    n_embd = 384
    n_head = 6
    n_layer = 6
    dropout = 0.2

print(f"device : {device}")

with open('input.txt','r') as f:
    text = f.read()

# set up
chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch : i for i, ch in enumerate(chars)}
itos = {i  : ch for i, ch in enumerate(chars)}

encode = lambda s : [stoi[c] for c in s]
decode = lambda l : ''.join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9*len(data))
train_data = data[:n]
val_data = data[n:]

def get_batch(split):
    """
    Get a batch of data given a split

    batch has 
        x random sets of data of length block_size
        y offset by 1 from x, targets of x

    Args:
        split (str): the batch to use
    Returns:
        Tuple:
            x : x data
            y : y data
     """
     
    data = train_data if split == 'train' else val_data

    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: (i+block_size)] for i in ix])

    y = torch.stack([data[(i+1):(i+block_size+1)] for i in ix])

    x, y = x.to(device), y.to(device)

    return x, y

    
class FeedForward(nn.Module):
    
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4*n_embd),
            nn.ReLU(),
            nn.Linear(4*n_embd, n_embd),
            nn.Dropout(dropout),
        )
    
    def forward(self, x):
        return self.net(x)
    
    
class Block(nn.Module):
    
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        
        self.sa = MultiHeadAttention(n_head, head_size)
        
        self.ffwd = FeedForward(n_embd=n_embd)
        
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)
        
    def forward(self, x):
        # pre sa / ffwd layer normalization.
        # the ln layers are normalizing each token of data to have zero mean and unit variance
        # (across the 32 dimensions that represent it)
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln1(x))
        
        return x


class Head(nn.Module):
    # decoder block. uses lower triangular masking
    # distinguish vs. encoder block : allow all nodes to talk to each other
    # cross attention : keys and values come from another encoder block allows for cross context learning
    #   (e.g. machine translation)
    # sqrt(head size = q), important due to weight matrix variance.
    #   if variance too high, softmax becomes sharp focusing on specific one hot values.
    #    sqrt allows to control the variance

    def __init__(self, head_size):
        super().__init__()
        
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        """
        Forward pass through the network for the head.

        Args:
            x: the tokenized and embedded characters

        Returns:
            x : x after a forward pass through the head

        """
        B,T,C = x.shape

        k = self.key(x) # (B, T, C)
        q = self.query(x) # (B, T, C)

        wei = q @ k.transpose(-2, -1) * C **-0.5 # (B, T, C) x (B, C, T) -> (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T]==0, float('-inf')) # this masks the upper part
        wei = F.softmax(wei, dim=-1) # B,T,T
        wei = self.dropout(wei)
        v = self.value(x) # (B,T,C)
        out = wei @ v # (B,T,T) @ (B,T,C) -> (B,T,C) v is a collapsed representation of x, with learned weights.

        return out

class MultiHeadAttention(nn.Module):
    """
    Multi head attention block using multiple heads of given head size
    """
    
    def __init__(self, num_heads : int, head_size : int):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj  = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out
    

class BigramLanguageModel(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        
        n_head = 4
        
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
            
        self.ln_f =  nn.LayerNorm(n_embd)
        
        self.lm_head = nn.Linear(n_embd, vocab_size)
    
    def forward(
            self,
            idx,
            targets=None):

        B, T = idx.shape
        
        tok_emb = self.token_embedding_table(idx) # B,T,C
        
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, C)
        
        x = tok_emb + pos_emb # (B,T,C)
        
        x = self.blocks(x)
        x = self.ln_f(x)
        
        logits = self.lm_head(x)
        
        B, T, C = logits.shape
        
        if targets is None:
            loss = None
        else:
            logits = logits.view(B*T, C)
            
            targets = targets.view(B*T)
            
            loss = F.cross_entropy(logits, targets)
            
        return logits, loss
    
    def generate(self, idx, max_new_tokens):
        
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            
            logits, loss = self(idx_cond)
            
            logits = logits[:, -1, :]
            
            probs = F.softmax(logits, dim=-1)
            
            idx_next = torch.multinomial(probs, num_samples=1)
            
            idx = torch.cat((idx, idx_next), dim=1)
        
        return idx
    


def train():
    model = BigramLanguageModel()
    m = model.to(device)

    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    @torch.no_grad()
    def estimate_loss():
        out = {}
        model.eval()
        
        for split in ['train', 'val']:
            losses = torch.zeros(eval_iters)
            for k in range(eval_iters):
                X,Y = get_batch(split)
                logits, loss = model(X,Y)
                losses[k] = loss.item()
                
            out[split] = losses.mean()
            
        model.train()
        return out
        
    for iter in range(max_iters):
        
        if iter % eval_interval == 0:
            losses = estimate_loss()
            print(f"iter : {iter} loss = {losses['train']:.4f}, val loss = {losses['val']:.4f}")
            
        xb, yb = get_batch('train')
        
        logits, loss = m(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        
        loss.backward()
        
        optimizer.step()
        
    idx = torch.zeros((1,1), dtype=torch.long, device=device)
    print(decode(m.generate(idx, max_new_tokens=300)[0].tolist()))

    torch.save(m.state_dict(), 'model.pt')


if __name__ == '__main__':
    train()
    
    
"""
PAULINA:
Are you may, riches.

Third of you, sister, I know not, Tranio.
Would you must, your lady your block trimes!
I Romeo, once your briches; we'll cherist
You hear me all.

CORIOLANUS:
Why, may beget the parts?

COMINIUS:
By the Lord Oxford; you knock haste, say't
How much eased, be gar of Mar
"""