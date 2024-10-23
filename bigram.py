import torch as pytorch
import torch.nn as nn
from torch.nn import functional as F

# CONSTS:
batchSize = 64 # nEmbd * nHeadsCount
blockSize = 264 # chunk size
nEmbd = 384 # batchSize / nHeadsCount
nHeadsCount = 6
nLayersCount = 6
learningRate = 3e-4
dropout = 0.2
epochs = 5000
evalIters = 200
device = 'cpu'

batchSize = 16 # 
blockSize = 32 # chunk size
nEmbd = 64
nHeadsCount = 4
nLayersCount = 4
learningRate = 1e-3

if pytorch.backends.mps.is_available():
    device = pytorch.device("mps")
    print ("MPS device detected.")
else:
    print ("MPS device not found.")

# read it in to inspect it
with open('input.txt', 'r', encoding='utf-8') as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocabSize = len(chars)
print(''.join(chars))
print(vocabSize)

# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string

print(encode("hii there"))
print(decode(encode("hii there")))

data = pytorch.tensor(encode(text), dtype=pytorch.long)

print(data.shape, data.dtype)

nNum = int(0.9*len(data)) # 90% of data set
trainData = data[:nNum]
validationData = data[nNum:]
trainData[:blockSize+1]

pytorch.manual_seed(1337)

def getBatch(split: str):
  # gen a small batch of data of inputs x and targets y
  data = trainData if split == 'train' else validationData
  ix = pytorch.randint(len(data) - blockSize, (batchSize, ))

  x = pytorch.stack([data[i:i+blockSize] for i in ix])
  y = pytorch.stack([data[i+1:i+blockSize+1] for i in ix])
  x = x.to(device)
  y = y.to(device)

  return x, y

@pytorch.no_grad()
def estimateLoss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = pytorch.zeros(evalIters, device=device)
        for k in range(evalIters):
            X, Y = getBatch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# xb, yb = getBatch('train')

# NOTE: FeedForward module:
class FeedForward(nn.Module):
  def __init__(self, n) -> None:
    super().__init__()
    self.net = nn.Sequential(
        nn.Linear(n, 4 * n),
        nn.ReLU(),
        nn.Linear(4 * n, n),
        nn.Dropout(dropout),
    )

  def forward(self, x):
    return self.net(x)

# NOTE: BatchLayerNormalization1D module:
# the same thing as a pytorch.nn.LayerNorm() module
class BatchLayerNormalization1D():
  def __init__(self, dim, esp=1e-5) -> None:
    self.esp = esp
    self.gamma = pytorch.ones(dim)
    self.beta = pytorch.ones(dim)

  def __call__(self, x):
    #calc the forward pass
    xMean = x.mean(1, keepdim=True) # batch mean
    xVar = x.var(1, keepdim=True) # batch variance
    xHat = (x - xMean) / pytorch.sqrt(xVar + self.esp) # normailize to unit variance
    out = self.gamma * xHat + self.beta
    return out

  def parameters(self):
    return [self.gamma, self.beta]

# NOTE: SelfAttentionHead module
class SelfAttentionHead(nn.Module):
  # one head of self-attention:

  def __init__(self, headSize) -> None:
    super().__init__()
    self.key = nn.Linear(nEmbd, headSize, bias=False)
    self.query = nn.Linear(nEmbd, headSize, bias=False)
    self.value = nn.Linear(nEmbd, headSize, bias=False)
    self.register_buffer('tril', pytorch.tril(pytorch.ones(blockSize, blockSize)))
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    B,T,C = x.shape
    k = self.key(x)
    q = self.query(x)

    #compute attention scores:
    weight = q @ k.transpose(-2, -1) * C**-0.5
    weight = weight.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
    weight = F.softmax(weight, dim=-1)
    weight = self.dropout(weight)

    v = self.value(x)
    out = weight @ v
    return out

# NOTE: MultiHeadSelfAttention module
class MultiHeadSelfAttention(nn.Module):
  def __init__(self, headSize, numOfHeads) -> None:
    super().__init__()
    self.heads = nn.ModuleList([SelfAttentionHead(headSize) for _ in range(numOfHeads)])
    self.projection = nn.Linear(nEmbd, nEmbd)
    self.dropout = nn.Dropout(dropout)

  def forward(self, x):
    out = pytorch.cat([head(x) for head in self.heads], dim=-1)
    out = self.projection(out)
    out = self.dropout(out)
    return out

# NOTE: Block module:
class Block(nn.Module):
  # transformer block: communication flollows the computation:
  def __init__(self, numOfHeads, numOfEmbd) -> None:
    super().__init__()
    headSize = numOfEmbd // numOfHeads
    self.selfAttention = MultiHeadSelfAttention(numOfHeads, headSize)
    self.feedForward = FeedForward(numOfEmbd)
    self.layerNorm1 = nn.LayerNorm(numOfEmbd) # same as BatchLayerNormalization1D class at the top
    self.layerNorm2 = nn.LayerNorm(numOfEmbd)

  def forward(self, x):
    x = x + self.selfAttention(self.layerNorm1(x))
    x = x + self.feedForward(self.layerNorm2(x))
    return x

# NOTE: BigramLanguageModel itself:
class BigramLanguageModel(nn.Module):
  def __init__(self) -> None:
     super().__init__()

     self.tokenEmbeddingTable = nn.Embedding(vocabSize, nEmbd)
     self.positionEmbeddingTable = nn.Embedding(blockSize, nEmbd)
     self.blocks = nn.Sequential(*[Block(numOfHeads=nHeadsCount, numOfEmbd=nEmbd) for _ in range(nLayersCount)])
     self.layerNormFinal = nn.LayerNorm(nEmbd)
     self.lmHead = nn.Linear(nEmbd, vocabSize)

  def forward(self, idx, targets=None):
    B, T = idx.shape

    tokenEmbed = self.tokenEmbeddingTable(idx) # (B, T, C)
    positionEmbd = self.positionEmbeddingTable(pytorch.arange(T, device=device))
    x = tokenEmbed + positionEmbd
    x = self.blocks(x)
    x = self.layerNormFinal(x)
    logits = self.lmHead(x)

    if targets is None:
      loss = None
    else:
      B, T, C = logits.shape
      logits = logits.view(B*T, C)
      targets = targets.view(B*T)

      loss = F.cross_entropy(logits, targets)

    return logits, loss

  def generate(self, idx, maxNewTokens):
    # idx is (B, T) array of indices in the current context
    for _ in range(maxNewTokens):
      # crop idx to the last blockSize tokens:
      idx_cond = idx[:, -blockSize:]
      # get the predictions
      logits, loss = self(idx=idx_cond)
      # focus only on the last time step
      logits = logits[:, -1, :] # becomes (B, C)
      # apply softmax to get probabilities
      probs = F.softmax(logits, dim=-1) # (B, C)
      # sample from the distribution
      idxNext = pytorch.multinomial(probs, num_samples=1) # (B, 1)
      # append sampled index to the running sequence
      idx = pytorch.cat((idx, idxNext), dim=1) # (B, T+1)
    return idx

m = BigramLanguageModel()
model = m.to(device) 
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

optimizer = pytorch.optim.AdamW(model.parameters(), lr=1e-3)

for epoch in range(epochs + 1):
  # every once in a while evaluate the loss on train and val sets
  if epoch % 500 == 0 or epoch == epochs - 1:
      losses = estimateLoss()
      print(f"step {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

  # sample a batch of data
  xb, yb = getBatch('train')

  logits, loss = model(xb, yb)
  optimizer.zero_grad(set_to_none=True)
  loss.backward()
  optimizer.step()
  if epoch % 1000 == 0:
    print('epoch: ' + str(epoch) + '; loss: ' + str(loss.item()))

print('Last loss:')
print(loss.item())

print('MODEL_OUTPUT:')
context = pytorch.zeros((1, 1), dtype=pytorch.long, device=device)
print(decode(model.generate(idx=context, maxNewTokens=500)[0].tolist()))
