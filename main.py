import re

import torch as pytorch
import torch.nn as nn
from torch.nn import functional as F
from tqdm import tqdm

from transformer.config import *
from transformer.model import GptImliModel
from utils.saveAndLoad import loadCheckpoint, saveCheckpoint

# read it in to inspect it
with open("./data/tolkien.txt", "r", encoding="utf-8") as f:
    text = f.read()

print("length of dataset in characters: ", len(text))

# NOTE: Character based tokenization:
# here are all the unique characters that occur in this text
chars = sorted(list(set(text)))
vocabSize = len(chars)
print("".join(chars))
print(vocabSize)

# create a mapping from characters to integers
stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [
    stoi[c] for c in s
]  # encoder: take a string, output a list of integers
decode = lambda l: "".join(
    [itos[i] for i in l]
)  # decoder: take a list of integers, output a string

# NOTE: Word based tokenization:
tokenized_raw_text = [item.strip() for item in re.split(r'([,.?_!"()\']|--|\s)', text) if item.strip()]
print("tokenized_raw_text:", len(tokenized_raw_text))
print("tokenized_raw_text first 20:", tokenized_raw_text[:20])

words = sorted(list(set(tokenized_raw_text)))
vocabSize = len(words)

stoi = {token:integer for integer, token in enumerate(words)}
itos = {integer:token for integer, token in enumerate(words)}

def encode(text):
    encoded_text = re.split(r'([,.?_!"()\']|--|\s)', text)
    encoded_text = [item.strip() for item in encoded_text if item.strip()]
    return [stoi[token] for token in encoded_text]

def decode(ids):
    decoded_text = " ".join([itos[i] for i in ids])
    # Replace spaces before the specified punctuations
    return re.sub(r'\s+([,.?!"()\'])', r'\1', decoded_text)

print(encode("what there"))
print(decode(encode("what there")))

data = pytorch.tensor(encode(text), dtype=pytorch.long)

print(data.shape, data.dtype)

nNum = int(0.9 * len(data))  # 90% of data set
trainData = data[:nNum]
validationData = data[nNum:]
trainData[: blockSize + 1]

pytorch.manual_seed(1337)


def getBatch(split: str):
    # gen a small batch of data of inputs x and targets y
    data = trainData if split == "train" else validationData
    ix = pytorch.randint(len(data) - blockSize, (batchSize,))

    x = pytorch.stack([data[i : i + blockSize] for i in ix])
    y = pytorch.stack([data[i + 1 : i + blockSize + 1] for i in ix])
    x = x.to(device)
    y = y.to(device)

    return x, y


@pytorch.no_grad()
def estimateLoss():
    out = {}
    model.eval()
    for split in ["train", "val"]:
        losses = pytorch.zeros(evalIters, device=device)
        for k in tqdm(range(evalIters)):
            X, Y = getBatch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out


# NOTE: model itself is here!
m = GptImliModel(vocabSize)
model = m.to(device)

optimizer = pytorch.optim.AdamW(model.parameters(), lr=learningRate)

# NOTE: if need to load pretrained weights:
isLoadModel = False # False 
saveAndLoadModelBaseName = 'Gptimli-WBRD-'
loadModelPath = saveAndLoadModelBaseName + '2000.pth'
if isLoadModel == True:
    model, optimizer, lastEpoch  = loadCheckpoint(model, optimizer, loadModelPath)
    print("Model " + loadModelPath + " loaded!")
else:
    lastEpoch = 0
    print("no model loaded; creating a new one: " + saveAndLoadModelBaseName)

# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters()) / 1e6, "M parameters")

# NOTE: Training:
def train():
    lastTrainedEpoch = lastEpoch
    for epoch in tqdm(range(lastTrainedEpoch, epochs + lastTrainedEpoch + 1)):
        # every once in a while evaluate the loss on train and val sets
        if epoch % 500 == 0:
            if epoch != lastTrainedEpoch and epoch != epochs + lastTrainedEpoch:
                saveModelPath = saveAndLoadModelBaseName + str(lastTrainedEpoch) + '.pth'
                saveCheckpoint(model, optimizer, saveModelPath, lastTrainedEpoch)
                print("Model " + saveModelPath + " saved!")

            losses = estimateLoss()
            print(
                f"Epoch {epoch}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )

        # sample a batch of data
        xb, yb = getBatch("train")

        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        lastTrainedEpoch = epoch

    saveModelPath = saveAndLoadModelBaseName + str(lastTrainedEpoch) + '.pth'
    saveCheckpoint(model, optimizer, saveModelPath, lastTrainedEpoch)
    print("Model " + saveModelPath + " saved!")

    print("Last loss:")
    print(loss.item())

train()

print("MODEL_OUTPUT:")
#context = pytorch.zeros((1, 1), dtype=pytorch.long, device=device)
context = pytorch.tensor(encode('where do we go?'), dtype=pytorch.long, device=device).unsqueeze(0)
print(decode(model.generate(idx=context, maxNewTokens=500)[0].tolist()))
