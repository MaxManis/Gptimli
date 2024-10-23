
m = GptImliModel(vocabSize)
model = m.to(device)

loadModelPath = 'Gptimli-deep-500.pth'
lastEpoch = 0
model, optimizer, lastEpoch  = loadCheckpoint(model, optimizer, loadModelPath)

print("MODEL_OUTPUT:")
#context = pytorch.zeros((1, 1), dtype=pytorch.long, device=device)
context = pytorch.tensor(encode('where do we go?'), dtype=pytorch.long, device=device).unsqueeze(0)
print(decode(model.generate(idx=context, maxNewTokens=500)[0].tolist()))
