import torch as pytorch

# Meta params:
batchSize = 64 # 64  # nEmbd * nHeadsCount
blockSize = 256 # 264  # chunk size
nHeadsCount = 8 # 6
nLayersCount = 8 # 6
nEmbd = batchSize * nHeadsCount  # batchSize = nEmbd / nHeadsCount

learningRate = 5e-4
dropout = 0.2

epochs = 500
evalIters = 200

device = pytorch.device("mps") if pytorch.backends.mps.is_available() else "cpu"

if pytorch.backends.mps.is_available():
    #device = pytorch.device("mps")
    print("MPS device detected: " + str(device))
else:
    print("MPS device not found.")
