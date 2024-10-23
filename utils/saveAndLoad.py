import torch as pytorch


def saveCheckpoint(model, optimizer, savePath, epoch):
    pytorch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch
    }, savePath)


def loadCheckpoint(model, optimizer, loadPath):
    checkpoint = pytorch.load(loadPath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    
    return model, optimizer, epoch
