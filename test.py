import torch

from utils import data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_loader = data_loader('./data', batch_size=64, test=True)

with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(device)
        labels = labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        del images, labels, outputs