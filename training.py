from tqdm import tqdm

import torch

import torch.nn as nn

from resnet import ResNet, ResidualBlock

import gc

from utils import data_loader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

torch.cuda.empty_cache()
gc.collect()

print(device)

train_loader, valid_loader = data_loader('./data', batch_size=64)
test_loader = data_loader('./data', batch_size=64, test=True)

model = ResNet(ResidualBlock, [3,4,6,3]).to(device=device)

criterion = nn.CrossEntropyLoss()

num_epochs = 20
learnin_rate = 0.01

#ay un weight decay en sgd
optimizer = torch.optim.SGD(model.parameters(), lr=learnin_rate, weight_decay=0.001, momentum=0.9)

print(len(train_loader) * num_epochs)

best_accuracy = 0

for epoch in range(num_epochs):

    loader = tqdm(train_loader, desc='training epoch')
    
    
    for images, labels in loader:

        images = images.to(device)
        labels = labels.to(device)

        pred = model(images)

        #print(pred.shape)
        loss = criterion(pred, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loader.set_postfix(loss=f"{loss.item():.4f}")

        del images, labels, pred
        gc.collect()
        torch.cuda.empty_cache()
    

    print ('Epoch [{}/{}], Loss: {:.4f}'.format(epoch+1, num_epochs, loss.item()))
    

    with torch.no_grad():
        
        correct = 0
        total = 0

        loader = tqdm(valid_loader, desc='valid epoch')

        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            #print(outputs.shape)

            _, predicted = torch.max(outputs.data, 1)
            
            #print("predicted shape", predicted.shape)
            #print("labels shape", labels.shape)

            total += labels.size(0)
            
            correct += (predicted == labels).sum().item()

            del images, labels, outputs
            gc.collect()
            torch.cuda.empty_cache()
        
        accuracy = 100 * correct / total
        print('Accuracy of the network on the {} validation images: {} %'.format(5000, accuracy))

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save({
                'epoch' : epoch,
                'model_state_dict' : model.state_dict(),
                'optimizer_state_dict' : optimizer.state_dict(),
                'accuracy':accuracy
            }, f='best_model.pt')


        