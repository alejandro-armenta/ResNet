from PIL import Image

import torch

import numpy as np

from resnet import ResNet, ResidualBlock

from torchvision import datasets

from torchvision import transforms

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

import matplotlib.pyplot as plt


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

print(device)

normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])

t = transforms.Compose(
    [
        #este si puede ser pil
        transforms.Resize(size=(224,224)),
        transforms.ToTensor(),
        normalize
    ]
)

test_dataset = datasets.CIFAR10(root='./data', download=True, train=False, transform=t)

image,label = test_dataset[0]

other = image.numpy()
other = other.transpose(1,2,0)

#print(label)

chech = torch.load('best_model.pt', map_location=device)

model = ResNet(ResidualBlock, [3,4,6,3])

model.load_state_dict(chech['model_state_dict'])

print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

model_empty = ResNet(ResidualBlock, [3,4,6,3])

def create_features(model, image, out):
    
    train_mode, eval_mode = get_graph_node_names(model)

    #print(train_mode)

    return_nodes = {

        'conv1.2':'fm0',
        
        'maxpool':'fm1',

        'layer0.2.relu':'fm2',

        'layer1.2.relu':'fm3',

        'layer2.2.relu':'fm4',

        'layer3.2.relu':'fm5',

        'avgpool':'fm6'

        }

    other_model = create_feature_extractor(model=model, return_nodes=return_nodes)

    image = image.unsqueeze(dim=0)

    #print(image.shape)

    output = other_model(image)

    feature_list = {}
    for k,v in return_nodes.items():
        feature_list[k] = output[v].squeeze(dim=0)


    elements = len(list(feature_list))+1
    fig, axes = plt.subplots(1, elements, figsize=(elements*5, 5))

    axes[0].imshow(other)

    for i, (k,v) in enumerate(feature_list.items(), start=1):
        axes[i].imshow(v[0].detach().numpy())
        axes[i].set_title(k)
        #axes[i].axis('off')

    plt.tight_layout()
    #plt.show()


    pred = model(image)

    #print(torch.max(pred.data,1))
    #print(torch.max(pred.squeeze(dim=0).shape)

    a = output['fm6'].squeeze(dim=0)

    b = a.reshape(32,-1)

    plt.imshow(b.detach().numpy())

    plt.savefig(out)


create_features(model=model, image=image, out='real_features.png')

create_features(model=model_empty, image=image, out='empty_features.png')
