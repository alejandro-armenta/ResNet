from PIL import Image

import torch

import numpy as np

from resnet import ResNet, ResidualBlock

from torchvision import datasets

from torchvision import transforms

from torchvision.models.feature_extraction import get_graph_node_names, create_feature_extractor

import matplotlib.pyplot as plt

from executorch.exir import to_edge_transform_and_lower

from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner

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

model.eval()

a = (image.unsqueeze(dim=0),)

#print(a)

et_program = to_edge_transform_and_lower(
    torch.export.export(model, a), partitioner=[XnnpackPartitioner()]
    ).to_executorch()


with open('model.pte', 'wb') as file:
    file.write(et_program.buffer)
    