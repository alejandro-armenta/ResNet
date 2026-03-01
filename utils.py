
import numpy as np

from torchvision import datasets

from torchvision import transforms

from torch.utils.data.sampler import SubsetRandomSampler

from torch.utils.data import DataLoader


def data_loader(data_dir, batch_size, test=False, shuffle=True, valid_size=0.1):

    normalize = transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],std=[0.2023, 0.1994, 0.2010])

    t = transforms.Compose(
        [
            #este si puede ser pil
            transforms.Resize(size=(224,224)),
            transforms.ToTensor(),
            normalize
        ]

    )

    if test:
        test_dataset = datasets.CIFAR10(root=data_dir, download=True, train=False, transform=t)
        #print(test_dataset)
        data_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=shuffle)
        return data_loader

    train_dataset = datasets.CIFAR10(root=data_dir, download=True, train=True, transform=t)
    valid_dataset = datasets.CIFAR10(root=data_dir, download=True, train=True, transform=t)

    num_train = len(train_dataset)

    indices = list(range(num_train))

    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(42)
        np.random.shuffle(indices)

    train_indices = indices[split:]
    val_indices = indices[:split]

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    #print(train_dataset is valid_dataset)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, valid_loader


#test_loader = data_loader('./data', batch_size=64, test=True)