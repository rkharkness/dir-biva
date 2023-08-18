import numpy as np
import torch
from biva.datasets import get_binmnist_datasets, get_cifar10_datasets, get_numpy2d_dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms

def get_dataloaders(data, path, bs = 64, num_workers = 4):
    if data == 'binmnist':
        train_dataset, valid_dataset, test_dataset = get_binmnist_datasets(path)

    elif data == 'cifar10':
        from transforms import Lambda

        transform = Lambda(lambda x: x * 2 - 1)
        train_dataset, valid_dataset, test_dataset = get_cifar10_datasets(path, transform=transform)

    elif data == 'mood':
        train_loader = get_numpy2d_dataset(path, batch_size=bs)
        valid_loader = get_numpy2d_dataset(path, batch_size=bs, mode="val")
        test_loader = valid_loader

    elif data == 'luna16':
        raise NotImplementedError
    
    else:
        raise NotImplementedError

    if data != 'mood':
        train_loader = DataLoader(train_dataset, batch_size=bs, shuffle=True, pin_memory=False, num_workers=num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=2 * bs, shuffle=True, pin_memory=False,
                                num_workers=num_workers)
        test_loader = DataLoader(test_dataset, batch_size=2 * bs, shuffle=True, pin_memory=False,
                                num_workers=num_workers)
        tensor_shp = (-1, *train_dataset[0].shape)
    
    else:
        batch = next(iter(train_loader))
        tensor_shp = (-1, batch[0].shape)
    
    return train_loader, valid_loader, test_loader, tensor_shp