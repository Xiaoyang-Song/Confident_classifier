# original code is from https://github.com/aaron-xichen/pytorch-playground
# modified by Kimin Lee
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import numpy.random as nr
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler

def getSVHN(batch_size, img_size=32, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'svhn-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building SVHN data loader with {} workers".format(num_workers))

    def target_transform(target):
        new_target = target - 1
        if new_target == -1:
            new_target = 9
        return new_target

    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='train', download=True,
                transform=transforms.Compose([
                    transforms.Scale(img_size),
                    transforms.ToTensor(),
                ]),
                target_transform=target_transform,
            ),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)

    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.SVHN(
                root=data_root, split='test', download=True,
                transform=transforms.Compose([
                    transforms.Scale(img_size),
                    transforms.ToTensor(),
                ]),
                target_transform=target_transform
            ),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def SVHN(bsz_tri, bsz_val, shuffle=True):
    
    # Ground truth mean & std
    # mean = torch.tensor([111.6095, 113.1610, 120.5650])
    # std = torch.tensor([50.4977, 51.2590, 50.2442])
    normalizer = transforms.Normalize(mean=[x/255.0 for x in [111.6095, 113.1610, 120.5650]],
                                      std=[x/255.0 for x in [50.4977, 51.2590, 50.2442]])
    transform = transforms.Compose([transforms.ToTensor(), normalizer])

    # Load dataset & Loader
    train_dataset = datasets.SVHN('./Datasets/SVHN', split='train',
                                  download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=bsz_tri, shuffle=shuffle)
    val_dataset = datasets.SVHN('./Datasets/SVHN', split='test', download=True,
                                transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=bsz_val, shuffle=shuffle)

    return train_dataset, val_dataset, train_loader, val_loader

def getCIFAR10(batch_size, img_size=32, data_root='/tmp/public_dataset/pytorch', train=True, val=True, **kwargs):
    data_root = os.path.expanduser(os.path.join(data_root, 'cifar10-data'))
    num_workers = kwargs.setdefault('num_workers', 1)
    kwargs.pop('input_size', None)
    print("Building CIFAR-10 data loader with {} workers".format(num_workers))
    ds = []
    if train:
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=True, download=True,
                transform=transforms.Compose([
                    transforms.Scale(img_size),
                    transforms.ToTensor(),
                ])),
            batch_size=batch_size, shuffle=True, **kwargs)
        ds.append(train_loader)
    if val:
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10(
                root=data_root, train=False, download=True,
                transform=transforms.Compose([
                    transforms.Scale(img_size),
                    transforms.ToTensor(),
                ])),
            batch_size=batch_size, shuffle=False, **kwargs)
        ds.append(test_loader)
    ds = ds[0] if len(ds) == 1 else ds
    return ds

def CIFAR10(batch_size, test_batch_size):
    
    # Ground truth mean & std:
    # mean = torch.tensor([125.3072, 122.9505, 113.8654])
    # std = torch.tensor([62.9932, 62.0887, 66.7049])
    normalizer = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                      std=[x/255.0 for x in [63.0, 62.1, 66.7]])
    transform = transforms.Compose([transforms.ToTensor(), normalizer])
    train_dataset = datasets.CIFAR10('./Datasets/CIFAR-10', train=True,
                                     download=True, transform=transform)
    # ic(len(train_dataset))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True)
    val_dataset = datasets.CIFAR10('./Datasets/CIFAR-10', train=False, download=True,
                                   transform=transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=test_batch_size, shuffle=True)

    return train_dataset, val_dataset, train_loader, val_loader

def getTargetDataSet(data_type, batch_size, imageSize, dataroot):
    if data_type == 'cifar10':
        train_loader, test_loader = getCIFAR10(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        train_loader, test_loader = getSVHN(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=1)

    return train_loader, test_loader

def getNonTargetDataSet(data_type, batch_size, imageSize, dataroot):
    if data_type == 'cifar10':
        _, test_loader = getCIFAR10(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=1)
    elif data_type == 'svhn':
        _, test_loader = getSVHN(batch_size=batch_size, img_size=imageSize, data_root=dataroot, num_workers=1)
    elif data_type == 'imagenet':
        testsetout = datasets.ImageFolder(dataroot+"/Imagenet_resize", transform=transforms.Compose([transforms.Scale(imageSize),transforms.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)
    elif data_type == 'lsun':
        testsetout = datasets.ImageFolder(dataroot+"/LSUN_resize", transform=transforms.Compose([transforms.Scale(imageSize),transforms.ToTensor()]))
        test_loader = torch.utils.data.DataLoader(testsetout, batch_size=batch_size, shuffle=False, num_workers=1)

    return test_loader
