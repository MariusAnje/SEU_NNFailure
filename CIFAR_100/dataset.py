import torch
import torchvision

a = torchvision.datasets.CIFAR100("./data", train=True, transform=None, target_transform=None, download=True)

a = torchvision.datasets.CIFAR100("./data", train=False, transform=None, target_transform=None, download=True)
