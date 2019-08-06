import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as dsets
from torch.autograd import Variable


# Getting datasets
train_dset = dsets.MNIST(root='./data',
						 train=True,
						 transform=transforms.ToTensor(),
						 download=True)
test_dset = dsets.MNIST(root='./data',)						 