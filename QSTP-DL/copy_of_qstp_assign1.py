# -*- coding: utf-8 -*-
"""Copy of qstp_assign1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1W8NBktriYc36O5w3RhXsL3dOtRZxe9dP
"""

# Imported the necessary libraries 

import numpy as np
import os
import torch
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
from torch import optim
from google.colab import drive

# %matplotlib inline

# I have loaded the dataset for you so you don't have to worry about this part :) 

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


# Print a sample image 
# Use our imshow function and iterators
# (Approx 5 - 10 lines of code)

### START CODE ### 
imshow(trainset[0][0])
###  END CODE  ###

input_size = 3072
hidden_sizes = [512 , 128]
output_size = 10

# Now make a model use the sizes of the hidden layers as given above 
# NOTE: DO NOT MAKE A CNN 
# You can use any of the 2 methods to define your model i.e. 
# 1) make a class and 
# 2) Define using nn.Sequential 
# Don't forget to add activation after the layer ( for non linear activation you can use relu, tanh , or any other that you find helpful)
# (No of lines will depend on the method you use sequential method will be around 5 to 10 lines of code)

### START CODE ###
class Neural_Net(nn.Module):
  
  def __init__(self, input_dim, hidden_dims, output_dim):
    super().__init__()
    self.fc1 = nn.Linear(input_dim, hidden_dims[0])
    self.activation1 = nn.ReLU()
    self.fc2 = nn.Linear(hidden_dims[0], hidden_dims[1])
    self.activation2 = nn.ReLU()
    self.fc3 = nn.Linear(hidden_dims[1], output_dim)
    
  def forward(self, X):
    z1 = self.fc1(X)
    z2 = self.activation1(z1)
    z3 = self.fc2(z2)
    z4 = self.activation2(z3)
    z5 = self.fc3(z4)
    return z5
  

model = Neural_Net(input_size, hidden_sizes, output_dim)
###  END CODE  ###

# Define your Loss Function and use an optimiser which will help your model converge 
# There are various optimisers out there I used Stocastic Gradient Descent in the example
# You can try using Adam, AdaGrad and check how the accuracy/time to converge varies 
# (Approx 2 lines of code)

### START CODE ###
loss_func = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
###  END CODE  ###

epochs = 2

# You can change the no of epochs 

# Write a loop that will train your model that you defined earlier 
# You'll have to use trainloader 
# Remember to do optimser.zero_grad() in the loop ( for those who don't know why? google it :) ) 
# Print the loss after every epoch
# Don't keep your epoch more than 20 
# This model will take time to train so you'll have to be patient 
# When i trained the model on my local setup it took 15 mins so it might take more time on colab so please be patient 
# Choose your learning rates wisely 
# You will also have to flatten the images 
# HINT : image.view(images.shape[0], -1) ( Not a hint this is the code :P )
# (Approx 15 - 20 lines of code)

### START CODE ###
for epoch in range(epochs):
  for images, labels in trainloader:
    images = torch.autograd.Variable(images.view(-1, 3072))
    labels = torch.autograd.Variable(labels)
    optimizer.zero_grad()
    labels_predict = model(images)
    loss = loss_func(labels_predict, labels)
    loss.backward()
    optimizer.step()
  
###  END CODE  ###

# Test your model
# Print the accuracy on the val set 
# ( Approx lines sorry i didn't count :( )

### START CODE ###
correct, total = 0, 0
for images, labels in testloader:
  images = torch.autograd.Variable(images.view(-1, 3072))
  labels_predicted = model(images)
  predicted_val, predicted_labels = torch.max(labels_predicted, 1)
  total += labels.size(0)
  correct += (labels == predicted_labels).sum()
accuracy = 100 * correct / total
print(accuracy)
###  END CODE  ###

