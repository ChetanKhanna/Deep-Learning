# Standard pytorch imports
import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision.transforms import transforms
import torchvision.datasets as dsets


# Getting dataset
train_dset = dsets.MNIST(root='./data',
                         train=True,
                         transform=transforms.ToTensor(),
                         download=True)
test_dset = dsets.MNIST(root='./data',
                        train=False,
                        transform=transforms.ToTensor(),
                        download=True)

# Make data interable
batch_size = 100
num_epochs = 5
train_loader = torch.utils.data.DataLoader(dataset=train_dset,
                                           batch_size=batch_size,
                                           shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dset,
                                          batch_size=batch_size,
                                          shuffle=False)


# Defining model class
class feed_forward_neural_network(nn.Module):

    def __init__(self, input_dim, hidden_dim,
                 output_dim, activation_func):

        super().__init__()
        # initializin forward-cost-1
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        # initializing sigmoid activation
        if activation_func == 1:
            self.activation_func = nn.Sigmoid()
        elif activation_func == 2:
            self.activation_func = nn.Tanh()
        else:
            self.activation_func = nn.ReLU()
        # initializing forward-cost-2
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, X):
        z1 = self.fc1(X)
        z2 = self.activation_func(z1)
        z3 = self.fc2(z2)
        return z3


# Input activation function:
activation_func = input('1. sigmoid\n2. Tanh\n3. ReLU\n')
# Instantiate model
input_dim = 28*28
hidden_dim = 100
output_dim = 10
model = feed_forward_neural_network(input_dim, hidden_dim,
                                    output_dim, activation_func)
# Checking for gpu
device = torch.device('cuda:0' if torch.cuda.is_available()
                      else 'cpu')
model.to(device)
# Instantiate loss class
loss_func = nn.CrossEntropyLoss()
# Instantiate optimizer class
lr = 0.1
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# Train model
iter_ = 0
for epoch in range(num_epochs):
    for images, labels in train_loader:
        # Load images
        images = Variable(images.view(-1, 28*28).to(device))
        labels = Variable(labels.to(device))
        # Clear out accumulated grads
        optimizer.zero_grad()
        # Feed forward
        labels_predit = model(images)
        # Get loss
        loss = loss_func(labels_predit, labels)
        # Get grads
        loss.backward()
        # Update params
        optimizer.step()

        iter_ += 1
        if iter_ % 500 == 0:
            # Calculate accuracy
            total, correct = 0, 0
            for images, labels in test_loader:
                # Load images
                images = Variable(images.view(-1, 28*28).to(device))
                # predict from current model
                labels_predit = model(images)
                predicted_val, predicted_labels = torch.max(labels_predit.data, 1)
                total += labels.size(0)
                correct += (labels.cpu() == predicted_labels.cpu()).sum()
            accuracy = 100 * correct / total
            print('iter:', iter_, 'accuracy:', accuracy)
