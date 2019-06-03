import numpy as np
from os.path import join, expanduser
import torch
from torch.autograd import Variable
import torch.nn as nn


# Loading dataset
HOME = expanduser('~')
path = 'Deep-Learning/linear-regression'
fname = join(HOME, path, 'single-feature.txt')
X, y = np.loadtxt(fname, delimiter=',', unpack=True, dtype=np.float32)
X = X.reshape(-1, 1)
y = y.reshape(-1, 1)


# 1. Making model class
class Linear_Regression(nn.Module):

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, X):
        return self.linear(X)


# 2. Instantiating model
input_dim, output_dim = 1, 1
model = Linear_Regression(input_dim, output_dim)
# 3. instantiate loss class
loss_func = nn.MSELoss()
# 4. instantiate optimizer class
lr = 0.006
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# 5. training model
epochs = 1000
X = Variable(torch.from_numpy(X), requires_grad=True)
y = Variable(torch.from_numpy(y), requires_grad=True)
for epoch in range(epochs):
    optimizer.zero_grad()
    y_predict = model(X)
    loss = loss_func(y_predict, y)
    loss.backward()
    optimizer.step()
    epoch += 1
    print('epoch:', epoch, 'loss:', loss.data)
