import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dsets
from torchvision.transforms import transforms


train_dset = dsets.MNIST(root='./data',
						 train=True,
						 transform=transforms.ToTensor(),
						 download=True)
test_dset = dsets.MNIST(root='./data',
						train=False,
						transform=transforms.ToTensor(),
						download=True)
# Making datasets iteratable
train_loader = torch.utils.data.DataLoader(dataset=train_dset,
										   batch_size=100,
										   shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_dset,
										  batch_size=100,
										  shuffle=False)
# defining model class
class logistic_regression_model(nn.Module):

	def __init__(self, input_dim, output_dim):
		super().__init__()
		self.linear = nn.Linear(input_dim, output_dim)

	def forward(self, X):
		return self.linear(X)


# Instantiate model
input_dim, output_dim = 28*28, 10
model = logistic_regression_model(input_dim, output_dim)
# Instantiate loss func
loss_func = nn.CrossEntropyLoss()
# Instantiate optimizer
lr = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr=lr)
# training model
num_epochs = 5
iter_ = 0
for epoch in range(num_epochs):
	for i, (images, labels) in enumerate(train_loader):
		# Making torch Variable
		images = Variable(images.view(-1, 28*28))
		labels = Variable(labels)
		# Clearing grads
		optimizer.zero_grad()
		# Forward pass
		labels_predict = model(images)
		# Calculate loss
		loss = loss_func(labels_predict, labels)
		# Getting gradients
		loss.backward()
		# Update params
		optimizer.step()
		iter_ += 1
		if iter_ % 500 == 0:
			# Calculate Accuracy
			correct, total = 0, 0
			# Going through the test set
			for images, labels in test_loader:
				images = Variable(images.view(-1 , 28*28))
				labels_predict = model(images)
				_, predicted = torch.max(labels_predict.data, 1)
				total += labels.size(0)
				correct += (predicted == labels).sum()
			accuracy = 100 * correct / total
			print('Iter:', iter_, 'loss:', loss.data, 'accuracy:', accuracy)
