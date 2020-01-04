# spatial transformer networks


from __future__ import print_function
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import torchvision
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
import numpy as np 

# interactive mode
plt.ion()

device=torch.device("cuda" if torch.cuda.is_available() else "cpu")


# downloading datasets and create folder automatically
train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(root='~/datasets', train=True, download=True,
                   transform=transforms.Compose([
                       transforms.ToTensor(),
                       transforms.Normalize((0.1307,), (0.3081,))])), 
    batch_size=64, shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(
	datasets.MNIST(root='~/datasets', train=False, download=True,
					transform=transforms.Compose([
						transforms.ToTensor(),
						transforms.Normalize((0.1307,), (0.3081,))
					])), 
	batch_size=64, shuffle=True, num_workers=4)


# define modele
class Net(nn.Module):
	def __init__(self):
		super(Net, self).__init__()
		self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
		self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
		self.fc1 = nn.Linear(320, 50)
		self.fc2 = nn.Linear(50, 10)


	def forward(self, x):
		x=F.relu(F.max_pool2d(self.conv1(x), kernel_size=2))
		x=F.relu(F.max_pool2d(self.conv2(x), kernel_size=2))
		x=x.view(-1,320)
		x=F.relu(self.fc1(x))
		x=F.dropout(x)
		x=self.fc2(x)
		return F.log_softmax(x, dim=1)


model = Net().to(device)

# training 
optimizer = optim.SGD(model.parameters(), lr=0.01)

def train(epoch_idx):
	model.train()
	for batch_idx, (data, target) in enumerate(train_loader):
		data, target = data.to(device), target.to(device)

		optimizer.zero_grad()
		output = model(data)
		loss = F.nll_loss(output, target)
		loss.backward()
		optimizer.step()
		if batch_idx % 500 == 0:
			print('Train Epoch: {}, [{}/{}({:.0f}%)] \tLoss: {}'.format(
				epoch_idx, batch_idx*len(data), len(train_loader.dataset), 
				100. *batch_idx/len(train_loader), loss.item()))



def test():
	with torch.no_grad():
		model.eval()
		test_loss=0
		correct=0
		for data, target in test_loader:
			data, target = data.to(device), target.to(device)
			output = model(data)

			# sum up batch loss to calculate average loss over test set
			test_loss+=F.nll_loss(output, target, size_average=False).item()
			# sum up batch accuracy to calculate average accuracy over test set
			# 获取最大值下标 
			pred = output.max(1, keepdim=True)[1]
			correct+= pred.eq(target.view_as(pred)).sum().item()

		test_loss /= len(test_loader.dataset)
		correct /= len(test_loader.dataset)
		print('\n Test set:  average loss: {:.4f} \t average accuracy: {:.6f}\n'.format(test_loss, correct))



if __name__ == '__main__':
	for epoch in range(1, 20+1):
		train(epoch)
		test()






