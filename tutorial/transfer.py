import torch
from torchvision import datasets, transforms, models
import copy
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# transforms
data_transforms = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.ToTensor()])


# loading datasets (create the image folder manually)
train_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder('~/datasets/partial_imageNet/train', data_transforms), 
	batch_size=4, shuffle=True, num_workers=4)

test_loader = torch.utils.data.DataLoader(
	datasets.ImageFolder('~/datasets/partial_imageNet/val', data_transforms),
	batch_size=4, shuffle=True, num_workers=4)


# training model
def train_model(model, optimizer, num_epoch):
	since= time.time()
	best_model = copy.deepcopy(model.state_dict())
	best_acc = 0.0

	for epoch in range(num_epoch):
		training_loss = 0.0
		training_acc=0.0
		test_loss=0.0
		test_acc=0.0

		# training: set model to training mode
		model.train() 
		for inputs, labels in train_loader:
			inputs = inputs.to(device)
			labels = labels.to(device)

			optimizer.zero_grad()
			outputs = model(inputs)
			loss_f =  nn.CrossEntropyLoss()
			loss = loss_f(outputs, labels)
			loss.backward()
			pred = outputs.max(1, keepdim=True)[1]
			optimizer.step()

			training_loss += loss.item() * inputs.size()[0]
			training_acc += pred.eq(labels.view_as(pred)).sum().item()

		# evaluating: set model to evaluating mode
		model.eval()
		for inputs, labels in test_loader:
			inputs = inputs.to(device)
			labels = labels.to(device)
			outputs = model(inputs)
			loss_f = nn.CrossEntropyLoss()
			loss = loss_f(outputs, labels)
			pred = outputs.max(1,keepdim=True)[1]

			test_loss += loss.item() * inputs.size()[0]
			test_acc += pred.eq(labels.view_as(pred)).sum().item()


		# logging
		training_epoch_loss = training_loss / len(train_loader.dataset)
		training_epoch_acc = training_acc / len(train_loader.dataset)
		test_epoch_loss = test_loss / len(test_loader.dataset)
		test_epoch_acc = test_acc / len(test_loader.dataset)

		if test_epoch_acc > best_acc:
			best_acc = test_epoch_acc
			best_model = copy.deepcopy(model.state_dict())


		print('Epoch: {}/{}'.format(epoch, num_epoch))
		print('-'*10)
		print('training loss: {}\t trainng accuracy: {}'.format(
			training_epoch_loss, training_epoch_acc))
		print('test loss: {}\t test accuracy: {}'.format(
			test_epoch_loss, test_epoch_acc))
		print('\n')


	# logging
	duration = time.time()-since
	print('best val accuracy:{}'.format(best_acc))	
	print('training complete in {:.0f}m:{:.0f}s'.format(duration // 60, duration % 60))

	model.load_state_dict(best_model)
	return model




if __name__ =='__main__':
	model_ft = models.resnet18(pretrained=True)
	num_ftrs = model_ft.fc.in_features
	model_ft.fc = nn.Linear(num_ftrs, 2)
	model_ft = model_ft.to(device)
	optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)

	model_ft = train_model(model_ft, optimizer_ft, 25)
