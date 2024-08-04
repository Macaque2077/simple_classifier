# get dataloaders from CIFAR, split data if needed
import torch
import torchvision
import torchvision.transforms as T
from sklearn.model_selection import StratifiedShuffleSplit
import numpy as np

batch_size = 64
seed = 998
torch.manual_seed(seed)


train_transformer = T.transforms.Compose([
	T.RandomHorizontalFlip(0.5),
	T.RandomResizedCrop(32,scale=(0.7,1),ratio=(1.0,1.0)), #NOTE dont think cropping too much will be good
	T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2,hue=0.2),
	T.RandAugment(num_ops=1, magnitude=8),
	T.ToTensor(),
	T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
	T.RandomErasing(0.5, scale=(0.01,0.3)),

])

test_transformer = T.transforms.Compose([
    T.ToTensor(),
	T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) 
])

classes = ('plane', 'car', 'bird', 'cat',
		'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

def get_data(split = True):

	train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=False, transform=train_transformer)

	if split:
		# Extract labels
		targets = np.array(train_dataset.targets)
		# Create stratified split
		sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=seed)
		train_index, val_index = next(sss.split(np.zeros(len(targets)), targets))

		# Create subsets
		train_data = torch.utils.data.Subset(train_dataset, train_index)
		val_data = torch.utils.data.Subset(train_dataset, val_index)

		# Check class distribution (optional)
		train_targets = np.array([train_dataset.targets[i] for i in train_index])
		val_targets = np.array([train_dataset.targets[i] for i in val_index])

		print("Train class distribution:", np.bincount(train_targets))
		print("Validation class distribution:", np.bincount(val_targets))

		validationloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size,
											shuffle=True, num_workers=2)

		trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size,
												shuffle=True, num_workers=2)
		
	else:
		trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size,
										shuffle=True, num_workers=2)

	test_data = torchvision.datasets.CIFAR10(root='./data', train=False,
										download=False, transform=test_transformer)
	
	testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,
											shuffle=False, num_workers=2)


	if split:
		return trainloader, validationloader ,testloader
	return trainloader, testloader