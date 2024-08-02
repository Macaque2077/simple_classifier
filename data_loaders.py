import torch
import torchvision
import torchvision.transforms as T

batch_size = 64
torch.manual_seed(7341)


train_transformer = T.transforms.Compose([
	T.RandomHorizontalFlip(0.5),
	T.RandomResizedCrop(32,scale=(0.7,1)), #NOTE dont think cropping too much will be good
	T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1,hue=0.2),
	T.RandomPerspective(0.1),
	T.ToTensor(),
	T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) 

])

test_transformer = T.transforms.Compose([
    T.ToTensor(),
	T.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)) 
])

dataset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=train_transformer)

train_size = int(0.8*len(dataset))
validation_size  = len(dataset)-train_size                                      
trainset, validationset = torch.utils.data.random_split(dataset, [train_size, validation_size])

trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

validationloader = torch.utils.data.DataLoader(validationset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=test_transformer)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')