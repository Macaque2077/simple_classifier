#Architecture based off of: https://arxiv.org/pdf/2201.09792v1
import torch.nn as nn

class residual(nn.Module):
	def __init__(self, fn):
		super().__init__()
		self.fn = fn
	
	def  forward(self, x):
		return self.fn(x) + x

def ConvMixer(dim,depth,kernel_size=7,patch_size=4,n_classes=10): 
	return nn.Sequential( 
		nn.Conv2d(3,dim,kernel_size=patch_size,stride=patch_size), 
		nn.LeakyReLU(), 
		nn.BatchNorm2d(dim), 
			*[nn.Sequential( 
				residual(nn.Sequential( 
					nn.Conv2d(dim,dim,kernel_size,groups=dim,padding="same"), 
					nn.LeakyReLU(), 
					nn.BatchNorm2d(dim) 
				)), 
				nn.Conv2d(dim,dim,kernel_size=1), 
				nn.LeakyReLU(), 
				nn.BatchNorm2d(dim) 
		) for i in range(depth)], 
		nn.AdaptiveAvgPool2d((1,1)), 
		nn.Flatten(), 
		nn.Linear(dim,n_classes)
	)