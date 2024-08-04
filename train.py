from model import *
from data_loaders import get_data
import torch.nn as nn
import torch
import os
import datetime
from time import time 

import numpy as np
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import confusion_matrix

import torch.optim as optim
import logging 
from train_config import *


classes = ('plane', 'car', 'bird', 'cat',
		'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

trainloader, testloader = get_data(split=False)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')

time_now = datetime.datetime.now().__format__(r'%y%m%d_%H_%M')

class trainer():
	def __init__(self, net):
		self.model= net
		self.wdecay = weight_decay
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=self.wdecay)
		self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.001, max_lr=0.005)
		self.scaler = torch.cuda.amp.GradScaler() #Gradient Scaler
		self.total_epochs = 25
		self.epoch=0
		self.lowest_validation_loss = None
		self.highest_validation_accuracy = 0.0

	def schedule_lr(self):
		total_steps = self.total_epochs
		step_size_up = total_steps * 2 // 5
		step_size_down = total_steps * 2 // 5

		# Set up CyclicLR scheduler
		self.scheduler = optim.lr_scheduler.CyclicLR(
			self.optimizer,
			base_lr=0,                # Minimum learning rate (corresponds to 0 in the interpolation)
			max_lr=learning_rate,       # Maximum learning rate (corresponds to args.lr_max in the interpolation)
			step_size_up=step_size_up,
			step_size_down=step_size_down,
			mode='triangular',        # Triangular mode for the cyclic schedule
			gamma=1.0,                # No decay in max_lr during cycles
			cycle_momentum=False      # Set to True if using momentum-based optimizers
		)

	 # Function to load model and optimizer state
	def load_model(self, checkpoint_path):
		checkpoint = torch.load(checkpoint_path)
		self.model = checkpoint
	
	#update accuracy at epoch end
	def update_best_accuracy(self, new_accuracy):
		if new_accuracy > self.highest_validation_accuracy: 
			self.highest_validation_accuracy=new_accuracy

	#get number of model params
	def n_params(self):
		return(sum(p.numel() for p in self.model.parameters()))

	#save model file
	def save_model(self, save_name:str):
		logger.info(msg=f'saving model at: {save_name}')
		torch.save(self.model, save_name+'.wb')

# Evaluate end of epoch performance
def validate_epoch(convM, validationloader):
	validation_loss =0 
	validation_accuracy = 0
	val_count =0
	convM.model.eval()
	with torch.no_grad():
		for data in validationloader:
			
			images, labels = data[0].to(device), data[1].to(device)
			# Infer
			with torch.cuda.amp.autocast():
				inferences = convM.model(images)
				loss = convM.criterion(inferences, labels)

			validation_loss += loss.item() * labels.size(0)
			val_count += labels.size(0)

			accuracy = torch.sum(torch.max(inferences.data, 1)[1] == labels)
			validation_accuracy += accuracy
		
		end_of_epoch_acc = 100*(validation_accuracy / val_count).item()
		if convM.lowest_validation_loss == None: 
			convM.lowest_validation_loss = validation_loss
		

		if end_of_epoch_acc > convM.highest_validation_accuracy:
			logger.info(msg='HIGHEST VAL ACC')
			print(end_of_epoch_acc)
			print(convM.highest_validation_accuracy)
			convM.update_best_accuracy(end_of_epoch_acc)
			save_name = os.path.join(model_dir,f'{end_of_epoch_acc:.3f}_{convM.epoch}')
			convM.save_model(save_name)
			
		print(loss.item())

		print('inferences: ', torch.max(inferences.data, 1)[1])
		print('labels: ', labels)
		#writer.add_scalar("Accuracy/validation", accuracy, self.epoch)
		logger.info(msg=f'{convM.epoch + 1}: Val accuracy%: {end_of_epoch_acc:.3f}')
		writer.add_scalar("Loss/validation", round(validation_loss / len(validationloader), 3), convM.epoch)
		writer.add_scalar("Accuracy/validation", end_of_epoch_acc, convM.epoch)
	
def run_epoch(convM, trainloader,validationloader):	
	# loop over the dataset multiple times
	
	train_loss = 0.0
	train_accuracy = 0.0
	train_count = 0
	total_train_loss = 0.0
	for step, (images, labels) in enumerate(trainloader, 0):
		convM.model.train()
		# get the inputs; data is a list of [inputs, labels]
		images, labels = images.cuda(), labels.cuda()
		

		# zero the parameter gradients
		convM.optimizer.zero_grad()

		# infer, calc loss, adjust
		with torch.cuda.amp.autocast():
			inferences = convM.model(images)
			loss = convM.criterion(inferences, labels)
					
		convM.scaler.scale(loss).backward()
		convM.scaler.unscale_(convM.optimizer)
		nn.utils.clip_grad_norm_(convM.model.parameters(), 1.0)
		convM.scaler.step(convM.optimizer)
		convM.scheduler.step()
		convM.scaler.update()			

		train_loss += loss.item()* labels.size(0)
		total_train_loss +=train_loss
		train_accuracy += torch.sum(torch.max(inferences.data, 1)[1] == labels)
		train_count += labels.size(0)
		if step % 100 == 99:    # print every 100 mini-batches
			logger.info(msg=f'{convM.epoch + 1}-{step + 1:5d} train_loss: {train_loss / 100:.3f} train_accuracy%: {100*(train_accuracy / train_count):.3f} LR: {convM.scheduler.get_last_lr()}')
			train_loss = 0.0

	writer.add_scalar("Loss/train", round(total_train_loss / len(trainloader), 3), convM.epoch)
	train_loss = 0.0
	validate_epoch(convM, validationloader)
		
		
	
def train_loop(convM, trainloader, validationloader):
	
	for epoch in range(0, convM.total_epochs, 1): 
		convM.epoch = epoch
		run_epoch(convM, trainloader, validationloader)

	save_name = os.path.join(model_dir,f'Last_{convM.epoch}')
	convM.save_model(save_name)

def test_model(convM, testloader):
	test_loss = 0 
	test_accuracy = 0
	test_count =0
	convM.model.eval()
	all_preds = []
	all_labels = []
	start = datetime.datetime.now()
	convM.model.cpu()
	with torch.no_grad():
		for data in testloader:
			images, labels = data[0], data[1]
			# calculate outputs by running images through the network
			inferences = convM.model(images)
			# the class with the highest energy is what we choose as prediction
			with torch.cuda.amp.autocast():
				inferences = convM.model(images)
				loss = convM.criterion(inferences, labels)
			test_loss += loss.item()
			test_count += labels.size(0)
			pred_classes = torch.max(inferences.data, 1)[1]
			test_accuracy += torch.sum( pred_classes== labels)
			all_preds.extend(pred_classes.numpy())
			all_labels.extend(labels.numpy())
		testset_acc = 100*(test_accuracy / test_count).item()
		logger.info(msg=f'END RESULT test_loss: {test_loss / test_count:.3f} accuracy: {testset_acc:.3f}')
		end = datetime.datetime.now()

	duration = end-start
	logger.info(msg=f'Images processed:{test_count}')
	logger.info(msg=f'time taken to test: {duration.seconds*1000}ms')
	logger.info(msg=f'time taken per image: {(duration.seconds*1000)/test_count:.3f}ms')
	print(confusion_matrix(all_labels,all_preds, labels=classes ))

if __name__ == '__main__':
	logger.info(msg=torch.__version__)
	device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
	logger.info(msg=f'Device:  {device}')
	model_dir = f'./models/{time_now}'
	os.makedirs(model_dir)

	writer = SummaryWriter(model_dir+'/logs')
	logger.info(msg='starting')

	convM = trainer(ConvMixer(width, depth,kernel_size=kernel_size,patch_size=patch_size))
	total_parameters = convM.n_params()
	convM.model.to(device)

	logger.info(msg=f'Total number of parameters: {total_parameters}')
	train_loop(convM,trainloader, testloader)
	test_model(convM,testloader)
	writer.flush()
	logger.info(msg='Finished Training')