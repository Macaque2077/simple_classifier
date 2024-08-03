from model import *
from data_loaders import *
import torch.nn as nn
import torch
import os
import datetime
from time import time 

import numpy as np
from torch.utils.tensorboard import SummaryWriter

import torch.optim as optim
import logging 

learning_rate = 0.01
weight_decay = 0.001	
width = 256
depth = 8
kernel_size = 7
patch_size = 4

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')


logger.info(msg=torch.__version__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(msg='Device:  {device}')

time_now = datetime.datetime.now().__format__(r'%y%m%d_%H_%M')
model_dir = f'./models/{time_now}'
os.makedirs(model_dir)
writer = SummaryWriter(model_dir+'/logs')

class trainer():
	def __init__(self, net):
		self.model= net
		self.wdecay = weight_decay
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=self.wdecay)
		self.scheduler = torch.optim.lr_scheduler.CyclicLR(self.optimizer, base_lr=0.01, max_lr=0.1)
		self.scaler = torch.cuda.amp.GradScaler() #Gradient Scaler
		self.model.to(device)
		self.total_epochs = 20
		self.epoch=0
		self.lowest_validation_loss = None
		self.lowest_validation_accuracy = None

	def calculate_params(self):
		return(sum(p.numel() for p in self.model.parameters()))

	def save_model(self, save_name:str):
		logger.info(msg=f'saving model at: {save_name}')
		torch.save(self.model.state_dict(), save_name+'.wb')

	def validate_epoch(self):
		validation_loss =0 
		validation_accuracy = 0
		val_count =0
		self.model.eval()
		with torch.no_grad():
			for data in validationloader:
				
				images, labels = data[0].to(device), data[1].to(device)
				# Infer
				with torch.cuda.amp.autocast():
					inferences = self.model(images)
					loss = self.criterion(inferences, labels)

				validation_loss += loss.item() * labels.size(0)
				val_count += labels.size(0)

				accuracy = torch.sum(torch.max(inferences.data, 1)[1] == labels)
				validation_accuracy += accuracy
			
			end_of_epoch_acc = 100*(validation_accuracy / val_count)
			if self.lowest_validation_loss == None: self.lowest_validation_loss = validation_loss
			if self.lowest_validation_accuracy ==None: self.highest_validation_accuracy = end_of_epoch_acc

			if end_of_epoch_acc > self.highest_validation_accuracy:
				logger.info(msg='HIGHEST VAL ACC')
				self.highest_validation_accuracys = end_of_epoch_acc
				save_name = os.path.join(model_dir,f'{end_of_epoch_acc}_{self.epoch}')
				self.save_model(save_name)
				
			print(loss.item())

			print('inferences: ', torch.max(inferences.data, 1)[1])
			print('labels: ', labels)
			#writer.add_scalar("Accuracy/validation", accuracy, self.epoch)
			logger.info(msg=f'{self.epoch + 1}: Val accuracy%: {end_of_epoch_acc:.3f}')
			writer.add_scalar("Loss/validation", round(validation_loss / len(validationloader), 3), self.epoch)
			writer.add_scalar("Accuracy/validation", end_of_epoch_acc, self.epoch)
	
	def run_epoch(self):
		# loop over the dataset multiple times
		
		train_loss = 0.0
		train_accuracy = 0.0
		train_count = 0
		total_train_loss = 0.0
		for step, data in enumerate(trainloader, 0):
			self.model.train()
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data[0].to(device), data[1].to(device)

			# zero the parameter gradients
			self.optimizer.zero_grad()

			# infer, calc loss, adjust
			with torch.cuda.amp.autocast():
				inferences = self.model(inputs)
				loss = self.criterion(inferences, labels)
						
			self.scaler.scale(loss).backward()
			self.scaler.step(self.optimizer)
			self.scaler.update()
			#loss.backward()
			#self.optimizer.step()
			self.scheduler.step()

			train_loss += loss.item()* labels.size(0)
			total_train_loss +=train_loss
			train_accuracy += torch.sum(torch.max(inferences.data, 1)[1] == labels)
			train_count += labels.size(0)
			if step % 100 == 99:    # print every 100 mini-batches
				logger.info(msg=f'{self.epoch + 1}-{step + 1:5d} train_loss: {train_loss / 100:.3f} train_accuracy%: {100*(train_accuracy / train_count):.3f}')
				train_loss = 0.0

		writer.add_scalar("Loss/train", round(total_train_loss / len(trainloader), 3), self.epoch)
		train_loss = 0.0
		self.validate_epoch()
		
		
	
	def train_loop(self):
		
		for epoch in range(0, self.total_epochs, 1): 
			self.epoch = epoch
			self.run_epoch()

		save_name = os.path.join(model_dir,f'Last_{self.epoch}')
		self.save_model(save_name)

	def test_model(self):
		test_loss = 0 
		test_accuracy = 0
		test_count =0
		self.model.eval()
		with torch.no_grad():
			for data in testloader:
				test_count += 1
				images, labels = data[0].to(device), data[1].to(device)
				# calculate outputs by running images through the network
				inferences = self.model(images)
				# the class with the highest energy is what we choose as prediction
				with torch.cuda.amp.autocast():
					inferences = self.model(images)
					loss = self.criterion(inferences, labels)
				test_loss += loss.item()
				test_accuracy += torch.sum(torch.max(inferences.data, 1) == labels)
				logger.info(msg=f'END RESULT test_loss: {test_loss / test_count:.3f} accuracy: {test_accuracy / test_count:.3f}')

if __name__ == '__main__':
	logger.info(msg='starting')
	classifier = trainer(Net(width, depth,kernel_size=kernel_size,patch_size=patch_size))
	total_parameters = classifier.calculate_params()
	logger.info(msg=f'Total number of parameters: {total_parameters}')
	classifier.train_loop()
	classifier.test_model()
	writer.flush()
	logger.info(msg='Finished Training')