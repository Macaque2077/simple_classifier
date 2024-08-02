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
import logging #TODO use this

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger('train')


logger.info(msg=torch.__version__)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
logger.info(msg='Device:  {device}')

time_now = datetime.datetime.now().__format__('%H:%M:%S')
model_dir = f'./models/{time_now}'
os.makedirs(model_dir)
writer = SummaryWriter(model_dir+'/logs')

class trainer():
	def __init__(self, net):
		self.model= net
		self.wdecay = 0.01
		self.criterion = nn.CrossEntropyLoss()
		self.optimizer = optim.Adam(self.model.parameters(), lr=0.001, weight_decay=self.wdecay)
		self.model.to(device)
		self.total_epochs = 100
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
		self.model.eval()
		with torch.no_grad():
			for data in validationloader:
				images, labels = data[0].to(device), data[1].to(device)
				# Infer
				inferences = self.model(images)

				loss = self.criterion(inferences, labels)
				writer.add_scalar("Loss/validation", loss, self.epoch)
				validation_loss += loss.item()
				accuracy = torch.sum(torch.max(inferences.data, 1)[1] == labels)
				validation_accuracy += accuracy
				writer.add_scalar("Accuracy/validation", accuracy, self.epoch)
			
			if self.lowest_validation_loss == None: self.lowest_validation_loss = validation_loss
			if self.lowest_validation_accuracy ==None: self.lowest_validation_accuracy = validation_accuracy

			if validation_loss < self.lowest_validation_loss:
				logger.info(msg='NEW LOWEST VAL LOSS')
				self.lowest_validation_loss = validation_loss
				save_name = f'{model_dir}_{validation_loss}_{self.epoch}'
				self.save_model(save_name)

			logger.info(msg=f'{self.epoch + 1}: Val_loss: {validation_loss / len(validationloader):.3f} accuracy%: {(validation_accuracy // len(validationloader)):.3f}')
	
	def run_epoch(self):
		# loop over the dataset multiple times
		self.model.train()
		train_loss = 0.0
		train_accuracy = 0.0
		for step, data in enumerate(trainloader, 0):
			# get the inputs; data is a list of [inputs, labels]
			inputs, labels = data[0].to(device), data[1].to(device)

			# zero the parameter gradients
			self.optimizer.zero_grad()

			# infer, calc loss, adjust
			inferences = self.model(inputs)
			loss = self.criterion(inferences, labels)
			loss.backward()
			self.optimizer.step()

			# print statistics
			train_loss += loss.item()
			train_accuracy += torch.sum(torch.max(inferences.data, 1)[1] == labels)
			if step % 100 == 99:    # print every 100 mini-batches
				logger.info(msg=f'{self.epoch + 1}-{step + 1:5d} train_loss: {train_loss / 100:.3f} train_accuracy%: {train_accuracy / step:.3f}')
				train_loss = 0.0

		self.validate_epoch()
	
	def train_loop(self):
		
		for epoch in range(self.total_epochs): 
			self.epoch = epoch
			self.run_epoch()
		self.save_model('last')

	def test_model(self):
		test_loss = 0 
		test_accuracy = 0
		for data in testloader:
			images, labels = data[0].to(device), data[1].to(device)
			# calculate outputs by running images through the network
			inferences = self.model(images)
			# the class with the highest energy is what we choose as prediction
			_, predicted = torch.max(inferences.data, 1)
			loss = self.criterion(predicted, labels)
			test_loss += loss.item()
			test_accuracy += torch.sum(torch.max(inferences.data, 1) == labels)
			logger.info(msg=f'END RESULT test_loss: {test_loss / len(testloader):.3f} accuracy: {test_accuracy / len(testloader):.3f}')

if __name__ == '__main__':
	logger.info(msg='starting')
	classifier = trainer(Net(256, 8))
	total_parameters = classifier.calculate_params()
	logger.info(msg=f'Total number of parameters: {total_parameters}')
	classifier.train_loop()
	classifier.test_model()
	writer.flush()



	logger.info(msg='Finished Training')