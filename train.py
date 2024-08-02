from model import *
from data_loaders import *

import torch.optim as optim

wdecay = 0.01
model = Net(256, 12)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=wdecay)
model_dir = './models/'
lowest_validation_loss = None
lowest_validation_accuracy = None


for epoch in range(100):  # loop over the dataset multiple times

	train_loss = 0.0
	train_accuracy = 0.0
	for step, data in enumerate(trainloader, 0):
		# get the inputs; data is a list of [inputs, labels]
		inputs, labels = data

		# zero the parameter gradients
		optimizer.zero_grad()

		# forward + backward + optimize
		outputs = model(inputs)
		loss = criterion(outputs, labels)
		loss.backward()
		optimizer.step()

		# print statistics
		train_loss += loss.item()
		train_accuracy += torch.sum(torch.max(outputs.data, 1)[1] == labels)
		if step % 2000 == 1999:    # print every 2000 mini-batches
			print(f'[{epoch + 1}, {step + 1:5d}] loss: {train_loss / 2000:.3f} accuracy: {train_accuracy / step:.3f}')
			train_loss = 0.0

		if step % 10 == 0:	
			validation_loss =0 
			validation_accuracy = 0
			with torch.no_grad():
				for data in validationloader:
					images, labels = data
					# calculate outputs by running images through the network
					outputs = model(images)
					# the class with the highest energy is what we choose as prediction
					_, predicted = torch.max(outputs.data, 1)
					loss = criterion(outputs, labels)

					validation_loss += loss.item()
					validation_accuracy += torch.sum(torch.max(outputs.data, 1)[0] == labels)
				if lowest_validation_loss == None: lowest_validation_loss = validation_loss
				if lowest_validation_accuracy ==None: lowest_validation_accuracy = validation_accuracy
				if validation_loss < lowest_validation_loss:
					print('NEW LOWEST VAL LOSS')
					lowest_validation_loss = validation_loss
					save_name = f'{model_dir}_{validation_loss}_{epoch}'
					print(f'saving model at: {save_name}')
					torch.save(model.state_dict(), save_name)
			print(f'[{epoch + 1}, {step + 1:5d}] Val_loss: {validation_loss / 2000:.3f} accuracy: {validation_accuracy / step:.3f}')
		
test_loss = 0 
test_accuracy = 0
for data in testloader:
	images, labels = data
	# calculate outputs by running images through the network
	outputs = model(images)
	# the class with the highest energy is what we choose as prediction
	_, predicted = torch.max(outputs.data, 1)
	loss = criterion(outputs, labels)
	test_loss += loss.item()
	test_accuracy += torch.sum(torch.max(outputs.data, 1) == labels)
	print(f'END RESULT test_loss: {test_loss / len(testloader):.3f} accuracy: {validation_accuracy / len(testloader):.3f}')

print('Finished Training')