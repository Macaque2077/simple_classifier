#test script for running model inference
import torch
from train_config import *
from model import *
from train import test_model,trainer
from data_loaders import get_data

model_file = '/mnt/c/Users/youk3/My Documents/A-codeprojects/classifier/simple_classifier/models/240804_11_20/86.400_24.wb'
convM = trainer(ConvMixer(width, depth,kernel_size=kernel_size,patch_size=patch_size))
convM.load_model(model_file)
_, testloader = get_data(split=False)
test_model(convM, testloader)