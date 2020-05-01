# Train a new network on a dataset and save the model as a checkpoint

'''
Basic usage: python train.py data_directory
Prints out training loss, validation loss, and validation accuracy as the network trains
Options:
- Set directory to save checkpoints: python train.py data_dir --save_dir save_directory
- Choose architecture: python train.py data_dir --arch "vgg13"
- Set hyperparameters: python train.py data_dir --learning_rate 0.01 --hidden_units 512 --epochs 20
- Use GPU for training: python train.py data_dir --gpu
Example usage:
python train.py flowers --gpu --save_dir assets
'''


# Import python modules
import time
import json
import torch
import argparse
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

from PIL import Image
from torch import nn, optim
from collections import OrderedDict
from train_preprocessing import preproc
from workspace_utils import active_session
from train_model import build_model, train_model
from torchvision import datasets, transforms, models




"""   
Create Parser using ArgumentParser to get
the command line input into the scripts
"""
parser = argparse.ArgumentParser()

# Basic usage: python train.py data_directory
parser.add_argument('data_directory', action='store',
                    default = 'flowers', help='Set directory to load training data, e.g., "flowers"')

parser.add_argument('--save_dir', action='store', default = '.',
                    dest='save_dir', help='Sets directory to save checkpoints, e.g., "assets"')

parser.add_argument('--arch', type = str, default = 'densenet121', 
                        help = 'The CNN model architecture to use') 

parser.add_argument('--learning_rate', type = float, default = 0.001, 
                        help = 'learnig rate of the model')

parser.add_argument('--hidden_units', type = int, default = 512, 
                        help = 'number of units in the hidden layer')

parser.add_argument('--epochs', type = int, default = 4, 
                        help = 'number of iterations the images should iterate to have a valid learning')

# Use GPU for training: python train.py data_dir --gpu
parser.add_argument('--gpu', action='store_true',default=False, dest='gpu',
                    help='Use GPU for training, set a switch to true')


#retrive respective cmd line arguments, usage - parse_results.data_directory
parse_results = parser.parse_args()


#values initialied as 

data_dir = parse_results.data_directory
save_dir = parse_results.save_dir
arch = parse_results.arch
learning_rate = parse_results.learning_rate
hidden_units = parse_results.hidden_units
epochs = parse_results.epochs
gpu = parse_results.gpu

# Load and preprocess data
image_datasets, train_loader, valid_loader, test_loader = preproc(data_dir)

# Building and training the classifier
init_model = build_model(arch, hidden_units)
model, optimizer, criterion = train_model(init_model, train_loader, valid_loader, learning_rate, epochs, gpu)

#adding different image loaders to our model
model.class_to_idx = image_datasets['train'].class_to_idx

#creating checkpoint to be used later 
checkpoint = {'model': model,
              'state_dict': model.state_dict(),
              'optimizer_state_dict': optimizer.state_dict,
              'criterion': criterion,
              'epochs': epochs,
              'class_to_idx': model.class_to_idx}

torch.save(checkpoint, save_dir + '/checkpoint.pth')

if save_dir == ".":
    save_dir_name = "current folder"
else:
    save_dir_name = save_dir + " folder"
    
print(f'Checkpoint saved to {save_dir_name}.')