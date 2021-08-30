'''Train CIFAR10 with PyTorch.'''
from __future__ import print_function

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms

import os
import argparse

from models import *
from utils import progress_bar

import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
best_acc = 0  # best test accuracy
start_epoch = 0  # start from epoch 0 or last checkpoint epoch

# Data
print('==> Preparing data..')
transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

trainset = torchvision.datasets.CIFAR10(root='/data/shared/shanbhag/sakr2/cifar_granular/data', train=True, download=True, transform=transform_train)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='/data/shared/shanbhag/sakr2/cifar_granular/data', train=False, download=True, transform=transform_test)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

#load numpy files
model_dictionary = {}
folder_name = 'extracted_params_full/'

for l in range(8):
	name = 'conv.'+str(l)
	model_dictionary[name+'.weight'] = torch.from_numpy(np.load(folder_name+name+'.weight.npy')).cuda()
	model_dictionary[name+'.bias']= torch.from_numpy(np.load(folder_name+name+'.bias.npy')).cuda()
	print('done with '+name)
name = 'classifier'
model_dictionary[name+'.weight'] = torch.from_numpy(np.load(folder_name+name+'.weight.npy')).cuda()
model_dictionary[name+'.bias']= torch.from_numpy(np.load(folder_name+name+'.bias.npy')).cuda()
print('done with '+name)

def quantizeSigned(X,B,R=1.0):
    S=1.0/R
    return R*torch.min(torch.pow(torch.tensor(2.0).cuda(),1.0-B)*torch.round(X*S*torch.pow(torch.tensor(2.0).cuda(),B-1.0)),1.0-torch.pow(torch.tensor(2.0).cuda(),1.0-B))
def quantizeUnsigned(X,B,R=2.0):
    S = 2.0/R
    return 0.5*R*torch.min(torch.pow(torch.tensor(2.0).cuda(),1.0-B)*torch.round(X*S*torch.pow(torch.tensor(2.0).cuda(),B-1.0)),2.0-torch.pow(torch.tensor(2.0).cuda(),1.0-B))

def feedforward(x,model_dictionary):
    activations = []
    weights = []
    quantized_weight = model_dictionary['conv.0.weight']
    conv0_out = F.relu_(F.conv2d(x,quantized_weight,model_dictionary['conv.0.bias'],padding=1))
    conv1_in = F.max_pool2d(conv0_out,kernel_size=2, stride=2)


    quantized_weight = model_dictionary['conv.1.weight']
    conv1_out = F.relu_(F.conv2d(conv1_in,quantized_weight,model_dictionary['conv.1.bias'],padding=1))
    conv2_in = F.max_pool2d(conv1_out,kernel_size=2, stride=2)

    #activations.append(np.random.choice((conv2_in.flatten()/2.61).cpu().numpy(),size=(1000,256)))
    #weights.append(np.random.choice((model_dictionary['conv.1.weight'].flatten()/0.36).cpu().numpy(),size=(1000,256)))

    quantized_weight = model_dictionary['conv.2.weight']
    conv2_out = F.relu_(F.conv2d(conv2_in,quantized_weight,model_dictionary['conv.2.bias'],padding=1))
    conv3_in = conv2_out

    #activations.append(np.random.choice((conv3_in.flatten()/2.58).cpu().numpy(),size=(1000,256)))
    #weights.append(np.random.choice((model_dictionary['conv.2.weight'].flatten()/0.21).cpu().numpy(),size=(1000,256)))

    quantized_weight = model_dictionary['conv.3.weight']
    conv3_out = F.relu_(F.conv2d(conv3_in,quantized_weight,model_dictionary['conv.3.bias'],padding=1))
    conv4_in = F.max_pool2d(conv3_out,kernel_size=2, stride=2)

    activations.append(np.random.choice((conv4_in.flatten()/2.41).cpu().numpy(),size=(1000,256)))
    weights.append(np.random.choice((model_dictionary['conv.3.weight'].flatten()/0.18).cpu().numpy(),size=(1000,256)))

    quantized_weight = model_dictionary['conv.4.weight']
    conv4_out = F.relu_(F.conv2d(conv4_in,quantized_weight,model_dictionary['conv.4.bias'],padding=1))
    conv5_in = conv4_out

    activations.append(np.random.choice((conv5_in.flatten()/2.56).cpu().numpy(),size=(1000,256)))
    weights.append(np.random.choice((model_dictionary['conv.4.weight'].flatten()/0.16).cpu().numpy(),size=(1000,256)))

    return np.concatenate(activations,axis=0),np.concatenate(weights,axis=0)

def test(model_dictionary,testloader):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            activations,weights = feedforward(inputs,model_dictionary)
            print(activations.shape)
            print(weights.shape)
            np.save('activations.npy',activations)
            np.save('weights.npy',weights)


test(model_dictionary,testloader)
