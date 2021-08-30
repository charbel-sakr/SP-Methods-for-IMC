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
testloader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False, num_workers=2)

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

def feedforward(x,model_dictionary,B):
    quantized_weight = model_dictionary['conv.0.weight']
    conv0_out = F.relu_(F.conv2d(x,quantized_weight,model_dictionary['conv.0.bias'],padding=1))
    conv1_in = F.max_pool2d(conv0_out,kernel_size=2, stride=2)
    conv1_in = quantizeUnsigned(conv1_in,B,2.53)

    quantized_weight = quantizeSigned(model_dictionary['conv.1.weight'],B,0.36)
    conv1_out = F.relu_(F.conv2d(conv1_in,quantized_weight,model_dictionary['conv.1.bias'],padding=1))
    conv2_in = F.max_pool2d(conv1_out,kernel_size=2, stride=2)
    conv2_in = quantizeUnsigned(conv2_in,B,2.65)

    quantized_weight = quantizeSigned(model_dictionary['conv.2.weight'],B,0.21)
    conv2_out = F.relu_(F.conv2d(conv2_in,quantized_weight,model_dictionary['conv.2.bias'],padding=1))
    conv3_in = conv2_out
    conv3_in = quantizeUnsigned(conv3_in,B,2.58)

    quantized_weight = quantizeSigned(model_dictionary['conv.3.weight'],B,0.18)
    conv3_out = F.relu_(F.conv2d(conv3_in,quantized_weight,model_dictionary['conv.3.bias'],padding=1))
    conv4_in = F.max_pool2d(conv3_out,kernel_size=2, stride=2)
    conv4_in = quantizeUnsigned(conv4_in,B,2.41)

    quantized_weight = quantizeSigned(model_dictionary['conv.4.weight'],B,0.16)
    conv4_out = F.relu_(F.conv2d(conv4_in,quantized_weight,model_dictionary['conv.4.bias'],padding=1))
    conv5_in = conv4_out
    conv5_in = quantizeUnsigned(conv5_in,B,2.56)

    quantized_weight = quantizeSigned(model_dictionary['conv.5.weight'],B,0.15)
    conv5_out = F.relu_(F.conv2d(conv5_in,quantized_weight,model_dictionary['conv.5.bias'],padding=1))
    conv6_in = F.max_pool2d(conv5_out,kernel_size=2, stride=2)
    conv6_in = quantizeUnsigned(conv6_in,B,2.61)

    quantized_weight = quantizeSigned(model_dictionary['conv.6.weight'],B,0.14)
    conv6_out = F.relu_(F.conv2d(conv6_in,quantized_weight,model_dictionary['conv.6.bias'],padding=1))
    conv7_in = conv6_out
    conv7_in = quantizeUnsigned(conv7_in,B,2.71)

    quantized_weight = quantizeSigned(model_dictionary['conv.7.weight'],B,0.28)
    conv7_out = F.relu_(F.conv2d(conv7_in,quantized_weight,model_dictionary['conv.7.bias'],padding=1))
    avg_pool_in = F.max_pool2d(conv7_out,kernel_size=2, stride=2)

    quantized_weight = model_dictionary['classifier.weight']
    linear_input = F.avg_pool2d(avg_pool_in,1)
    linear_input = linear_input.view(linear_input.size(0),-1)
    y = torch.matmul(quantized_weight,linear_input.transpose(0,1))+model_dictionary['classifier.bias'][:,None]
    #result is 10x BS
    return y

def test(model_dictionary,testloader,B):
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            _,predicted = feedforward(inputs,model_dictionary,B).max(0)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            progress_bar(batch_idx, len(testloader), 'Acc: %.3f%% (%d/%d)'
                    %(100.*correct/total, correct, total))

for b in [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]:
    print(b)
    test(model_dictionary,testloader,b)
