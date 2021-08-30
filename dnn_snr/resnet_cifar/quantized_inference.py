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
folder_name = 'extracted_params/'
name = 'conv1.'
model_dictionary[name+'weight'] = torch.from_numpy(np.load(folder_name+name+'weight.npy')).cuda()
model_dictionary[name+'bias']= torch.from_numpy(np.load(folder_name+name+'bias.npy')).cuda()
print('done with '+name)

for l in ['1','2','3','4']:
    for s in ['0','1']:
        for c in ['1','2']:
            name = 'layer'+l+'.'+s+'.conv'+c
            model_dictionary[name+'.weight']=torch.from_numpy(np.load(folder_name+name+'.weight.npy')).cuda()
            model_dictionary[name+'.bias']=torch.from_numpy(np.load(folder_name+name+'.bias.npy')).cuda()
            print('done with '+name)
        if (l!='1') and (s=='0'):
            name = 'layer'+l+'.'+s+'.shortcut'
            model_dictionary[name+'.weight']=torch.from_numpy(np.load(folder_name+name+'.weight.npy')).cuda()
            model_dictionary[name+'.bias']=torch.from_numpy(np.load(folder_name+name+'.bias.npy')).cuda()
            print('done with '+name)
name = 'linear.'
model_dictionary[name+'weight'] = torch.from_numpy(np.load(folder_name+name+'weight.npy')).cuda()
#print(model_dictionary[name+'weight'].size())
model_dictionary[name+'bias']= torch.from_numpy(np.load(folder_name+name+'bias.npy')).cuda()
print('done with '+name)

def quantizeSigned(X,B,R=1.0):
    S=1.0/R
    return R*torch.min(torch.pow(torch.tensor(2.0).cuda(),1.0-B)*torch.round(X*S*torch.pow(torch.tensor(2.0).cuda(),B-1.0)),1.0-torch.pow(torch.tensor(2.0).cuda(),1.0-B))
def quantizeUnsigned(X,B,R=2.0):
    S = 2.0/R
    return 0.5*R*torch.min(torch.pow(torch.tensor(2.0).cuda(),1.0-B)*torch.round(X*S*torch.pow(torch.tensor(2.0).cuda(),B-1.0)),2.0-torch.pow(torch.tensor(2.0).cuda(),1.0-B))

def feedforward(x,model_dictionary,B):
    x = quantizeSigned(x,B,4.0)
    quantized_weight = quantizeSigned(model_dictionary['conv1.weight'],B,torch.from_numpy(np.load('scalars/conv1.weight.npy')).cuda())
    block_input = F.hardtanh_(F.conv2d(x,quantized_weight,model_dictionary['conv1.bias'],padding=1),0,2)
    block_input = quantizeUnsigned(block_input,B,torch.from_numpy(np.load('scalars/conv1.activation.npy')))
    for l in ['1','2','3','4']:
        for s in ['0','1']:
            name = 'layer'+l+'.'+s
            stride = 2 if ((l!='1') and (s=='0')) else 1
            quantized_weight = quantizeSigned(model_dictionary[name+'.conv1.weight'],B,torch.from_numpy(np.load('scalars/'+name+'.conv1.weight.npy')).cuda())
            intermediate_activation = F.hardtanh_(F.conv2d(block_input,quantized_weight,model_dictionary[name+'.conv1.bias'],stride=stride,padding=1) ,0,2)
            intermediate_activation = quantizeUnsigned(intermediate_activation,B,torch.from_numpy(np.load('scalars/'+name+'.inside.activation.npy')))
            quantized_weight = quantizeSigned(model_dictionary[name+'.conv2.weight'],B,torch.from_numpy(np.load('scalars/'+name+'.conv2.weight.npy')).cuda())
            block_output = F.conv2d(intermediate_activation,quantized_weight,model_dictionary[name+'.conv2.bias'],padding=1)
            quantized_weight = quantizeSigned(model_dictionary[name+'.shortcut.weight'],B,torch.from_numpy(np.load('scalars/'+name+'.shortcut.weight.npy')).cuda()) if ((l!='1') and (s=='0')) else quantized_weight
            shortcut = F.conv2d(block_input,quantized_weight,model_dictionary[name+'.shortcut.bias'],stride=stride) if ((l!='1') and (s=='0')) else block_input
            block_input = F.hardtanh_(block_output+shortcut,0,2)
            block_input = quantizeUnsigned(block_input,B,torch.from_numpy(np.load('scalars/'+name+'.outside.activation.npy')))
    linear_input = F.avg_pool2d(block_input,4)
    linear_input = linear_input.view(linear_input.size(0),-1)
    quantized_weight = quantizeSigned(model_dictionary['linear.weight'],B,torch.from_numpy(np.load('scalars/linear.weight.npy')).cuda())
    y = torch.matmul(quantized_weight,linear_input.transpose(0,1))+model_dictionary['linear.bias'][:,None]
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
