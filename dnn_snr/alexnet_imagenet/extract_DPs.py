import argparse
import os
import random
import shutil
import time
import warnings
import sys

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models

import numpy as np

normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
traindir = '/scratch/IMAGENET/data-dir/rawd-data/train'
valdir = '/scratch/IMAGENET/data-dir/raw-data/validation'

#train_dataset = datasets.ImageFolder(traindir,transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize]))
#train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=256,shuffle=True)

val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir,transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize])),batch_size=256,shuffle=True)
print('the dataset has been loaded')
#alexnet = models.alexnet(pretrained=True)
'''
for name,param in alexnet.state_dict().items():
    print(name)
    print(param.size())
    print
'''
folder_name='extracted_params/'
weights={}
weights['features.0.weight']=torch.from_numpy(np.load(folder_name+'features.0.weight.npy')).cuda()
weights['features.0.bias']=torch.from_numpy(np.load(folder_name+'features.0.bias.npy')).cuda()
weights['features.3.weight']=torch.from_numpy(np.load(folder_name+'features.3.weight.npy')).cuda()
weights['features.3.bias']=torch.from_numpy(np.load(folder_name+'features.3.bias.npy')).cuda()
weights['features.6.weight']=torch.from_numpy(np.load(folder_name+'features.6.weight.npy')).cuda()
weights['features.6.bias']=torch.from_numpy(np.load(folder_name+'features.6.bias.npy')).cuda()
weights['features.8.weight']=torch.from_numpy(np.load(folder_name+'features.8.weight.npy')).cuda()
weights['features.8.bias']=torch.from_numpy(np.load(folder_name+'features.8.bias.npy')).cuda()
weights['features.10.weight']=torch.from_numpy(np.load(folder_name+'features.10.weight.npy')).cuda()
weights['features.10.bias']=torch.from_numpy(np.load(folder_name+'features.10.bias.npy')).cuda()
weights['classifier.1.weight']=torch.from_numpy(np.load(folder_name+'classifier.1.weight.npy')).cuda()
weights['classifier.1.bias']=torch.from_numpy(np.load(folder_name+'classifier.1.bias.npy')).cuda()
weights['classifier.4.weight']=torch.from_numpy(np.load(folder_name+'classifier.4.weight.npy')).cuda()
weights['classifier.4.bias']=torch.from_numpy(np.load(folder_name+'classifier.4.bias.npy')).cuda()
weights['classifier.6.weight']=torch.from_numpy(np.load(folder_name+'classifier.6.weight.npy')).cuda()
weights['classifier.6.bias']=torch.from_numpy(np.load(folder_name+'classifier.6.bias.npy')).cuda()
print('the model weights have been loaded')
def forward(X,weights):
    activations = []
    extracted_weights = []
    with torch.no_grad():
        X = X.clamp_(-4,4)
        WF1=weights['features.0.weight']
        BF1=weights['features.0.bias']
        F1=F.conv2d(X,WF1,bias=BF1,stride=4,padding=2)
        F2=F.max_pool2d(F.hardtanh_(F1,0,16),kernel_size=3,stride=2)
        activations.append(np.random.choice(F2.flatten().cpu().numpy(),size=(2000,256))/16.0)

        WF3=weights['features.3.weight']
        BF3=weights['features.3.bias']
        extracted_weights.append(np.random.choice(WF3.flatten().cpu().numpy(),size=(2000,256))/4.0)
        F3=F.conv2d(F2,WF3,bias=BF3,padding=2)
        F4=F.max_pool2d(F.hardtanh_(F3,0,16),kernel_size=3,stride=2)


        activations.append(np.random.choice(F4.flatten().cpu().numpy(),size=(2000,256))/16.0)

        WF6=weights['features.6.weight']
        BF6=weights['features.6.bias']

        extracted_weights.append(np.random.choice(WF6.flatten().cpu().numpy(),size=(2000,256))/1.0)
        
        return np.concatenate(activations,axis=0),np.concatenate(extracted_weights,axis=0)


def validate(val_loader,weights):#, criterion):#, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #progress = ProgressMeter(len(val_loader), batch_time, top1, top5,
    #                         prefix='Test: ')

    # switch to evaluate mode
    #model.eval()

    with torch.no_grad():
        end = time.time()
        for i, (input, target) in enumerate(val_loader):
            input = input.to(device)
            target = target.to(device)

            # compute output
            activations,extracted_weights = forward(input,weights)
            print(activations.shape)
            print(extracted_weights.shape)
            np.save('activations.npy',activations)
            np.save('weights.npy',extracted_weights)
            return

val_acc = validate(val_loader,weights)
print(val_acc)
