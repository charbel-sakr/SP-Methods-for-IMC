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
def quantizeSigned(X,B,R=1.0):
    S=1.0/R
    return R*torch.min(torch.pow(torch.tensor(2.0).cuda(),1.0-B)*torch.round(X*S*torch.pow(torch.tensor(2.0).cuda(),B-1.0)),1.0-torch.pow(torch.tensor(2.0).cuda(),1.0-B))
def quantizeUnsigned(X,B,R=2.0):
    S = 2.0/R
    return 0.5*R*torch.min(torch.pow(torch.tensor(2.0).cuda(),1.0-B)*torch.round(X*S*torch.pow(torch.tensor(2.0).cuda(),B-1.0)),2.0-torch.pow(torch.tensor(2.0).cuda(),1.0-B))
normalize = transforms.Normalize(mean=[0.485,0.456,0.406],std=[0.229,0.224,0.225])
traindir = '/scratch/IMAGENET/data-dir/rawd-data/train'
valdir = '/scratch/IMAGENET/data-dir/raw-data/validation'

#train_dataset = datasets.ImageFolder(traindir,transforms.Compose([transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip(),transforms.ToTensor(),normalize]))
#train_loader = torch.utils.data.DataLoader(train_dataset,batch_size=256,shuffle=True)

val_loader = torch.utils.data.DataLoader(datasets.ImageFolder(valdir,transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),normalize])),batch_size=256,shuffle=False)
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
    with torch.no_grad():
        B=8.
        X = quantizeSigned(X,B,4)
        WF1=quantizeSigned(weights['features.0.weight'],B,1)
        BF1=weights['features.0.bias']
        F1=F.conv2d(X,WF1,bias=BF1,stride=4,padding=2)
        F1 = quantizeUnsigned(F.relu_(F1),B,16)
        F2=F.max_pool2d(F1,kernel_size=3,stride=2)

        WF3=quantizeSigned(weights['features.3.weight'],B,4)
        BF3=weights['features.3.bias']
        F3=F.conv2d(F2,WF3,bias=BF3,padding=2)
        F4=F.max_pool2d(F.relu_(F3),kernel_size=3,stride=2)
        F4 = quantizeUnsigned(F4,B,16)

        WF6=quantizeSigned(weights['features.6.weight'],B,1)
        BF6=weights['features.6.bias']
        F6=F.conv2d(F4,WF6,bias=BF6,padding=1)
        F6 = quantizeUnsigned(F.relu_(F6),B,16)

        WF8=quantizeSigned(weights['features.8.weight'],B,0.5)
        BF8=weights['features.8.bias']
        F8=F.conv2d(F6,WF8,bias=BF8,padding=1)
        F8 = quantizeUnsigned(F.relu_(F8),B,32)

        WF10=quantizeSigned(weights['features.10.weight'],B,0.25)
        BF10=weights['features.10.bias']
        F10=F.conv2d(F8,WF10,bias=BF10,padding=1)

        F12=F.max_pool2d(F.relu_(F10),kernel_size=3,stride=2)

        CIN = F.adaptive_avg_pool2d(F12,(6,6))
        CIN=CIN.view(CIN.size(0),256*6*6)
        CIN = quantizeUnsigned(CIN,B,32)

        WC1=quantizeSigned(weights['classifier.1.weight'],B,0.125)
        BC1=weights['classifier.1.bias']
        C1 = torch.matmul(WC1,0.5*CIN.transpose(0,1))+BC1[:,None]
        C1 = quantizeUnsigned(F.relu_(C1),B,16)

        WC4=quantizeSigned(weights['classifier.4.weight'],B,0.125)
        BC4=weights['classifier.4.bias']
        C4 = torch.matmul(WC4,0.5*C1)+BC4[:,None]
        C4 = quantizeUnsigned(F.relu_(C4),B,8)

        WC6=quantizeSigned(weights['classifier.6.weight'],B,0.25)
        BC6=weights['classifier.6.bias']
        C6 = torch.matmul(WC6,F.relu_(C4))+BC6[:,None]

        return C6.transpose(0,1)


def validate(val_loader,weights):#, criterion):#, args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #model = model.to(device)
    batch_time = AverageMeter('Time', ':6.3f')
    #losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
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
            output = forward(input,weights)
            #loss = criterion(output, target)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            #losses.update(loss.item(), input.size(0))
            top1.update(acc1[0], input.size(0))
            top5.update(acc5[0], input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            #if i % args.print_freq == 0:
            #    progress.print(i)

        # TODO: this should also be done with the ProgressMeter
            print(' * Acc@1 {top1.avg:.3f} Acc@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))
            print(i)


    return top1.avg

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


val_acc = validate(val_loader,weights)
print(val_acc)
