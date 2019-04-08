from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import torch.nn as nn
import torch.optim as optim
import tqdm
import torchvision
import torchvision.transforms as transforms
import numpy as np
import torch.nn.functional as F
import models

def save_state(model, best_acc):
    if (args.verbose):
        print('==> Saving model ...')
    state = {
            'best_acc': best_acc,
            'state_dict': model.state_dict(),
            }
    for key in state['state_dict'].keys():
        if 'module' in key:
            state['state_dict'][key.replace('module.', '')] = \
                    state['state_dict'].pop(key)
    torch.save(state, args.arch + '.' + args.filename +'.pth.tar')

def train(epoch):
    model.train()
    with tqdm.tqdm(trainloader) as Loader:
        tLoss = 0
        regC = 0.1
        for batch_idx, (data, target) in enumerate(Loader):

            # forwarding
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            # backwarding
            loss = criterion(output, target)
            tLoss += loss
            #Loader.set_description("c: %.2f, r: %f"%(cLoss, regLoss))
            Loader.set_description("c: %.2f"%(loss))
            loss.backward()

            # restore weights

            optimizer.step()
            if batch_idx % 100 == 0 and (args.verbose):
                print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                    epoch, batch_idx * len(data), len(trainloader.dataset),
                    100. * batch_idx / len(trainloader), loss.data.item(),
                    optimizer.param_groups[0]['lr']))
    return tLoss/(batch_idx + 1)

def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
                                    
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100. * float(correct) / float(len(testloader.dataset))

    if acc > best_acc:
        best_acc = acc
        save_state(model, best_acc)
    
    test_loss /= len(testloader.dataset)
    if args.verbose:
        print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
            test_loss * 128., correct, len(testloader.dataset),
            100. * correct / len(testloader.dataset)))
        print('Best Accuracy: {:.2f}%\n'.format(best_acc))
    
    return acc, best_acc

def adjust_learning_rate(optimizer, epoch):
    update_list = [120, 200, 240, 280]
    if epoch in update_list:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * 0.1
    return

if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', action='store_true',
            help='set if only CPU is available')
    parser.add_argument('--data', action='store', default='./data/',
            help='dataset path')
    parser.add_argument('--arch', action='store', default='nin',
            help='the architecture for the network: network in network')
    parser.add_argument('--lr', action='store', default='0.001',
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,#'nin.best.pth.tar',
            help='the path to the pretrained model')
    parser.add_argument('--verbose', action='store_true',
            help='output details')
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    parser.add_argument('--filename', action='store', default='',
            help='additional file names')
    parser.add_argument('--test_only', action='store_true', default=False,
            help='only test')
    parser.add_argument('--same', action='store_true', default=False,
            help='if datas are the same')
    parser.add_argument('--all', action='store_true', default=False,
            help='if retrain all of them')
    args = parser.parse_args()
    
    if (args.verbose):
        print('==> Options:',args)
    if (args.cpu) or (args.device == 'cpu'):
        args.device = 'cpu'
        args.cpu = True

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")



    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='../CIFAR_10/data', train=True,
                                            download=False, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128,
                                              shuffle=True, num_workers=2, pin_memory=True)

    testset = torchvision.datasets.CIFAR10(root='../CIFAR_10/data', train=False,
                                           download=False, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128,
                                             shuffle=False, num_workers=2, pin_memory=True)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = models.nin()
    model.to(device)
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            m.weight.data.normal_(0, 0.05)
            m.bias.data.normal_(0, 0.0)
    
    criterion = nn.CrossEntropyLoss()
    base_lr = float(args.lr)
    base_lr = 0.1
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        if key == 'classifier.20.weight':
            params += [{'params':[value], 'lr':0.1 * base_lr, 
                'momentum':0.95, 'weight_decay':0.0001}]
        elif key == 'classifier.20.bias':
            params += [{'params':[value], 'lr':0.1 * base_lr, 
                'momentum':0.95, 'weight_decay':0.0000}]
        elif 'weight' in key:
            params += [{'params':[value], 'lr':1.0 * base_lr,
                'momentum':0.95, 'weight_decay':0.0001}]
        else:
            params += [{'params':[value], 'lr':2.0 * base_lr,
                'momentum':0.95, 'weight_decay':0.0000}]

    optimizer = optim.SGD(params, lr=0.1, momentum=0.9)
    
    with tqdm.tqdm(range(1, 320)) as Loader:
        best_acc = 0
        best_loss = 10000
        for epoch in Loader:
            adjust_learning_rate(optimizer, epoch)
            avg = train(epoch)
            acc, bacc = test()
            Loader.set_description("loss: %.2f acc: %.2f best_acc: %.2f"%(avg, acc, bacc))