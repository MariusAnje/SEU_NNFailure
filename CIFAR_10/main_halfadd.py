from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import data
import util_fuc as util
import torch.nn as nn
import torch.optim as optim
import tqdm

from models import nin_halfadd as nin
from torch.autograd import Variable

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
    torch.save(state, 'models/nin_halfaddMarch.pth.tar')

def train(epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(tqdm.tqdm(trainloader, leave = False)):
        # process the weights including binarization
        bin_op.binarization()
        
        # forwarding
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        
        # backwarding
        loss = criterion(output, target)
        loss.backward()
        
        # restore weights
        bin_op.restore()
        bin_op.updateBinaryGradWeight()
        
        optimizer.step()
        if batch_idx % 100 == 0 and (args.verbose):
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tLR: {}'.format(
                epoch, batch_idx * len(data), len(trainloader.dataset),
                100. * batch_idx / len(trainloader), loss.data.item(),
                optimizer.param_groups[0]['lr']))
    return

def test():
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    for data, target in testloader:
        data, target = data.to(device), target.to(device)
                                    
        output = model(data)
        test_loss += criterion(output, target).data.item()
        pred = output.data.max(1, keepdim=True)[1]
        correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    bin_op.restore()
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
            help='the architecture for the network: nin')
    parser.add_argument('--lr', action='store', default='0.01',
            help='the intial learning rate')
    parser.add_argument('--pretrained', action='store', default=None,
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true',
            help='evaluate the model')
    parser.add_argument('--verbose', action='store_true',
            help='output details')
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    args = parser.parse_args()
    
    if (args.verbose):
        print('==> Options:',args)

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # prepare the data
    if not os.path.isfile(args.data+'/train_data'):
        # check the data path
        raise Exception\
                ('Please assign the correct data path with --data <DATA_PATH>')

    trainset = data.dataset(root=args.data, train=True)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=256,
            shuffle=True, num_workers=4)

    testset = data.dataset(root=args.data, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
            shuffle=False, num_workers=4)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    if (args.verbose):
        print('==> building model',args.arch,'...')
    if args.arch == 'nin':
        model = nin.Net()
    else:
        raise Exception(args.arch+' is currently not supported')

    # initialize the model
    if not args.pretrained:
        if (args.verbose):
            print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()
    else:
        if (args.verbose):
            print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.to(device)
        model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    if (args.verbose):
        print(model)

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []

    for key, value in param_dict.items():
        params += [{'params':[value], 'lr': base_lr,
            'weight_decay':0.00001}]

        optimizer = optim.Adam(params, lr=0.10,weight_decay=0.00001)
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
        test()
        exit(0)

    # start training
    with tqdm.tqdm(range(1, 320)) as Loader:
        for epoch in Loader:
            adjust_learning_rate(optimizer, epoch)
            train(epoch)
            acc, bacc = test()
            Loader.set_description("acc: %.2f best_acc: %.2f"%(acc, bacc))
