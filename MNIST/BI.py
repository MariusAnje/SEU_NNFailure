from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sys
import models
import util
import tqdm
from torchvision import datasets, transforms
from torch.autograd import Variable

import util

def Dict2File(Dict, filename):
    F = open(filename, 'a+')
    F.write(str(Dict))
    F.close()

def test(i, evaluate=False):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0

    bin_op.binarization()
    state_dict = model.state_dict()
    for key in state_dict.keys():
        if key.find("ip2.weight") != -1:
            #print (state_dict[key].shape)
            #size1 = state_dict[key].shape[1]
            #size2 = state_dict[key].shape[2]
            #size3 = state_dict[key].shape[3]
            #(state_dict[key][i/size1/size2/size3][i/size2/size3%size1][i/size3%size2][i%size3]).mul_(-1)
            #return ((state_dict[key][i/size1/size2/size3][i/size2/size3%size1][i/size3%size2][i%size3]) == state_dict[key].view(-1)[i]).float()
            #state_dict[key][i].mul_(-1)
            size = state_dict[key].shape[1]
            (state_dict[key][i/size][i%size]).mul_(-1)
 
    with torch.no_grad():
        for data, target in test_loader:
            if args.cuda:
                data, target = data.cuda(), target.cuda()
            data, target = Variable(data), Variable(target)
            output = model(data)
            test_loss += criterion(output, target).item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()

    bin_op.restore()
    
    acc = float(100 * correct) / float(len(test_loader.dataset))
    if (acc > best_acc):
        best_acc = acc
        if not evaluate:
            save_state(model, best_acc)

    test_loss /= len(test_loader.dataset)
    #print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)'.format(
    #    test_loss * args.batch_size, correct, len(test_loader.dataset),
    #    float(100 * correct) / float(len(test_loader.dataset))))
    if not evaluate:
        print('Best Accuracy: {:.2f}%'.format(best_acc))
    return acc

if __name__=='__main__':
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=8192, metavar='N',
            help='input batch size for training (default: 4096)')
    parser.add_argument('--test-batch-size', type=int, default=8192, metavar='N',
            help='input batch size for testing (default: 4096)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
            help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
            help='random seed (default: 1)')
    parser.add_argument('--arch', action='store', default='LeNet_5',
            help='the MNIST network structure: LeNet_5')
    parser.add_argument('--pretrained', action='store', default='models/BTI.pth.tar',
            help='pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=True,
            help='whether to run evaluation')
    parser.add_argument('--verbose', action='store_true', default=False,
            help='display more information')
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    if args.verbose:
        print(args)
    
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    
    # load data
    kwargs = {'num_workers': 4, 'pin_memory': True} if args.cuda else {}
    test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data', train=False, transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
            batch_size=args.test_batch_size, shuffle=True, **kwargs)
    
    # generate the model
    if args.arch == 'LeNet_5':
        model = models.LeNet_5()
    else:
        print('ERROR: specified arch is not suppported')
        exit()

    if not args.pretrained:
        best_acc = 0.0
    else:
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if args.cuda:
        model.cuda()
    
    if args.verbose:
        print(model)
    param_dict = dict(model.named_parameters())
    params = []
    
    base_lr = 0.1
    
    criterion = nn.CrossEntropyLoss()

    # define the binarization operator
    bin_op = util.BinOp(model)

    if args.evaluate:
        a = 0
        b = 0
        one = []
        point = []
        with tqdm.tqdm(range(10*500)) as Loader:
            for i in Loader:
                pretrained_model = torch.load(args.pretrained)
                best_acc = pretrained_model['acc']
                model.load_state_dict(pretrained_model['state_dict'])
                bin_op = util.BinOp(model)
                acc = test(i, evaluate=True)

                if ( acc + 0.1) < 99.23:
                    point += [acc]
                    a += 1
                if (acc + 1) < 99.23:
                    print (i)
                    one += [acc]
                    b += 1
                #a += acc
                Loader.set_description("a: %d, b: %d"%(a, b))
        print (a)
        print (b)
        Dict2File(point, 'point.txt')
        Dict2File(one, 'one.txt')
        exit()
