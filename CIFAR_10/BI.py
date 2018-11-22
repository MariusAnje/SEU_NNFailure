from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import data
import util
import torch.nn as nn
import torch.optim as optim

from models import nin
from torch.autograd import Variable
import tqdm

def Dict2File(Dict, filename):
    F = open(filename, 'a+')
    F.write(str(Dict))
    F.close()

def test(i, find_key):
    global best_acc
    model.eval()
    test_loss = 0
    correct = 0
    bin_op.binarization()
    state_dict = model.state_dict()
    
    for key in state_dict.keys():
        if key.find(find_key) != -1:
            if len(state_dict[key].shape) == 4:
                size1 = state_dict[key].shape[1]
                size2 = state_dict[key].shape[2]
                size3 = state_dict[key].shape[3]
                (state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)]).mul_(-1)
                
            if len(state_dict[key].shape) == 1:
                state_dict[key][i].mul_(-1)
                
            if len(state_dict[key].shape) == 2:
                size = state_dict[key].shape[1]
                (state_dict[key][int(i/size)][i%size]).mul_(-1)
            
    with torch.no_grad():
        for data, target in testloader:
            data, target = Variable(data.to(device)), Variable(target.to(device))

            output = model(data)
            test_loss += criterion(output, target).data.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    bin_op.restore()
    acc = 100. * float(correct) / len(testloader.dataset)
    return acc


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
    parser.add_argument('--pretrained', action='store', default='nin.best.pth.tar',
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=True,
            help='evaluate the model')
    parser.add_argument('--verbose', action='store_true', default=False,
            help='display more information')
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    args = parser.parse_args()
    if args.verbose:
        print('==> Options:',args)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)

    # prepare the data
    if not os.path.isfile(args.data+'/train_data'):
        # check the data path
        raise Exception\
                ('Please assign the correct data path with --data <DATA_PATH>')

    testset = data.dataset(root=args.data, train=False)
    testloader = torch.utils.data.DataLoader(testset, batch_size=512,
            shuffle=False, num_workers=4)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    if args.verbose:
        print('==> building model',args.arch,'...')
    if args.arch == 'nin':
        model = nin.Net()
    else:
        raise Exception(args.arch+' is currently not supported')

    # initialize the model
    if not args.pretrained:
        if args.verbose:
            print('==> Initializing model parameters ...')
        best_acc = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                m.bias.data.zero_()
    else:
        if args.verbose:
            print('==> Load pretrained model form', args.pretrained, '...')
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])

    if not args.cpu:
        model.to(device)
        #model = torch.nn.DataParallel(model, device_ids=range(torch.cuda.device_count()))
    if args.verbose:
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
    happy = model.state_dict()

    # do the evaluation if specified
    if args.evaluate:
        a = 0
        b = 0
        one = []
        point = []
        find_key = "3.conv.weight"
        state_dict = model.state_dict()
    
        for key in state_dict.keys():
            if key.find(find_key) != -1:
                total = 1
                for t in range(len(state_dict[key].shape)):
                    total *= state_dict[key].shape[t]       
        
        with tqdm.tqdm(range(total)) as Loader:
            for i in Loader:
                model = nin.Net()
                pretrained_model = torch.load(args.pretrained)
                best_acc = pretrained_model['best_acc']
                model.load_state_dict(pretrained_model['state_dict'])
                model.to(device)
                bin_op = util.BinOp(model)
                acc = test(i, find_key)

                if ( acc + 0.5) < 86.28:
                    point += [i, acc]
                    a += 1
                if (acc + 1) < 86.28:
                    one += [i, acc]
                    b += 1
                #a += acc
                Loader.set_description("a: %d, b: %d"%(a, b))
        print (a)
        print (b)
        Dict2File(point, 'point.txt')
        Dict2File(one, 'one.txt')
        exit()
