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
import time
import numpy as np


def test(i, key, shape, rand = False, bypass = False, randFactor = None, memoryData = None):
    global best_acc
    test_loss = 0
    correct = 0
    if (not rand) or (len(shape) != 4):
        model = nin.Net()
        pretrained_model = torch.load(args.pretrained)
        best_acc = pretrained_model['best_acc']
        model.load_state_dict(pretrained_model['state_dict'])
        model.to(device)
        bin_op = util.BinOp(model)
        model.eval()
        bin_op.binarization()
        state_dict = model.state_dict()
    

    if len(shape) == 4:
        size1 = shape[1]
        size2 = shape[2]
        size3 = shape[3]
        if rand:
            if ((int(i/(size2*size3))%int(size1)) == torch.randint(0,size1-1,[1]) or bypass):
                try:
                    flag = int(int(i)%randFactor) == torch.randint(0,randFactor-1,[1])
                except:
                    flag = True
                if (flag):
                    model = nin.Net()
                    pretrained_model = torch.load(args.pretrained)
                    model.load_state_dict(pretrained_model['state_dict'])
                    model.to(device)
                    bin_op = util.BinOp(model)
                    model.eval()
                    bin_op.binarization()
                    state_dict = model.state_dict()
                    (state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)]).mul_(-1)
                else:
                    return 100
            else:
                return 100
        else:
            (state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)]).mul_(-1)

    if len(shape) == 1:
        if rand:
            if (int(int(i)%randFactor) == torch.randint(0,randFactor-1,[1])):
                model = nin.Net()
                pretrained_model = torch.load(args.pretrained)
                model.load_state_dict(pretrained_model['state_dict'])
                model.to(device)
                bin_op = util.BinOp(model)
                model.eval()
                bin_op.binarization()
                state_dict[key][i].mul_(-1)
            else:
                return 100
        else:
            state_dict[key][i].mul_(-1)

    if len(shape) == 2:
        size = state_dict[key].shape[1]
        if rand:
            if (int(int(i)%randFactor) == torch.randint(0,randFactor-1,[1])):
                model = nin.Net()
                pretrained_model = torch.load(args.pretrained)
                model.load_state_dict(pretrained_model['state_dict'])
                model.to(device)
                bin_op = util.BinOp(model)
                model.eval()
                bin_op.binarization()
                (state_dict[key][int(i/size)][i%size]).mul_(-1)
            else:
                return 100
        else:
            (state_dict[key][int(i/size)][i%size]).mul_(-1)
            
    with torch.no_grad():
        for data, target in memoryData:
            data, target = Variable(data.to(device)), Variable(target.to(device))

            output = model(data)
            #test_loss += criterion(output, target).data.item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    bin_op.restore()
    acc = 100. * float(correct) / float(len(testloader.dataset))
    #print (acc,correct,len(testloader.dataset))
    return acc


if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', default='./data/',
            help='dataset path')
    parser.add_argument('--pretrained', action='store', default='nin.best.pth.tar',
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=True,
            help='evaluate the model')
    parser.add_argument('--verbose', action='store_true', default=False,
            help='display more information')
    parser.add_argument('--device', action='store', default='cuda:2',
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
    indices = np.load("subset_CIFAR10.npy")
    testloader = torch.utils.data.DataLoader(testset,
                                 batch_size=512, shuffle=False, num_workers=4)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    model = nin.Net()
    pretrained_model = torch.load(args.pretrained)
    best_acc = pretrained_model['best_acc']
    model.load_state_dict(pretrained_model['state_dict'])

    model.to(device)

    # define solver and criterion
    param_dict = dict(model.named_parameters())
    params = []

    # define the binarization operator
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
        rand = True
        bypass = True
        randFactor = 4
        count = 0
        tLoss = 0
        lMax = 0
        lAvg = 0
        bestAcc = 86.28
        save = []
        memoryData = []

        find_key = "6.conv.weight"
        print(find_key)
        state_dict = model.state_dict()
    
        for key in state_dict.keys():
            if key.find(find_key) != -1:
                total = 1
                shape = state_dict[key].shape
                use_key = key
                for t in range(len(state_dict[key].shape)):
                    total *= state_dict[key].shape[t]       
        
        for data in tqdm.tqdm(testloader, leave = False):
            memoryData += [data]
        
        
        with tqdm.tqdm(range(total)) as Loader:
            start = time.time()
            for i in Loader:
                acc = test(i, use_key, shape = shape, rand = rand, bypass = bypass, randFactor=randFactor, memoryData = memoryData)
                loss = bestAcc - acc
                
                if (acc != 100):
                    count += 1
                    lAvg  = tLoss / float(count)
                    tLoss += loss
                    save.append((i,loss))
                    Loader.set_description("T: %d, Av: %.2f%%, M: %.2f%%"%(count, lAvg, lMax))
 
                if (loss > lMax):
                    lMax = loss

                end = time.time()
                if (end - start > 300):
                    np.save(find_key+'_tmp',save)
                    start = end

        np.save(find_key+'.neg', save)
        print ("lAvg = %f%%, Max = %f%%"%(lAvg, lMax))
        exit()
