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

from models import nin_halfadd
from torch.autograd import Variable
import tqdm
import time
import numpy as np

def load_pretrained(filePath, same):
    model = nin_halfadd.Net()
    pretrained_model = torch.load(filePath)
    useState_dict = model.state_dict()
    preState_dict = pretrained_model['state_dict']
    if same:
        useState_dict = preState_dict
    else:
        useKeys = useState_dict.keys()
        preKeys = preState_dict.keys()
        j = 0
        for key in useKeys:
            if j == 50:
                j = 0
            if key.find('num_batches_tracked') == -1:
                useState_dict[key].data = preState_dict[preKeys[j]].data
                j +=1
    model.load_state_dict(useState_dict)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=range(2))
    return model

def BitInverse(i, key, shape, same):
    
    model = load_pretrained(args.pretrained, same)
    bin_op = util.BinOp(model)
    model.eval()
    bin_op.binarization()
    state_dict = model.state_dict()
    

    if len(shape) == 4:
        size1 = shape[1]
        size2 = shape[2]
        size3 = shape[3]
        (state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)]).mul_(-1)

    if len(shape) == 1:
        state_dict[key][i].mul_(-1)

    if len(shape) == 2:
        size = state_dict[key].shape[1]
        (state_dict[key][int(i/size)][i%size]).mul_(-1)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def test(i, key, shape, memoryData, same):
    test_loss = 0
    correct = 0
    
    model = BitInverse(i, key, shape, same)
    model.eval()
    with torch.no_grad():
        for data, target in memoryData:
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
            
    bin_op.restore()
    acc = 100. * float(correct) / float(len(testloader.dataset))
    return acc


if __name__=='__main__':
    # prepare the options
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', action='store', default='./data/',
            help='dataset path')
    parser.add_argument('--pretrained', action='store', default='nin_halfadd_best.pth.tar',
            help='the path to the pretrained model')
    parser.add_argument('--evaluate', action='store_true', default=True,
            help='evaluate the model')
    parser.add_argument('--verbose', action='store_true', default=False,
            help='display more information')
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    parser.add_argument('--same', action='store_true', default=False,
            help='if use same model')
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
    testloader = torch.utils.data.DataLoader(testset,
                                 batch_size=1024, shuffle=False, num_workers=4)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    model = load_pretrained(args.pretrained, True)

    # define the binarization operator
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
        same = args.same
        rand = False
        bypass = True
        randFactor = 4
        count = 0
        tLoss = 0
        lMax = 0
        lAvg = 0
        bestAcc = 86.84
        save = []
        memoryData = []

        find_key = "bconv2.conv.weight"
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
                acc = test(i, use_key, shape = shape, memoryData = memoryData, same = same)
                loss = bestAcc - acc
                
                if (acc != 100):
                    count += 1
                    tLoss += loss
                    lAvg  = tLoss / float(count)
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
