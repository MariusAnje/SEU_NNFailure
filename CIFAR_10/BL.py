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

def BI(Num, Base):
    if Num != 0:
        Exp = int(torch.log2(Num.abs())) + 127
        Bi = int(Exp / Base) % 2
        if Base == 128:
            scale = pow(2., Base - 1)
            if Bi == 1:
                return Num / scale / 2.
            else:
                return Num * scale * 2.
        else:
            scale = pow(2., Base)
            if Bi == 1:
                return Num / scale
            else:
                return Num * scale
    else:
        return Num * scale

def load_pretrained(filePath, same):
    model = nin.Net()
    pretrained_model = torch.load(filePath)
    useState_dict = model.state_dict()
    preState_dict = pretrained_model['state_dict']
    best_acc = pretrained_model['best_acc']
    if same:
        useState_dict = preState_dict
    else:
        useKeys = useState_dict.keys()
        preKeys = preState_dict.keys()
        j = 0
        for key in useKeys:
            useState_dict[key].data = preState_dict[preKeys[j]].data
            j +=1
    model.load_state_dict(useState_dict)
    model.to(device)
    #if args.device == 'cuda:0':
    #    model = torch.nn.DataParallel(model, device_ids=[0,2,3])
    return model, best_acc

def test(i, key, shape, rand = False, bypass = False, randFactor = None, memoryData = None, same = False):
    global best_acc
    test_loss = 0
    correct = 0
    Flip = int(args.bit)
    if (not rand) or (len(shape) != 4):
        model, best_acc = load_pretrained(args.pretrained, same)
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
                    model, best_acc = load_pretrained(args.pretrained, same)
                    model.to(device)
                    bin_op = util.BinOp(model)
                    model.eval()
                    bin_op.binarization()
                    state_dict = model.state_dict()
                    (state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)]) = BI(state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)], Flip)
                else:
                    return 100
            else:
                return 100
        else:
            (state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)]) = BI(state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)], Flip)
        

    if len(shape) == 1:
        if rand:
            if (int(int(i)%randFactor) == torch.randint(0,randFactor-1,[1])):
                model, best_acc = load_pretrained(args.pretrained, same)
                model.to(device)
                bin_op = util.BinOp(model)
                model.eval()
                bin_op.binarization()
                state_dict[key][i] = BI(state_dict[key][i], Flip)
            else:
                return 100
        else:
            state_dict[key][i] = BI(state_dict[key][i], Flip)

    if len(shape) == 2:
        size = state_dict[key].shape[1]
        if rand:
            if (int(int(i)%randFactor) == torch.randint(0,randFactor-1,[1])):
                model, best_acc = load_pretrained(args.pretrained, same)
                model.to(device)
                bin_op = util.BinOp(model)
                model.eval()
                bin_op.binarization()
                (state_dict[key][int(i/size)][i%size]) = BI(state_dict[key][int(i/size)][i%size] ,Flip)
            else:
                return 100
        else:
            (state_dict[key][int(i/size)][i%size]) = BI(state_dict[key][int(i/size)][i%size] ,Flip)
            
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
    parser.add_argument('--device', action='store', default='cuda:0',
            help='input the device you want to use')
    parser.add_argument('--different', action='store_true', default=False,
            help='if using the original model')
    parser.add_argument('--filename', action='store', default='',
            help='The filename we use')
    parser.add_argument('--bit', action='store', default='128',
            help='bit in the exponent to change')
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
                                 batch_size=1024, shuffle=False, num_workers=4)

    # define classes
    classes = ('plane', 'car', 'bird', 'cat',
            'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    # define the model
    same = not args.different
    model, best_acc = load_pretrained(args.pretrained, same)
    model.to(device)

    # define solver and criterion
    param_dict = dict(model.named_parameters())
    params = []

    # define the binarization operator
    bin_op = util.BinOp(model)

    # do the evaluation if specified
    if args.evaluate:
        rand = False
        bypass = True
        randFactor = 4
        count = 0
        tLoss = 0
        lMax = 0
        lAvg = 0
        bestAcc = best_acc
        save = []
        memoryData = []

        find_key = "0.weight"
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
                acc = test(i, use_key, shape = shape, rand = rand, bypass = bypass, randFactor=randFactor, memoryData = memoryData, same=same)
                loss = bestAcc - acc
                
                if (acc != 100):
                    count += 1
                    tLoss += loss
                    lAvg  = tLoss / float(count)
                    save.append((i,loss))
                    if (loss > lMax):
                        lMax = loss
                    Loader.set_description("T: %d, Av: %.2f%%, M: %.2f%%"%(count, lAvg, lMax))
                    
                end = time.time()
                if (end - start > 300):
                    np.save(find_key+'_tmp',save)
                    start = end

        np.save(find_key+'.'+args.filename+'.BL'+args.bit, save)
        print ("lAvg = %f%%, Max = %f%%"%(lAvg, lMax))
        exit()
