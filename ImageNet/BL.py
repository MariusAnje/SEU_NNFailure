from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os
import torch
import argparse
import Pytorch_VGG

from torch.autograd import Variable
import torchvision
import torchvision.transforms as transforms
import torchvision.datasets as datasets
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

def test(i, key, shape, rand = False, bypass = False, randFactor = None, memoryData = None, same = False):
    global best_acc
    test_loss = 0
    correct = 0
    Flip = int(args.bit)
    if (not rand) or (len(shape) != 4):
        model = Pytorch_VGG.vgg16(pretrained = True)
        model.to(device)
        model.eval()
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
                    model = Pytorch_VGG.vgg16(pretrained = True)
                    model.to(device)
                    model.eval()
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
                model = Pytorch_VGG.vgg16(pretrained = True)
                model.to(device)
                model.eval()
                state_dict[key][i] = BI(state_dict[key][i], Flip)
            else:
                return 100
        else:
            state_dict[key][i] = BI(state_dict[key][i], Flip)

    if len(shape) == 2:
        size = state_dict[key].shape[1]
        """
        if rand:
            if (int(int(i)%randFactor) == torch.randint(0,randFactor-1,[1])):
                model = Pytorch_VGG.vgg16(pretrained = True)
                model.to(device)
                model.eval()
                (state_dict[key][int(i/size)][i%size]) = BI(state_dict[key][int(i/size)][i%size] ,Flip)
            else:
                return 100
        else:
        """
        (state_dict[key][int(i/size)][i%size]) = BI(state_dict[key][int(i/size)][i%size] ,Flip)
            
    with torch.no_grad():
        for data, target in tqdm.tqdm(memoryData):
            data, target = data.to(device), target.to(device)

            output = model(data)
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).cpu().sum()
    acc = 100. * float(correct) / float(len(val_loader.dataset))
    #print (acc,correct,len(testloader.dataset))
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
    parser.add_argument('--device', action='store', default='cuda:3',
            help='input the device you want to use')
    parser.add_argument('--bit', action='store', default='16',
            help='bit to flip')
    args = parser.parse_args()
    if args.verbose:
        print('==> Options:',args)
    
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # set the seed
    torch.manual_seed(1)
    torch.cuda.manual_seed(1)
    indices = np.load("subset.npy")


    valdir = '/home/yanzy/data/val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.Subset(
            datasets.ImageFolder(valdir, transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ])), indices),
        batch_size=64, shuffle=False,
        num_workers=8, pin_memory=False)
    
    model = Pytorch_VGG.vgg16(pretrained = True)
    
    if not args.cpu:
        model.to(device)

    if args.verbose:
        print(model)
        
    #key_list = ["features.0.weight","features.2.weight", "features.5.weight","features.7.weight", "features.10.weight", "classifier.6.weight" ]
    key_list = ["features.0.weight" ]
    memoryData = []
    for data in tqdm.tqdm(val_loader, leave = False):
            memoryData += [data]
    
    for find_key in key_list:
        rand = True
        randFactor = 1
        testFactor = 1
        count = 0
        tLoss = 0
        lMax = 0
        lAvg = 0
        bestAcc = 71.31
        save = []
        

        #find_key = "features.0.weight"
        print(find_key)
        state_dict = model.state_dict()

        for key in state_dict.keys():
            if key.find(find_key) != -1:
                total = 1
                shape = state_dict[key].shape
                use_key = key
                for t in range(len(state_dict[key].shape)):
                    total *= state_dict[key].shape[t]
        

        i_list = range(total)
        if rand and len(shape) == 2:
            i_list = []
            for i in range(int(total/randFactor)):
                theRand = np.random.randint(0,randFactor)
                i_list +=[i*randFactor + theRand]
        print(len(i_list))
        with tqdm.tqdm(i_list) as Loader:
            start = time.time()
            for i in Loader:
                acc = test(i, use_key, shape = shape, rand = rand, randFactor=randFactor, memoryData = memoryData)
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
                if acc < 0.13:
                    break

        np.save(find_key+'.BL'+args.bit, save)
        print ("lAvg = %f%%, Max = %f%%"%(lAvg, lMax))
    exit()
