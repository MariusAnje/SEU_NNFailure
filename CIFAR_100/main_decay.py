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

from models import resnet
import torchvision
import torchvision.transforms as transforms
    

def load_pretrained(model, filePath, same):
    pretrained_model = torch.load(filePath)
    useState_dict = model.state_dict()
    preState_dict = pretrained_model['state_dict']
    best_acc = pretrained_model['best_acc']
    #print(pretrained_model['best_acc'])
    if same:
        useState_dict = preState_dict
    else:
        useKeys = useState_dict.keys()
        preKeys = preState_dict.keys()
        j = 0
        for key in useKeys:
            if key.find('num_batches_tracked') == -1:
                useState_dict[key].data = preState_dict[preKeys[j]].data
                j +=1
     
    model.load_state_dict(useState_dict)
    
    for m in model.modules():
        if isinstance(m, nn.Conv2d) and (m.in_channels == 3):
            m.weight.data.normal_(0, 0.05)
                      
    
    return model, best_acc

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
    torch.save(state, 'models/'+args.arch + '.' + args.filename +'.pth.tar')

def train(epoch):
    model.train()
    with tqdm.tqdm(trainloader, leave = False) as Loader:
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

def adjust_learning_rate(optimizer, epoch, flag):
    update_list = [80, 120, 200]
    if (epoch in update_list) or flag:
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
    parser.add_argument('--arch', action='store', default='56',
            help='the architecture for the network: res20')
    parser.add_argument('--lr', action='store', default='0.1',
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

    normalize = transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])

    train_dataset = torchvision.datasets.CIFAR100(
        root=args.data,
        train=True,
        download=True,
        transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    trainloader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)

    test_dataset = torchvision.datasets.CIFAR100(
        root=args.data,
        train=False,
        download=True,
        transform=transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ]))
    testloader = torch.utils.data.DataLoader(test_dataset, batch_size=1024, shuffle=False, num_workers=4)

    # define the model
    if args.arch == '56':
        model = resnet.resnet56_cifar(num_classes=100)

    # initialize the model
    if args.pretrained:
        if (args.verbose):
            print('==> Load pretrained model form', args.pretrained, '...')
        model, best_acc = load_pretrained(model, args.pretrained, args.same)
        best_acc = 0
    else:
        best_acc = 0

    if not args.cpu:
        model.to(device)
        #if args.device == 'cuda:0':
        #    model = torch.nn.DataParallel(model, device_ids=[0,2,3])
            
    if (args.verbose):
        print(model)

    # define solver and criterion
    base_lr = float(args.lr)
    param_dict = dict(model.named_parameters())
    params = []
    
    for key, value in param_dict.items():
        if args.all:
            params += [{'params':[value], 'lr': base_lr,
                'weight_decay':0.00001}]
        else:
            if ((key.find('conv1') != -1) and (key.find('layer') == -1) ):
                print(key)
                params += [{'params':[value], 'lr': base_lr,
                    'weight_decay':0.00001}]


    optimizer = optim.SGD(params, lr = base_lr, momentum =0.9, weight_decay=1e-3)
    criterion = nn.CrossEntropyLoss()


    # do the evaluation if specified
    if args.test_only:
        acc, bacc = test()
        print(acc)
        exit(0)

    # start training
    with tqdm.tqdm(range(1, 320)) as Loader:
        best_loss = 10000
        flag_count = 0
        flag = False
        for epoch in Loader:
            avg = train(epoch)
            if avg > best_loss:
                flag_count += 1
            else:
                best_loss = avg
            if flag_count == 10:
                flag = True
                flag_count = 0
            adjust_learning_rate(optimizer, epoch, flag)
            if flag:
                flag = False
            acc, bacc = test()
            Loader.set_description("loss: %.2f acc: %.2f best_acc: %.2f"%(avg, acc, bacc))
