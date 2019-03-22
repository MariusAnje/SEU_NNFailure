import torch.nn as nn
import torch
import torch.nn.functional as F
from   blocks import *


class randNet(nn.Module):
    def __init__(self):
        super(randNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2)
        self.bn1 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False)
        self.relu1 = nn.ReLU(inplace=True)
        self.bconv2 = BinConv2d(192, 160, kernel_size=1, stride=1, padding=0)
        self.bconv3 = BinConv2d(160,  96, kernel_size=1, stride=1, padding=0)
        self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.bconv4 = BinConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5)
        self.bconv5 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.bconv6 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.pool2 = nn.AvgPool2d(kernel_size=3, stride=2, padding=1)

        self.bconv7 = BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5)
        self.bconv8 = BinConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.bn2 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False)
        self.conv9 = nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool3 = nn.AvgPool2d(kernel_size=8, stride=1, padding=0)
        
        self.conv1_1 = nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2)
        self.bn1_1 = nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False)
        self.relu1_1 = nn.ReLU(inplace=True)
        
        self.drop = rDropout2D(1)

    def forward(self, x):
        x_0 = self.conv1(x)
        x_0 = self.bn1(x_0)
        x_1 = self.conv1_1(x)
        x_1 = self.bn1_1(x_1)
        x = (x_0 + x_1)/2
        x = self.relu1(x)
        x = self.drop(x)
        x = self.bconv2(x)
        x = self.bconv3(x)
        x = self.pool1(x)
        x = self.bconv4(x)
        x = self.bconv5(x)
        x = self.bconv6(x)
        x = self.pool2(x)
        x = self.bconv7(x)
        x = self.bconv8(x)
        x = self.bn2(x)
        x = self.conv9(x)
        x = self.relu2(x)
        x = self.pool3(x)
        x = x.view(x.size(0), 10)
        return x
