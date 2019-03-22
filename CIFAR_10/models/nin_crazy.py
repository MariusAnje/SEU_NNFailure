import torch.nn as nn
import torch
import torch.nn.functional as F
from blocks import *

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.xnor = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),
                zDropout2D(),
                BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                BinConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x


class randNet(nn.Module):
    def __init__(self):
        super(randNet, self).__init__()
        self.xnor = nn.Sequential(
                nn.Conv2d(3, 192, kernel_size=5, stride=1, padding=2),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.ReLU(inplace=True),
                rDropout2D(3),
                BinConv2d(192, 160, kernel_size=1, stride=1, padding=0),
                BinConv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d( 96, 192, kernel_size=5, stride=1, padding=2, dropout=0.5),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),

                BinConv2d(192, 192, kernel_size=3, stride=1, padding=1, dropout=0.5),
                BinConv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.BatchNorm2d(192, eps=1e-4, momentum=0.1, affine=False),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),
                )

    def forward(self, x):
        x = self.xnor(x)
        x = x.view(x.size(0), 10)
        return x
