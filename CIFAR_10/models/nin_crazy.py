import torch.nn as nn
import torch
import torch.nn.functional as F

class BinActive(torch.autograd.Function):
    '''
    Binarize the input activations and calculate the mean across channel dimension.
    '''
    def forward(self, input):
        self.save_for_backward(input)
        size = input.size()
        mean = torch.mean(input.abs(), 1, keepdim=True)
        input = input.sign()
        return input, mean

    def backward(self, grad_output, grad_output_mean):
        input, = self.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input.ge(1)] = 0
        grad_input[input.le(-1)] = 0
        return grad_input

class BinConv2d(nn.Module):
    def __init__(self, input_channels, output_channels,
            kernel_size=-1, stride=-1, padding=-1, dropout=0):
        super(BinConv2d, self).__init__()
        self.layer_type = 'BinConv2d'
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dropout_ratio = dropout

        self.bn = nn.BatchNorm2d(input_channels, eps=1e-4, momentum=0.1, affine=True)
        if dropout!=0:
            self.dropout = nn.Dropout(dropout)
        self.conv = nn.Conv2d(input_channels, output_channels,
                kernel_size=kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        x = self.bn(x)
        x, mean = BinActive()(x)
        if self.dropout_ratio!=0:
            x = self.dropout(x)
        x = self.conv(x)
        x = self.relu(x)
        return x
    
class zDropout2D(nn.Module):
    """
    Dropout without multiplication
    """                            
    def __init__(self):
        super(zDropout2D, self).__init__()

    def forward(self, Input):
        if self.training:
            theShape = Input.size()
            Output = Input * 1.
            for _ in range(2):
                index = int(torch.randint(0,theShape[1],[1]))
                Output[:,index,:,:] = Input[:,index,:,:] * 0.
            return Output
        else:
            return Input
        
class rDropout2D(nn.Module):
    """
    Dropout without multiplication
    """                            
    def __init__(self):
        super(rDropout2D, self).__init__()

    def forward(self, Input):
        if self.training:
            theShape = Input.size()
            Output = Input * 1.
            for _ in range(3):
                index = int(torch.randint(0,theShape[1],[1]))
                for i in range (len(Input)):
                    mean = float(Input[i,index,:,:].mean())
                    std =  float(Input[i,index,:,:].std())
                    size = Input[i,index,:,:].size()
                    Output[i,index,:,:] = torch.normal(mean = mean * torch.ones(size), std = std)
            return Output
        else:
            return Input

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
                rDropout2D(),
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
