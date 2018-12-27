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

class StoreActivation(nn.Module):
    def __init__(self, name):
        super(StoreActivation, self).__init__()
        self.name = str(name)
    
    def forward(self, x):
        torch.save(x, name)
        return x

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
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

    def forward(self, x):
        activations = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        activations += [x.cpu()]
        x = self.bconv2(x)
        activations += [x.cpu()]
        x = self.bconv3(x)
        activations += [x.cpu()]
        x = self.pool1(x)
        x = self.bconv4(x)
        activations += [x.cpu()]
        x = self.bconv5(x)
        activations += [x.cpu()]
        x = self.bconv6(x)
        activations += [x.cpu()]
        x = self.pool2(x)
        x = self.bconv7(x)
        activations += [x.cpu()]
        x = self.bconv8(x)
        activations += [x.cpu()]
        x = self.bn2(x)
        x = self.conv9(x)
        activations += [x.cpu()]
        x = self.relu2(x)
        x = self.pool3(x)
        x = x.view(x.size(0), 10)
        return x, activations
