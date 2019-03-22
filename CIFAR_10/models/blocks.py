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
        
class rDropout2D_bak(nn.Module):
    """
    Dropout without multiplication
    """                            
    def __init__(self, dropnum):
        super(rDropout2D, self).__init__()
        self.dropnum = dropnum

    def forward(self, Input):
        if self.training:
            theShape = Input.size()
            Output = Input * 1.
            for _ in range(self.dropnum):
                index = int(torch.randint(0,theShape[1],[1]))
                for i in range (len(Input)):
                    mean = float(Input[i,index,:,:].mean())
                    std =  float(Input[i,index,:,:].std())
                    size = Input[i,index,:,:].size()
                    Output[i,index,:,:] = torch.normal(mean = mean * torch.ones(size), std = std)
            return Output
        else:
            return Input

class rDropout2D_bak1(nn.Module):
    """
    Dropout without multiplication
    """                            
    def __init__(self, dropnum):
        super(rDropout2D, self).__init__()
        self.dropnum = dropnum

    def forward(self, Input):
        if self.training:
            theShape = Input.size()
            #Output = Input * 1.
            #utput = Input
            for _ in range(self.dropnum):
                index = int(torch.randint(0,theShape[1],[1]))
                for i in range (len(Input)):
                    mean = Input[i,index,:,:].mean()
                    std =  Input[i,index,:,:].std()
                    size = Input[i,index,:,:].size()
                    Input[i,index,:,:].data = (torch.randn(size) + mean) * std
            return Input
        else:
            return Input

class rDropout2D(nn.Module):
    """
    Dropout without multiplication
    """                            
    def __init__(self, dropnum):
        super(rDropout2D, self).__init__()
        self.dropnum = dropnum

    def forward(self, Input):
        if self.training:
            theShape = Input.size()
            #Output = Input * 1.
            #utput = Input
            for _ in range(self.dropnum):
                index = int(torch.randint(0,theShape[1],[1]))
                for i in range (len(Input)):
                    mean = Input[i,index,:,:].mean()
                    std =  Input[i,index,:,:].std()
                    size = Input[i,index,:,:].size()
                    Input[i,index,:,:].data = Input[i,index,:,:] + ((torch.randn_like(Input[i,index,:,:]) + mean) * std).data
            return Input
        else:
            return Input


        
class StoreActivation(nn.Module):
    def __init__(self, name):
        super(StoreActivation, self).__init__()
        self.name = str(name)
    
    def forward(self, x):
        torch.save(x, name)
        return x