import torch

def loadfirst(filepath, key = 'conv1.weight'):
    thefile = torch.load(filepath)
    state_dict = thefile['state_dict']
    return state_dict[key]
