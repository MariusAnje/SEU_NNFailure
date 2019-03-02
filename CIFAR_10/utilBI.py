import torch
import util

def load_pretrained(filePath, same, model, device):
    pretrained_model = torch.load(filePath)
    useState_dict = model.state_dict()
    preState_dict = pretrained_model['state_dict']
    best_acc = pretrained_model['best_acc']
    
    useKeys = useState_dict.keys()
    preKeys = preState_dict.keys()
    j = 0
    for key in useKeys:
        if key.find('num_batch') == -1:
            useState_dict[key].data = preState_dict[preKeys[j]].data
            j +=1
    model.load_state_dict(useState_dict)
    model.to(device)
    model = torch.nn.DataParallel(model, device_ids=[0, 1, 2, 3])
    return model, best_acc

def BitInverse(i, key, shape, same, model, device):
    
    model, best_acc = load_pretrained(args.pretrained, same, model, device)
    bin_op = util.BinOp(model)
    model.eval()
    bin_op.binarization()
    state_dict = model.state_dict()
    

    if len(shape) == 4:
        size1 = shape[1]
        size2 = shape[2]
        size3 = shape[3]
        if not args.testonly:
            (state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)]).mul_(-1)

    if len(shape) == 1:
        state_dict[key][i].mul_(-1)

    if len(shape) == 2:
        size = state_dict[key].shape[1]
        (state_dict[key][int(i/size)][i%size]).mul_(-1)
    
    model.load_state_dict(state_dict)
    model.eval()
    return model

def bit_inverse_ex(x):
    Expo = (x.log2().floor())
    num = x * pow(2,2 * (-Expo))
    return num

def BitInverseEx(i, key, shape, same, model, device, filepath):
    
    model, best_acc = load_pretrained(filepath, False, model, device)
    bin_op = util.BinOp(model)
    model.eval()
    bin_op.binarization()
    state_dict = model.state_dict()
    

    if len(shape) == 4:
        size1 = shape[1]
        size2 = shape[2]
        size3 = shape[3]
        (state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)]) = bit_inverse_ex(state_dict[key][int(i/size1/size2/size3)][int(i/size2/size3%size1)][int(i/size3%size2)][int(i%size3)])

    if len(shape) == 1:
        state_dict[key][i] = bit_inverse_ex(state_dict[key][i])

    if len(shape) == 2:
        size = state_dict[key].shape[1]
        (state_dict[key][int(i/size)][i%size]) = bit_inverse_ex(state_dict[key][int(i/size)][i%size])
    
    model.load_state_dict(state_dict)
    model.eval()
    return model