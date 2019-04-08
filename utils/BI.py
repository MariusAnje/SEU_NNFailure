import torch

def BF(Num, bit):
    aNum = Num.abs()
    Ex = int(aNum.log2())
    Frac = aNum / pow(2.,Ex) - 1
    FracB = int(Frac * pow(2.,bit))%2
    Num = (Num/aNum)*(aNum + ((-1)**FracB)*pow(2.,-bit))
    return Num

def BL(Num, Base):
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

def BS(Num):
    return Num*(-1)