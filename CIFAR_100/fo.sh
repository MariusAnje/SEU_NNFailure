DEVICE='cuda:0'
python main_firstonlyRand.py --filename 3 --arch 56 --pretrained models/res56.best.pth.tar --device $DEVICE
python BI_HalfRand.py --filename firstonly3 --arch 56 --pretrained models/56.firstRand.3.pth.tar --device $DEVICE
python main_firstonly.py --filename again --arch 56 --pretrained models/res56.best.pth.tar --device $DEVICE
python BI_Half.py --filename firstonlyagain --arch 56 --pretrained models/56.firstonly.again.pth.tar --device $DEVICE
