DEVICE='cuda:0'
python main_decay.py --filename decayfo  --pretrained models/res56.best.pth.tar --device $DEVICE --same
python BI.py --pretrained models/56.decayfo.pth.tar --filename decayfo --device $DEVICE --different
