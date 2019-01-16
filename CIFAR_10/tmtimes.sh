
cp models/nin_halfadd.pth.tar models/nin_halfadd.pth.tar.1
python main_halfadd.py --pretrained models/nin_halfadd.pth.tar
cp models/nin_halfadd.pth.tar models/nin_halfadd.pth.tar.2
python main_halfadd.py --pretrained models/nin_halfadd.pth.tar
cp models/nin_halfadd.pth.tar models/nin_halfadd.pth.tar.3
python main_halfadd.py --pretrained models/nin_halfadd.pth.tar
cp nin_halfadd.pth.tar models/nin_halfadd.pth.tar.4
