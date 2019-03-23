python main_Crazy.py  --filename plus2 --arch 56 --pretrained models/res56.best.pth.tar
python BI.py --filename plus2 --arch 56 --pretrained models/56.plus2.crazy.pth.tar
python main_Crazy.py  --filename plus3 --arch 56 --pretrained models/res56.best.pth.tar
python BI.py --filename plus3 --arch 56 --pretrained models/56.plus3.crazy.pth.tar
python main_firstonly.py --filename firstonly --arch 56 --pretrained models/res56.best.pth.tar
python BI_Half.py --filename firstonly --arch 56 --pretrained models/56.firstonly.firstonly.pth.tar

