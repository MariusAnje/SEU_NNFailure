DEVICE='cuda:3'
python main.py --device $DEVICE --filename 1 --lr 0.01
python main.py --device $DEVICE --filename 2
python main.py --device $DEVICE --filename 3 --lr 2e-3
python main.py --device $DEVICE --filename 4 --lr 2e-3
python main.py --device $DEVICE --filename 5 --lr 1e-4
python main.py --device $DEVICE --filename 6 --lr 1e-4
