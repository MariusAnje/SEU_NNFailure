DEVICE='cuda:2'
python BL.py --bit 128 --device $DEVICE
python BL.py --bit 64 --device $DEVICE
python BL.py --bit 32 --device $DEVICE
python BL.py --bit 16 --device $DEVICE
python BL.py --bit 8 --device $DEVICE
python BL.py --bit 4 --device $DEVICE
python BL.py --bit 2 --device $DEVICE
