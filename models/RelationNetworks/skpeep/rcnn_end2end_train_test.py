import os
import sys

from models.RelationNetworks.relation_rcnn import test
from models.RelationNetworks.relation_rcnn import train_end2end

os.environ['PYTHONUNBUFFERED'] = '1'
os.environ['MXNET_CUDNN_AUTOTUNE_DEFAULT'] = '0'
os.environ['MXNET_ENABLE_GPU_P2P'] = '0'
# os.environ['MXNET_ENGINE_TYPE'] = 'NaiveEngine'
WDIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(WDIR, '../relation_rcnn'))

if __name__ == "__main__":
    train_end2end.main()
    test.main()
