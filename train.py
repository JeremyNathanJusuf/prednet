import os
import hickle as hkl

def train():
    DATA_DIR = '/kitti_data'

    train_file = os.path.join(DATA_DIR, 'X_train.hkl')
    train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
    val_file = os.path.join(DATA_DIR, 'X_val.hkl')
    val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

    print(hkl.load(train_file))