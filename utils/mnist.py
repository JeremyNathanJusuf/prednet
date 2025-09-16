import numpy as np
from torch.utils.data import Dataset, DataLoader

import os
from sklearn.model_selection import train_test_split

def split_mnist_data(datapath, nt, train_ratio=0.8, random_state=42):
    X = np.load(datapath)  # (20, batch, h, w)
    X = np.transpose(X, [1, 0, 2, 3])  # (batch, 20, h, w)
    X = X[:, :nt, ...]  # (batch, nt, h, w)
    X = np.expand_dims(X, axis=2)  # (batch, nt, 1, h, w)

    train_X, val_X = train_test_split(
        X, 
        train_size=train_ratio, 
        random_state=random_state, 
        shuffle=True
    )

    parent_dir = os.path.dirname(datapath)
    train_path = os.path.join(parent_dir, "mnist_train.npy")
    val_path = os.path.join(parent_dir, "mnist_val.npy")

    np.save(train_path, train_X)
    np.save(val_path, val_X)
    return train_path, val_path

class MNISTDataset(Dataset):
    def __init__(
        self, 
        data_path,
    ):
        self.X = np.load(data_path)  # (batch, nt, 1, h, w)
        
    def preprocess(self, X):
        return X.astype(np.float32) / 255.
    
    def __getitem__(self, pos_idx):
        return self.preprocess(self.X[pos_idx, ...])
    
    def __len__(self):
        return len(self.X)
    
class MNISTDataloader:
    def __init__(
        self, 
        data_path,
        batch_size,
        num_workers,
    ):
        self.data_path=data_path
        self.batch_size=batch_size
        self.num_workers=num_workers
        
    def dataloader(self):
        mnist_dataset = MNISTDataset(
            self.data_path
        )
        dataloader = DataLoader(
            mnist_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            drop_last=True
        )
        
        return dataloader
        
        
    