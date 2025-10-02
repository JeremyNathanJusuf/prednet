import numpy as np
from torch.utils.data import Dataset, DataLoader

import os
from sklearn.model_selection import train_test_split

def split_mnist_data(datapath, nt, train_ratio=0.8, random_state=42):
    X = np.load(datapath)  # (20, batch, h, w)
    X = np.transpose(X, [1, 0, 2, 3])  # (batch, 20, h, w)
    batch, total_frames, h, w = X.shape

    # TODO: Remove later, lets try overfiting one video
    # Create one batch worth of the same video sequence repeated
    videos_per_batch = 64  # batch_size from train.py
    num_repeats = 50  # Repeat the batch this many times to create more training data
    
    total_sequences = videos_per_batch * num_repeats
    subsequences = np.zeros((total_sequences, nt, h, w), dtype=X.dtype)
    
    # Take the first video and first temporal chunk
    single_video = X[0, 0:nt, ...]  # (nt, h, w) - same video for all sequences
    
    # Repeat this single video for all sequences
    for i in range(total_sequences):
        subsequences[i] = single_video
            
    subsequences = np.expand_dims(subsequences, axis=2)  # (total_sequences, nt, 1, h, w)

    # For overfitting, use all data as training data (no train/val split)
    train_X = subsequences
    val_X = subsequences  # Same as train for overfitting

    print(f"Created dataset for overfitting:")
    print(f"  - Total sequences created: {len(subsequences)}")
    print(f"  - Sequence shape: {subsequences.shape}")
    print(f"  - Using 1 video repeated for all sequences")
    print(f"  - Repeated {num_repeats} times for longer epochs")
    print(f"  - This will create {total_sequences // videos_per_batch} batches per epoch")

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
        X = X.astype(np.float32) / 255.0
        X = np.clip(X, 0.0, 1.0)
        return X
    
    def __getitem__(self, pos_idx):
        x = self.X[pos_idx, ...]
        x = self.preprocess(x)
        return x
    
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
            shuffle=False, 
            num_workers=self.num_workers,
            drop_last=True
        )
        
        return dataloader