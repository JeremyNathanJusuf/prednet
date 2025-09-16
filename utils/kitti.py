import os
import numpy as np
import h5py
from torch.utils.data import Dataset, DataLoader

class KittiDataset(Dataset):
    def __init__(
        self, 
        data_path, 
        source_path, 
        nt, 
        batch_size, 
        sequence_start_mode='all', 
        output_mode='error'
    ):
        with h5py.File(data_path, 'r') as f:
            self.X  = f['data_0'][:] # (n_images, nb_cols, nb_rows, nb_channels)
            self.X = np.transpose(self.X, (0, 3, 1, 2)) # we will use (n_images, nb_channels, nb_cols, nb_rows)
        with h5py.File(source_path, 'r') as f:
            self.sources  = f['data_0'][:] # (n_images,)

        self.nt = nt
        self.batch_size = batch_size
        self.sequence_start_mode = sequence_start_mode
        self.num_samples = len(self.X)
        self.output_mode = output_mode
        
        if self.sequence_start_mode == 'all':
            self.possible_starts = [
                idx for idx in range(self.X.shape[0]-self.nt) if self.sources[idx] == self.sources[idx+self.nt-1]
            ]
        elif self.sequence_start_mode == 'unique':  # create sequences where each unique frame is in at most one sequence
            idx = 0
            possible_starts = []
            while idx < self.num_samples - self.nt + 1:
                if self.sources[idx] == self.sources[idx + self.nt - 1]:
                    possible_starts.append(idx)
                    idx += self.nt 
                else:
                    idx += 1
            self.possible_starts = possible_starts
            
    def preprocess(self, X):
        return X.astype(np.float32) / 255.
    
    def __getitem__(self, pos_idx):
        idx = self.possible_starts[pos_idx]
        frames = self.preprocess(self.X[idx:idx+self.nt])
        
        if self.output_mode == 'error':
            target = 0.
        elif self.output_mode == 'prediction':
            target = frames
            
        return frames, target
    
    def __len__(self):
        return len(self.possible_starts)
    
class KittiDataloader:
    def __init__(
        self, 
        data_path, 
        source_path, 
        nt, 
        batch_size, 
        sequence_start_mode, 
        output_mode,
        num_workers,
    ):
        self.data_path=data_path
        self.source_path=source_path
        self.nt=nt
        self.batch_size=batch_size 
        self.sequence_start_mode=sequence_start_mode 
        self.output_mode=output_mode
        self.num_workers=num_workers
        
    def dataloader(self):
        kitti_dataset = KittiDataset(
            self.data_path, 
            self.source_path, 
            self.nt, 
            self.batch_size, 
            self.sequence_start_mode, 
            self.output_mode
        )
        dataloader = DataLoader(
            kitti_dataset, 
            batch_size=self.batch_size, 
            shuffle=True, 
            num_workers=self.num_workers,
            drop_last=True
        )
        
        return dataloader
        
        
    