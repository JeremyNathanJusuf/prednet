import os
import h5py
from torch.utils.data import Dataset 

class KittiDataset:
    def __init__(self, data_path, source_path, nt, batch_size):
        with h5py.File(data_path, 'r') as f:
            self.X  = f['data_0'][:] # (n_images, nb_cols, nb_rows, nb_channels)
            
        with h5py.File(source_path, 'r') as f:
            self.sources  = f['data_0'][:] # (n_images,)
            
        self.nt = nt
        self.batch_size = batch_size
        
        self.possible_starts = [idx for idx in range(self.X.shape[0]-self.nt) if self.sources[idx] == self.sources[idx+self.nt-1]]
        
    def __getitem__(self, pos_idx):
        idx = self.possible_starts[pos_idx]
        return self.X[idx:idx+self.nt]
    
    def __len__(self):
        return len(self.possible_starts)
    