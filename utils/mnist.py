import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from sklearn.model_selection import train_test_split


def split_mnist_data(datapath, nt, target_h=128, target_w=160, train_ratio=0.8, random_state=42):
    """
    Split raw MNIST sequence data into train and validation sets.
    
    Args:
        datapath: Path to the raw MNIST npy file (shape: (20, num_videos, h, w))
        nt: Number of timesteps per subsequence
        target_h: Target height for upsampling
        target_w: Target width for upsampling
        train_ratio: Ratio of training data
        random_state: Random seed for reproducibility
    
    Returns:
        train_path, val_path: Paths to saved train and val npy files
    """
    X = np.load(datapath)  # (20, num_videos, h, w)
    X = np.transpose(X, [1, 0, 2, 3])  # (num_videos, 20, h, w)
    num_videos, total_frames, h, w = X.shape
    
    # Calculate total number of subsequences
    num_chunks = total_frames // nt
    total_subseq = num_videos * num_chunks
    
    # Pre-allocate array for subsequences (with upsampling and 3 channels)
    subsequences = np.zeros((total_subseq, nt, 3, target_h, target_w), dtype=np.float32)
    
    subseq_idx = 0
    for i in range(num_videos):
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * nt
            end_idx = start_idx + nt
            chunk = X[i, start_idx:end_idx, ...]  # (nt, h, w)
            
            # Process each frame: normalize, upsample, convert to 3 channels
            for t in range(nt):
                frame = chunk[t].astype(np.float32) / 255.0
                frame = np.clip(frame, 0.0, 1.0)
                
                # Upsample using PIL
                frame_uint8 = (frame * 255).astype(np.uint8)
                pil_img = Image.fromarray(frame_uint8, mode='L')
                pil_img_resized = pil_img.resize((target_w, target_h), Image.LANCZOS)
                frame_resized = np.array(pil_img_resized).astype(np.float32) / 255.0
                
                # Replicate to 3 channels
                subsequences[subseq_idx, t, 0, :, :] = frame_resized
                subsequences[subseq_idx, t, 1, :, :] = frame_resized
                subsequences[subseq_idx, t, 2, :, :] = frame_resized
            
            subseq_idx += 1
    
    print(f"Created {total_subseq} subsequences with shape: {subsequences.shape}")
    
    # Split into train and validation
    train_X, val_X = train_test_split(
        subsequences, 
        train_size=train_ratio, 
        random_state=random_state, 
        shuffle=True
    )
    
    parent_dir = os.path.dirname(datapath)
    train_path = os.path.join(parent_dir, "mnist_train.npy")
    val_path = os.path.join(parent_dir, "mnist_val.npy")
    
    np.save(train_path, train_X)
    np.save(val_path, val_X)
    
    print(f"Saved train set: {train_X.shape} to {train_path}")
    print(f"Saved val set: {val_X.shape} to {val_path}")
    
    return train_path, val_path


def split_custom_mnist_data(datapath, nt, target_h=128, target_w=160, train_ratio=0.8, random_state=42):
    """
    Split custom MNIST dataset (from generator) into train and validation sets.
    Handles upsampling and channel expansion if needed.
    
    Args:
        datapath: Path to the custom MNIST npy file (shape: (batch, nt, channels, h, w))
        nt: Number of timesteps per subsequence
        target_h: Target height for upsampling
        target_w: Target width for upsampling
        train_ratio: Ratio of training data
        random_state: Random seed for reproducibility
    
    Returns:
        train_path, val_path: Paths to saved train and val npy files
    """
    parent_dir = os.path.dirname(datapath)
    train_path = os.path.join(parent_dir, "mnist_train.npy")
    val_path = os.path.join(parent_dir, "mnist_val.npy")
    
    # Check if already processed
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Train and val files already exist at {train_path} and {val_path}")
        return train_path, val_path
    
    X = np.load(datapath)  # (batch, nt, channels, h, w)
    num_samples, num_frames, channels, h, w = X.shape
    
    # Check if upsampling or channel expansion is needed
    needs_upsample = (h != target_h) or (w != target_w)
    needs_channel_expand = (channels == 1)
    
    if needs_upsample or needs_channel_expand:
        target_channels = 3 if needs_channel_expand else channels
        X_processed = np.zeros((num_samples, num_frames, target_channels, target_h, target_w), dtype=np.float32)
        
        for i in range(num_samples):
            for t in range(num_frames):
                if channels == 1:
                    frame = X[i, t, 0, :, :]  # (h, w)
                else:
                    frame = X[i, t, 0, :, :]  # Use first channel for grayscale processing
                
                # Normalize if needed (check if values are in 0-255 range)
                if frame.max() > 1.0:
                    frame = frame.astype(np.float32) / 255.0
                
                frame = np.clip(frame, 0.0, 1.0)
                
                if needs_upsample:
                    frame_uint8 = (frame * 255).astype(np.uint8)
                    pil_img = Image.fromarray(frame_uint8, mode='L')
                    pil_img_resized = pil_img.resize((target_w, target_h), Image.LANCZOS)
                    frame = np.array(pil_img_resized).astype(np.float32) / 255.0
                
                # Replicate to 3 channels
                X_processed[i, t, 0, :, :] = frame
                X_processed[i, t, 1, :, :] = frame
                X_processed[i, t, 2, :, :] = frame
        
        X = X_processed
        print(f"Processed dataset shape: {X.shape}")
    
    # Split into train and validation
    train_X, val_X = train_test_split(
        X, 
        train_size=train_ratio, 
        random_state=random_state, 
        shuffle=True
    )
    
    np.save(train_path, train_X)
    np.save(val_path, val_X)
    
    print(f"Saved train set: {train_X.shape} to {train_path}")
    print(f"Saved val set: {val_X.shape} to {val_path}")
    
    return train_path, val_path


class MNISTDataset(Dataset):
    """
    Dataset for loading pre-processed MNIST sequences.
    Expects data in shape (batch, nt, channels, h, w) with values in [0, 1].
    Handles upsampling and channel expansion if needed.
    """
    def __init__(
        self, 
        data_path,
        target_h=128,
        target_w=160,
        target_channels=3,
    ):
        self.X = np.load(data_path)
        self.target_h = target_h
        self.target_w = target_w
        self.target_channels = target_channels
        
        # Get current shape
        if len(self.X.shape) == 5:
            _, _, current_channels, current_h, current_w = self.X.shape
        else:
            raise ValueError(f"Expected 5D array, got shape {self.X.shape}")
        
        # Check if preprocessing is needed
        self.needs_upsample = (current_h != target_h) or (current_w != target_w)
        self.needs_channel_expand = (current_channels == 1) and (target_channels == 3)
        
        # If no preprocessing needed and channels match, just expand channels if needed
        if not self.needs_upsample and current_channels == 1 and target_channels == 3:
            self.X = np.repeat(self.X, 3, axis=2)
            self.needs_channel_expand = False
            print(f"Expanded dataset from 1 channel to 3 channels. New shape: {self.X.shape}")
        
    def preprocess(self, X):
        """Preprocess a single sample: upsample and expand channels if needed."""
        # X shape: (nt, channels, h, w)
        if not self.needs_upsample and not self.needs_channel_expand:
            return X.astype(np.float32)
        
        nt = X.shape[0]
        X_processed = np.zeros((nt, self.target_channels, self.target_h, self.target_w), dtype=np.float32)
        
        for t in range(nt):
            frame = X[t, 0, :, :]  # (h, w)
            
            # Normalize if needed
            if frame.max() > 1.0:
                frame = frame.astype(np.float32) / 255.0
            frame = np.clip(frame, 0.0, 1.0)
            
            if self.needs_upsample:
                frame_uint8 = (frame * 255).astype(np.uint8)
                pil_img = Image.fromarray(frame_uint8, mode='L')
                pil_img_resized = pil_img.resize((self.target_w, self.target_h), Image.LANCZOS)
                frame = np.array(pil_img_resized).astype(np.float32) / 255.0
            
            # Replicate to target channels
            for c in range(self.target_channels):
                X_processed[t, c, :, :] = frame
        
        return X_processed
    
    def __getitem__(self, idx):
        x = self.X[idx, ...]
        x = self.preprocess(x)
        return x
    
    def __len__(self):
        return len(self.X)


class MNISTDataloader:
    """
    Dataloader wrapper for MNIST datasets.
    """
    def __init__(
        self, 
        data_path,
        batch_size,
        num_workers,
        target_h=128,
        target_w=160,
        target_channels=3,
    ):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_h = target_h
        self.target_w = target_w
        self.target_channels = target_channels
        
    def dataloader(self, shuffle=True):
        """Create and return a DataLoader."""
        dataset = MNISTDataset(
            self.data_path,
            target_h=self.target_h,
            target_w=self.target_w,
            target_channels=self.target_channels,
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            num_workers=self.num_workers,
            drop_last=False
        )
        
        return dataloader


if __name__ == "__main__":
    # Test with raw MNIST data
    data_path = "./data/MNIST/mnist_test_seq.npy"
    if os.path.exists(data_path):
        train_path, val_path = split_mnist_data(data_path, nt=10)
        
        # Test dataloader
        dataloader = MNISTDataloader(
            data_path=train_path,
            batch_size=8,
            num_workers=2
        ).dataloader()
        
        for batch in dataloader:
            print(f"Batch shape: {batch.shape}")
            break
