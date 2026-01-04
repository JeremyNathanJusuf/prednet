import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import os
from sklearn.model_selection import train_test_split

def _randomize_dataset(datasets):
    """
    Randomize the order of the data in the given dataset
    Currently redundant as we have dataloader to shuffle dataset
    """
    merged_dataset = np.concatenate(datasets) # (merged_num_videos, nt, c, h, w)
    n_videos = merged_dataset.shape[0]
    randomized_idx = np.random.permutation(range(n_videos))

    final_dataset = np.zeros(merged_dataset.shape)

    for idx in range(n_videos):
        random_idx = randomized_idx[idx]
        final_dataset[idx, ...] = merged_dataset[random_idx, ...]

    return final_dataset

def merge_and_split_adapt_data(adapt_datapath, train_ratio=0.8, random_state=42):
    """
    Used to merge adapt dataset (sudden appear, disappear, transform) and splitting it into training and validation dataset
    """
    train_datasets = []
    test_datasets = []

    dataset_paths = [os.path.join(adapt_datapath, file_path) for file_path in os.listdir(adapt_datapath)]

    for file_path in dataset_paths:
        if 'sudden_' in file_path:
            dataset = np.load(file_path) # (num_videos, nt, c, h, w)
            
            X_train, X_test = train_test_split(
                dataset, 
                train_size=train_ratio, 
                random_state=random_state, 
                shuffle=True
            )

            train_datasets.append(X_train)
            test_datasets.append(X_test)

    if train_datasets and test_datasets:
        final_train_dataset = _randomize_dataset(train_datasets)
        final_test_dataset = _randomize_dataset(test_datasets)
        
        os.makedirs('./data/adapt_train', exist_ok=True)
        final_train_path = './data/adapt_train/mnist_adapt_train.npy'
        final_test_path = './data/adapt_train/mnist_adapt_test.npy'

        np.save(final_train_path, final_train_dataset)
        np.save(final_test_path, final_test_dataset)

        print('Added training and validation set for adapt dataset model training')

        return final_train_path, final_test_path

    return None, None
    
def split_mnist_data(datapath, nt, target_h=128, target_w=160, train_ratio=0.8, random_state=42, val_path="mnist_val_multi.npy", train_path="mnist_train_multi.npy"):
    """ 
    Used for initial Moving MNIST Data (num_videos, 20, 64, 64)
    """
    
    parent_dir = os.path.dirname(datapath)
    train_path = os.path.join(parent_dir, train_path)
    val_path = os.path.join(parent_dir, val_path)
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Train and val files already exist at {train_path} and {val_path}")
        return train_path, val_path

    X = np.load(datapath)  # (20, num_videos, h, w)
    X = np.transpose(X, [1, 0, 2, 3])  # (num_videos, 20, h, w)
    num_videos, total_frames, h, w = X.shape
    
    num_chunks = total_frames // nt
    total_subseq = num_videos * num_chunks
    
    # Store as uint8, 1 channel - conversion to float32/3ch happens in dataloader
    subsequences = np.zeros((total_subseq, nt, 1, target_h, target_w), dtype=np.uint8)
    
    subseq_idx = 0
    for i in range(num_videos):
        for chunk_idx in range(num_chunks):
            start_idx = chunk_idx * nt
            end_idx = start_idx + nt
            chunk = X[i, start_idx:end_idx, ...]  # (nt, h, w)
            
            for t in range(nt):
                frame = chunk[t]
                pil_img = Image.fromarray(frame, mode='L')
                pil_img_resized = pil_img.resize((target_w, target_h), Image.LANCZOS)
                subsequences[subseq_idx, t, 0, :, :] = np.array(pil_img_resized)
            
            subseq_idx += 1
    
    print(f"Created {total_subseq} subsequences with shape: {subsequences.shape}")
    
    train_X, val_X = train_test_split(
        subsequences, 
        train_size=train_ratio, 
        random_state=random_state, 
        shuffle=True
    )
    
    np.save(train_path, train_X)
    np.save(val_path, val_X)
    
    print(f"Saved train set: {train_X.shape} to {train_path}")
    print(f"Saved val set: {val_X.shape} to {val_path}")
    
    return train_path, val_path


def split_custom_mnist_data(datapath, nt, target_h=128, target_w=160, train_ratio=0.8, random_state=42):
    """
    used to split custom mnist (generated dataset) with shape (num_videos, nt, c, h, w)
    """
    parent_dir = os.path.dirname(datapath)
    train_path = os.path.join(parent_dir, "mnist_train.npy")
    val_path = os.path.join(parent_dir, "mnist_val.npy")
    
    if os.path.exists(train_path) and os.path.exists(val_path):
        print(f"Train and val files already exist at {train_path} and {val_path}")
        return train_path, val_path
    
    X = np.load(datapath)  # (num_videos, nt, c, h, w)
    
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
    def __init__(self, data_path, target_h=128, target_w=160, target_channels=3):
        self.X = np.load(data_path)  # (batch, nt, 1, h, w) uint8
        self.target_h = target_h
        self.target_w = target_w
        self.target_channels = target_channels
        
    def __getitem__(self, idx):
        x = self.X[idx, ...]  # (nt, 1, h, w) uint8
        
        # Convert to float32 and normalize to [0, 1]
        x = x.astype(np.float32) / 255.0
        
        # Expand to 3 channels if needed
        if x.shape[1] == 1 and self.target_channels == 3:
            x = np.repeat(x, 3, axis=1)  # (nt, 3, h, w)
        
        return x
    
    def __len__(self):
        return len(self.X)


class MNISTDataloader:
    def __init__(self, data_path, batch_size, num_workers, target_h=128, target_w=160, target_channels=3):
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.target_h = target_h
        self.target_w = target_w
        self.target_channels = target_channels
        
    def dataloader(self, shuffle=True):
        dataset = MNISTDataset(
            self.data_path,
            target_h=self.target_h,
            target_w=self.target_w,
            target_channels=self.target_channels,
        )
        
        return DataLoader(
            dataset, 
            batch_size=self.batch_size, 
            shuffle=shuffle, 
            num_workers=self.num_workers,
            drop_last=False
        )


if __name__ == "__main__":
    data_path = "./data/mnist_test_seq.npy"
    if os.path.exists(data_path):
        train_path, val_path = split_mnist_data(data_path, nt=10)
        
        dataloader = MNISTDataloader(
            data_path=train_path,
            batch_size=8,
            num_workers=2
        ).dataloader()
        
        for batch in dataloader:
            print(f"Batch shape: {batch.shape}, dtype: {batch.dtype}")
            break
