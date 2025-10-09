import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image

import os
from sklearn.model_selection import train_test_split

def split_mnist_data(datapath, nt, train_ratio=0.8, random_state=42):
    X = np.load(datapath)  # (20, batch, h, w)
    X = np.transpose(X, [1, 0, 2, 3])  # (batch, 20, h, w)
    batch, total_frames, h, w = X.shape

    # DEBUG: we use only one video in the dataset (overfit the model)
    # we instead just use the same frame over and over again
    subsequences = X[0, :nt, ...]  # (nt, h, w)
    
    # subsequences = X[0, 0, ...].reshape(1, h, w).repeat(nt, axis=0) # (nt, h, w)
    subsequences = np.expand_dims(subsequences, axis=0)  # (1, nt, h, w)
    subsequences = np.expand_dims(subsequences, axis=2)  # (1, nt, 1, h, w)
    subsequences = np.repeat(subsequences, 4000, axis=0)  # (400000, nt, 1, h, w), use batch = 4 to overfit the model
    
    parent_dir = os.path.dirname(datapath)
    train_path = os.path.join(parent_dir, "mnist_overfit_one_video.npy")

    np.save(train_path, subsequences)
    return train_path, train_path
    
    # TODO: revert back to this
    # # Calculate total number of subsequences to pre-allocate array
    # num_chunks = total_frames // nt
    # total_subseq = batch * num_chunks
    # subsequences = np.zeros((total_subseq, nt, h, w), dtype=X.dtype)
    
    # subseq_idx = 0
    # for i in range(batch):
    #     for chunk_idx in range(num_chunks):
    #         start_idx = chunk_idx * nt
    #         end_idx = start_idx + nt
    #         subsequences[subseq_idx] = X[i, start_idx:end_idx, ...]  # (nt, h, w)
    #         subseq_idx += 1
            
    # subsequences = np.expand_dims(subsequences, axis=2)  # (num_subseq, nt, 1, h, w)

    # train_X, val_X = train_test_split(
    #     subsequences, 
    #     train_size=train_ratio, 
    #     random_state=random_state, 
    #     shuffle=True
    # )

    # parent_dir = os.path.dirname(datapath)
    # train_path = os.path.join(parent_dir, "mnist_train.npy")
    # val_path = os.path.join(parent_dir, "mnist_val.npy")

    # np.save(train_path, train_X)
    # np.save(val_path, val_X)
    
    # return train_path, val_path

class MNISTDataset(Dataset):
    def __init__(
        self, 
        data_path,
    ):
        self.X = np.load(data_path)  # (batch, nt, 1, h, w)
        
    def preprocess(self, X):
        # X shape: (nt, 1, h, w)
        X = X.astype(np.float32) / 255.0
        X = np.clip(X, 0.0, 1.0)
        
        # Upsample to 128x160 and convert to RGB (3 channels)
        # nt, c, h, w = X.shape
        # target_h, target_w = 128, 160
        # X_upsampled = np.zeros((nt, 3, target_h, target_w), dtype=np.float32)
        # X_upsampled = X.repeat(3, axis=1)
        X_upsampled = X
        
        # for t in range(nt):
        #     # Get single frame (1, h, w) and squeeze to (h, w)
        #     frame = X[t, 0, :, :]  # (h, w)
            
        #     # Convert to PIL Image for upsampling
        #     # Scale to 0-255 for PIL, then back to 0-1
        #     frame_uint8 = (frame * 255).astype(np.uint8)
        #     pil_img = Image.fromarray(frame_uint8, mode='L')
            
        #     # Resize using LANCZOS interpolation
        #     pil_img_resized = pil_img.resize((target_w, target_h), Image.LANCZOS)
            
        #     # Convert back to numpy and normalize
        #     frame_resized = np.array(pil_img_resized).astype(np.float32) / 255.0
            
        #     # Convert grayscale to RGB by replicating the channel
        #     X_upsampled[t, 0, :, :] = frame_resized
        #     X_upsampled[t, 1, :, :] = frame_resized
        #     X_upsampled[t, 2, :, :] = frame_resized
        
        # print(X_upsampled.shape)
        return X_upsampled
    
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
            shuffle=True, 
            num_workers=self.num_workers,
            drop_last=True
        )
        
        return dataloader
    

if __name__ == "__main__":
    data_path = "./mnist_data/mnist_test_seq.npy"
    batch_size = 32
    num_workers = 4
    split_mnist_data(data_path, 5)
        
    