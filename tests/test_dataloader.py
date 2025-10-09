from utils.mnist import split_mnist_data, MNISTDataloader
from utils.kitti import KittiDataset, KittiDataloader
import matplotlib.pyplot as plt
import torch
import os


def plot_input_image(images, output_dir, batch_size, nt):
    parent_dir = './tests/data'
    # images of size (batch_size, nt, 1, h, w)
    # we want batch_size number of plots where each plot contains nt images in a row
    for i in range(batch_size):
        fig, axes = plt.subplots(1, nt, figsize=(nt * 3, 3))
        for j in range(nt):
            image = images[i, j, 0, :, :]
            axes[j].imshow(image, cmap='gray', vmin=0, vmax=1)
            axes[j].set_title(f't={j}')
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(parent_dir, f"input_image_{i}.png"), dpi=150, bbox_inches='tight')
        plt.close()
        
def plot_input_image_kitti(images, output_dir, batch_size, nt):
    parent_dir = './tests/data'
    os.makedirs(parent_dir, exist_ok=True)
    
    # images of size (batch_size, nt, 3, h, w) for RGB
    # we want batch_size number of plots where each plot contains nt images in a row
    for i in range(batch_size):
        fig, axes = plt.subplots(1, nt, figsize=(nt * 3, 3))
        for j in range(nt):
            # Extract RGB image and transpose from (3, h, w) to (h, w, 3)
            image = images[i, j, :, :, :].transpose(1, 2, 0)  # (h, w, 3)
            axes[j].imshow(image, vmin=0, vmax=1)
            axes[j].set_title(f't={j}')
            axes[j].axis('off')
        plt.tight_layout()
        plt.savefig(os.path.join(parent_dir, f"kitti_input_image_{i}.png"), dpi=150, bbox_inches='tight')
        plt.close()

def test_split_mnist_data():
    train_path, val_path = split_mnist_data(datapath='./mnist_data/mnist_test_seq.npy', nt=2)
    nt = 2
    assert train_path is not None
    assert val_path is not None
    
    train_dataloader = MNISTDataloader(
        data_path=train_path,
        batch_size=1,
        num_workers=4
    ).dataloader()

    val_dataloader = MNISTDataloader(
        data_path=val_path,
        batch_size=1,
        num_workers=4
    ).dataloader()

    for _, frames in enumerate(train_dataloader):
        images = frames.cpu().numpy()  # (batch, nt, c, h, w)
        print(images.shape)
        plot_input_image(images, "train", 4, nt)
        break
        
    # for _, frames in enumerate(val_dataloader):
    #     images = frames.cpu().numpy()  # (batch, nt, c, h, w)
    #     plot_input_image(images, "val", 4, nt)

def test_kitti_dataloader():
    nt = 5
    train_dataloader = KittiDataloader(
        data_path='./kitti_data/X_train.hkl',
        source_path='./kitti_data/sources_train.hkl',
        nt=nt,
        batch_size=4,
        sequence_start_mode='all',
        output_mode='error',
        num_workers=4
    ).dataloader()
    
    for _, frames in enumerate(train_dataloader):
        images = frames.cpu().numpy()  # (batch, nt, c, h, w)
        print(images.shape)
        plot_input_image_kitti(images, "kitti_train", 4, nt)
        break 
    
if __name__ == "__main__":
    test_split_mnist_data()
    # test_kitti_dataloader()