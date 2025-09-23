import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.mnist import MNISTDataloader, split_mnist_data
from prednet import Prednet

# MNIST Data paths
MNIST_DATA_DIR = './mnist_data/mnist_test_seq.npy'
PRED_DIR = './predictions'
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

n_samples = 10
batch_size = 1
# n_channels, im_height, im_width = (1, 64, 64)
# input_shape = (batch_size, n_channels, im_height, im_width)
# A_stack_sizes = (n_channels, 48, 96, 192)
# R_stack_sizes = A_stack_sizes
# A_filter_sizes = (3, 3, 3)
# Ahat_filter_sizes = (3, 3, 3, 3)
# R_filter_sizes = (3, 3, 3, 3)
nt = 5

n_channels, im_height, im_width = (1, 64, 64)
input_shape = (batch_size, n_channels, im_height, im_width)
A_stack_sizes = (n_channels, 24, 48)
R_stack_sizes = A_stack_sizes
A_filter_sizes = (3, 3)
Ahat_filter_sizes = (3, 3, 3)
R_filter_sizes = (3, 3, 3)

model_path = './checkpoints/epoch_58.pth'
# print(model_path)

def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def evaluate_dataset(data_split):
    train_path = './mnist_data/mnist_train.npy'
    val_path = './mnist_data/mnist_val.npy'
    
    # Choose the appropriate data file
    if data_split == 'train':
        data_file = train_path
    else:
        data_file = val_path
    
    output_dir = os.path.join(PRED_DIR, data_split)
    os.makedirs(output_dir, exist_ok=True)

    dataloader = MNISTDataloader(
        data_path=data_file,
        batch_size=batch_size, 
        num_workers=2
    ).dataloader()
    
    model = Prednet(
        A_stack_sizes=A_stack_sizes, 
        R_stack_sizes=R_stack_sizes, 
        A_filter_sizes=A_filter_sizes, 
        R_filter_sizes=R_filter_sizes, 
        Ahat_filter_sizes=Ahat_filter_sizes,
        pixel_max=1.0,
        lstm_activation='relu', 
        A_activation='relu', 
        extrap_time=None, 
        output_type='prediction',
        device=device
    )
    model.to(device=device)
    model = load_model(model, model_path)
    model.eval()
    
    samples_collected = 0
    
    with torch.no_grad():
        for batch_idx, frames in enumerate(dataloader):
            if samples_collected >= n_samples:
                break
                
            initial_states = model.get_initial_states(input_shape)
            frames_device = frames.to(device)
            
            predictions = model(frames_device, initial_states)
            original = frames.squeeze(0).cpu().numpy()  # (nt, c, h, w)
            predicted = torch.stack(predictions).cpu().numpy()  # (nt, batch, c, h, w)
            predicted = predicted.squeeze(1)  # Remove batch dimension: (nt, c, h, w)
            # print("LOL", predicted.sum(), original.sum())
            
            save_sequence_plot(original, predicted, output_dir, f'{data_split}_{samples_collected + 1}.png')
            samples_collected += 1
    
    print(f'Saved {samples_collected} predictions to {output_dir}')

def evaluate():
    evaluate_dataset('train')
    evaluate_dataset('val')

def save_sequence_plot(original, predicted, output_dir, filename):
    fig, axes = plt.subplots(2, nt, figsize=(nt * 2, 4))
    
    for t in range(nt):
        # Handle grayscale images (1, H, W) -> (H, W)
        orig_img = original[t, 0, :, :] if original.shape[1] == 1 else original[t].transpose(1, 2, 0)
        pred_img = predicted[t, 0, :, :] if predicted.shape[1] == 1 else predicted[t].transpose(1, 2, 0)

        # Images are already normalized to [0, 1], so just clip
        orig_img = np.clip(orig_img, 0, 1)
        pred_img = np.clip(pred_img, 0, 1)
        
        # Display images directly with [0, 1] range for better contrast
        axes[0, t].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
        axes[0, t].set_title(f'Original t={t}')
        axes[0, t].axis('off')
        
        axes[1, t].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
        axes[1, t].set_title(f'Predicted t={t}')
        axes[1, t].axis('off')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, filename), dpi=150, bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    evaluate()
