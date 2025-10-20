import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.mnist import MNISTDataloader
from utils.plot import plot_hidden_states_list
from prednet import Prednet

# MNIST Data paths
MNIST_DATA_DIR = './custom_dataset/mnist_dataset_4000_5.npy'
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

n_samples = 10
batch_size = 8
nt = 5

n_channels, im_height, im_width = (3, 128, 128)  # RGB images
input_shape = (batch_size, n_channels, im_height, im_width)
A_stack_sizes = (n_channels, 32, 64, 128, 256)
R_stack_sizes = A_stack_sizes
A_filter_sizes = (3, 3, 3, 3)
Ahat_filter_sizes = (3, 3, 3, 3, 3)
R_filter_sizes = (3, 3, 3, 3, 3)

model_path = './checkpoints/epoch_400.pth'  # TODO: Update to appropriate checkpoint

def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def evaluate_dataset(data_split):
    train_path, val_path = "./custom_dataset/mnist_train.npy", "./custom_dataset/mnist_val.npy"
    
    # Choose the appropriate data file
    if data_split == 'train':
        data_file = train_path
    else:
        data_file = val_path

    dataloader = MNISTDataloader(
        data_path=data_file,
        batch_size=batch_size, 
        num_workers=2
    ).dataloader(mnist_dataset_type="custom_mnist")
    
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
        output_type='error',
        device=device
    )
    model.to(device=device)
    model = load_model(model, model_path)
    model.eval()
    
    samples_collected = 0
    
    with torch.no_grad(): 
        for step, frames in enumerate(dataloader, start=1):
            # if step > 3: break
            initial_states = model.get_initial_states(input_shape)
            
            output_list, hidden_states_list = model(frames.to(device), initial_states)
            
        
            plot_hidden_states_list(hidden_states_list, frames, step, 'val')
            
            del initial_states, output_list, hidden_states_list
            
def evaluate():
    evaluate_dataset('train')
    evaluate_dataset('val')

if __name__ == '__main__':
    evaluate()
