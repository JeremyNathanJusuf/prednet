"""
Configuration file for PredNet training and evaluation.
Contains all hyperparameters, model architecture settings, and paths.
"""

import numpy as np
import torch

# Device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# Dataset parameters
n_digits = 3
num_samples = 3840
min_scale = 1.0
max_scale = 2.3
num_dilate_iterations = 1

# Data paths
train_path = "./custom_dataset/mnist_train.npy"
val_path = "./custom_dataset/mnist_val.npy"

# Training parameters
nb_epoch = 1000
batch_size = 8
num_workers = 4
patience = 15
init_lr = 6e-4
latter_lr = 5e-4

# Model Checkpointing
num_save = 100
num_plot = 25
checkpoint_dir = './checkpoints'

# Model parameters
n_channels, im_height, im_width = (3, 64, 64)  # RGB images
input_shape = (batch_size, n_channels, im_height, im_width)
A_stack_sizes = (n_channels, 32, 64, 128, 256)
R_stack_sizes = A_stack_sizes
A_filter_sizes = (3, 3, 3, 3)
Ahat_filter_sizes = (3, 3, 3, 3, 3)
R_filter_sizes = (3, 3, 3, 3, 3)

# Model training parameters
nt = 8  # number of timesteps used for sequences in training
pixel_max = 1.0
lstm_activation = 'relu'
A_activation = 'relu'

# Loss weights
# layer_loss_weights_array = np.array([1., .1, .1, .1, .1])  # weighting for each layer in final loss
layer_loss_weights_array = np.array([1., 0, 0, 0, 0])
time_loss_weights = 1./ (nt - 1) * np.ones(nt)  # equally weight all timesteps except the first
time_loss_weights[0] = 0

# Convert layer_loss_weights to torch tensor (will be moved to device in train.py)
def get_layer_loss_weights(device):
    """Returns layer loss weights as a torch tensor on the specified device."""
    return torch.tensor(np.expand_dims(layer_loss_weights_array, 1), device=device, dtype=torch.float32)

# LR scheduler
def get_lr_lambda(init_lr, latter_lr):
    """Returns learning rate lambda function for scheduler."""
    return lambda epoch: 1.0 if epoch < 4000 else (latter_lr / init_lr)

# Evaluation parameters
n_samples_eval = 10
model_path = './checkpoints_14_nov_1/epoch_100.pth'  # Default model checkpoint path

# Finetuning
is_finetuning = True
extrap_time = 4
model_checkpoint_path = './checkpoints_14_nov_1/epoch_100.pth'  

