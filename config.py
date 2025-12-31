"""
Configuration file for PredNet training and evaluation.
Contains all hyperparameters, model architecture settings, and paths.
"""

import numpy as np
import torch

# Device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# Training name
training_suffix = 'finetuning_mnist_disrupt'

# Directory paths with training suffix
checkpoint_dir = f'./checkpoints_{training_suffix}'
debug_images_dir = f'./debug_layer_images_{training_suffix}'

# Dataset parameters
n_digits = 3
num_samples = 3840
min_scale = 1.0
max_scale = 2.3
num_dilate_iterations = 1

# Data paths
mnist_raw_path = "./data/multi_digit_10000_16_d2-5.npy"  # Raw MNIST sequence data (will be split into train/val)
adapt_parent_path = "./data/adapt" 
# train_path = "./data/mnist_train_multi.npy"  # Created by split_mnist_data
# val_path = "./data/mnist_val_multi.npy"  # Created by split_mnist_data
train_path = "./data/mnist_train.npy"  # Created by split_mnist_data
val_path = "./data/mnist_val.npy"  # Created by split_mnist_data

disrupt_sudden_appear_path = "./data/adapt/sudden_appear_2000_16_t8.npy"
disrupt_sudden_transform_path = "./data/adapt/sudden_transform_2000_16_t8.npy"
disrupt_sudden_disappear_path = "./data/adapt/sudden_disappear_2000_16_t8.npy"
disrupt_base_path = "./data/adapt/mnist_val_disrupt_base.npy"

# Training parameters
nb_epoch = 1000
batch_size = 8
num_workers = 4
patience = 15
init_lr = 1e-4  # Lower LR for finetuning (was 5e-3, too high)
latter_lr = 1e-5  # Lower final LR (was 5e-4)

# Model Checkpointing
num_save = 100
num_plot = 25

# Model parameters
n_channels, im_height, im_width = (3, 128, 160)  # RGB images (128x160 for moving MNIST)
input_shape = (batch_size, n_channels, im_height, im_width)
# Architecture must match pretrained KITTI weights: 4 layers with (3, 48, 96, 192) channels
A_stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = A_stack_sizes
A_filter_sizes = (3, 3, 3)
Ahat_filter_sizes = (3, 3, 3, 3)
R_filter_sizes = (3, 3, 3, 3)

# Model training parameters
nt = 16  # number of timesteps used for sequences in training
pixel_max = 1.0
lstm_activation = 'relu'
A_activation = 'relu'

layer_loss_weights_array = np.array([1., .1, .1, .1])  # weighting for each layer in final loss
# layer_loss_weights_array = np.array([1., 0, 0, 0])
time_loss_weights = 1./ (nt - 1) * np.ones(nt)  # equally weight all timesteps except the first
time_loss_weights[0] = 0

def get_layer_loss_weights(device):
    return torch.tensor(np.expand_dims(layer_loss_weights_array, 1), device=device, dtype=torch.float32)

def get_lr_lambda(init_lr, latter_lr):
    decay_epoch = 500
    return lambda epoch: 1.0 if epoch < decay_epoch else (latter_lr / init_lr)

# Evaluation parameters
n_samples_eval = 10
# model_path = './pretrained/preTrained_weights_forPyTorch.pkl'  # Use pretrained weights for evaluation
model_path = './checkpoints_finetuning_mnist_3/epoch_best_val.pth'

# Finetuning
is_finetuning = True
extrap_time = 8
# Use pretrained KITTI weights for finetuning on new data
# pretrained_weights_path = './pretrained/preTrained_weights_forPyTorch.pkl'
pretrained_weights_path = './checkpoints_finetuning_mnist_3/epoch_best_val.pth'
model_checkpoint_path = pretrained_weights_path  # Path to checkpoint/pretrained weights for finetuning  


