import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
import time
import logging
import wandb
import matplotlib.pyplot as plt

from utils import EarlyStopping
from prednet import Prednet
from utils.mnist import MNISTDataloader, split_custom_mnist_data
from utils.plot import plot_hidden_states_list
from utils.dataset_generator import MovingMnistDatasetGenerator

if os.path.exists('.env'):
    load_dotenv()
    wandb_key = os.getenv("WANDB_API_KEY")

# DATA DIR
# GENERATE DATASET
generator = MovingMnistDatasetGenerator(nt=5, h=128, w=128)
dataset = generator.generate_dataset(num_samples=1000, n_digits=5, max_scale=4)
MNIST_DATA_DIR = './custom_dataset/mnist_dataset_1000_5.npy'

# Device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# Training parameters
nb_epoch = 150
batch_size = 4
N_seq_val = 100  # number of sequences to use for validation
num_workers = 4
patience = 15
init_lr = 0.001
latter_lr = 0.0001  

# Model Checkpointing
num_save = 1
checkpoint_dir = './checkpoints'

# Model parameters
n_channels, im_height, im_width = (3, 128, 128)  # KITTI RGB images
input_shape = (batch_size, n_channels, im_height, im_width)
A_stack_sizes = (n_channels, 32, 64, 128, 256)  # Adapted for KITTI
R_stack_sizes = A_stack_sizes
A_filter_sizes = (3, 3, 3, 3)
Ahat_filter_sizes = (3, 3, 3, 3, 3)
R_filter_sizes = (3, 3, 3, 3, 3)
layer_loss_weights = np.array([1., .1, .1, .1, .1]) # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = torch.tensor(np.expand_dims(layer_loss_weights, 1), device=device, dtype=torch.float32)
nt = 5  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones(nt)  # equally weight all timesteps except the first
time_loss_weights[0] = 0

# LR scheduler
lr_lambda = lambda epoch: 1.0 if epoch < 75 else (latter_lr / init_lr)


def debug():
    writer = None
    # wandb.init(
    #     project="prednet-mnist",
    #     name=f"prednet_debug",
    #     config={
    #         "epochs": nb_epoch,
    #         "batch_size": batch_size,
    #         "learning_rate": init_lr,
    #         "patience": patience,
    #         "n_channels": n_channels,
    #         "im_height": im_height,
    #         "im_width": im_width,
    #         "A_stack_sizes": A_stack_sizes,
    #         "R_stack_sizes": R_stack_sizes,
    #         "A_filter_sizes": A_filter_sizes,
    #         "Ahat_filter_sizes": Ahat_filter_sizes,
    #         "R_filter_sizes": R_filter_sizes,
    #         "layer_loss_weights": layer_loss_weights.cpu().numpy().tolist(),
    #         "nt": nt,
    #         "device": device
    #     }
    # )
    train_path, val_path = split_custom_mnist_data(datapath=MNIST_DATA_DIR, nt=nt)

    train_dataloader = MNISTDataloader(
        data_path=train_path,
        batch_size=batch_size, 
        num_workers=num_workers
    ).dataloader(mnist_dataset_type="custom_mnist")
    
    val_dataloader = MNISTDataloader(
        data_path=val_path,
        batch_size=batch_size, 
        num_workers=num_workers
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
    
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    train_error_list, val_error_list = [], []

    global_step = 1
    
    for epoch in range(1, nb_epoch+1):
        train_error, global_step = train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, input_shape, writer, global_step, epoch)
        # val_error = val_one_epoch(val_dataloader, model, input_shape, writer, global_step, epoch)
        
        avg_train_error = train_error / len(train_dataloader)
        # avg_val_error = val_error / len(val_dataloader)
        
        train_error_list.append(avg_train_error)
        # val_error_list.append(avg_val_error)
        
        # wandb.log({
        #     "epoch": epoch,
        #     "train_error_epoch": avg_train_error,
        #     # "val_error_epoch": avg_val_error,   
        #     "learning_rate_epoch": optimizer.param_groups[0]['lr']
        # }, step=global_step - 1)
        
        # print(f'Epoch: {epoch} global step: {global_step - 1} | Train Error: {avg_train_error:3f} | Val Error: {avg_val_error:3f}')
        
        torch.cuda.empty_cache()
        
    # wandb.finish()


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, input_shape, writer, global_step, epoch):
    total_error = 0.0
    print(f'Starting epoch: {epoch}')
    
    for step, frames in enumerate(train_dataloader, start=1):
        # if step > 30: break
        
        initial_states = model.get_initial_states(input_shape)
        
        output_list, hidden_states_list = model(frames.to(device), initial_states)
        error = 0.0
        
        for t, output in enumerate(output_list):
            # print(output)
            # print(torch.matmul(output, layer_loss_weights))
            # return
            # assert output.size() == (batch_size, model.nb_layers), 'wrong size for output'
            # print(output.size(), layer_loss_weights.size(), output.dtype, layer_loss_weights.dtype)
            weighted_layer_error = torch.matmul(output, layer_loss_weights)
            # print(weighted_layer_error.size())
            # assert weighted_layer_error.size() == (batch_size, 1), 'wrong size for weighted layer output'
            # print(time_loss_weights[t], torch.mean(weighted_layer_error))
            error += time_loss_weights[t] * torch.mean(weighted_layer_error)
        
        total_error += error
        # print(model.conv_layers['Ahat'][0].weight)
        # print(model.conv_layers['Ahat'][0].weight.grad)
        # print(model.conv_layers['Ahat'][0].weight.grad.shape)
        # return 
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        
        # Log metrics
        avg_error_so_far = total_error / step
    
        print(f"Epoch: {epoch} step: {step} | Train Error: {error:3f} | Train Avg Error: {avg_error_so_far:3f}")
        # wandb.log({
        #     "train_step_error": error.item(),
        #     "train_step_avg_error": avg_error_so_far.item(),
        #     "learning_rate_step": optimizer.param_groups[0]['lr']
        # }, step=global_step)
        
        if global_step % 500 == 0:
            plot_hidden_states_list(hidden_states_list, frames, global_step)
        
        global_step += 1
        
        del initial_states, output_list, error, hidden_states_list
        torch.cuda.empty_cache()  # Clear GPU cache
        
    lr_scheduler.step()
        
    return total_error, global_step


def val_one_epoch(val_dataloader, model, input_shape, writer, global_step, epoch):
    total_error = 0.0
    print('Starting validation')
    
    with torch.no_grad(): 
        for step, frames in enumerate(val_dataloader, start=1):
            # if step > 3: break
            initial_states = model.get_initial_states(input_shape)
            
            output_list, hidden_states_list = model(frames.to(device), initial_states)
            error = 0.0
            
            for t, output in enumerate(output_list):
                # assert output.size() == (batch_size, model.nb_layers), 'wrong size for output'
                # print(output.size(), layer_loss_weights.size(), output.dtype, layer_loss_weights.dtype)
                weighted_layer_error = torch.matmul(output, layer_loss_weights)
                # print(weighted_layer_error.size())
                # assert weighted_layer_error.size() == (batch_size, 1), 'wrong size for weighted layer output'
                # print(time_loss_weights[t], torch.mean(weighted_layer_error))
                error += time_loss_weights[t] * torch.mean(weighted_layer_error)
            
            total_error += error
            
            del initial_states, output_list, error, hidden_states_list
            
    return total_error


if __name__ == '__main__':
    debug()
