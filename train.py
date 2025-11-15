import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
import time
import logging
import wandb
import random
import matplotlib.pyplot as plt

from utils import EarlyStopping
from prednet import Prednet
from utils.mnist import MNISTDataloader, split_custom_mnist_data
from utils.plot import plot_hidden_states_list
from utils.dataset_generator import MovingMnistDatasetGenerator
import config

if os.path.exists('.env'):
    load_dotenv()
    wandb_key = os.getenv("WANDB_API_KEY")

# Import parameters from config
device = config.device
n_digits = config.n_digits
num_samples = config.num_samples
min_scale = config.min_scale
max_scale = config.max_scale
num_dilate_iterations = config.num_dilate_iterations

nb_epoch = config.nb_epoch
batch_size = config.batch_size
num_workers = config.num_workers
patience = config.patience
init_lr = config.init_lr
latter_lr = config.latter_lr

num_save = config.num_save
num_plot = config.num_plot
checkpoint_dir = config.checkpoint_dir
debug_images_dir = config.debug_images_dir

n_channels = config.n_channels
im_height = config.im_height
im_width = config.im_width
input_shape = config.input_shape
A_stack_sizes = config.A_stack_sizes
R_stack_sizes = config.R_stack_sizes
A_filter_sizes = config.A_filter_sizes
Ahat_filter_sizes = config.Ahat_filter_sizes
R_filter_sizes = config.R_filter_sizes
nt = config.nt
pixel_max = config.pixel_max
lstm_activation = config.lstm_activation
A_activation = config.A_activation

is_finetuning = config.is_finetuning
extrap_time = config.extrap_time

layer_loss_weights = config.get_layer_loss_weights(device)
time_loss_weights = config.time_loss_weights
model_checkpoint_path = config.model_checkpoint_path

lr_lambda = config.get_lr_lambda(init_lr, latter_lr)
# LR scheduler parameters
# lr_scheduler_factor = 0.5  # Factor by which to reduce LR
# lr_scheduler_patience = 5  # Number of epochs with no improvement after which LR will be reduced
# lr_scheduler_min_lr = 1e-6  # Minimum learning rate

def save_model(model, optimizer, epoch, avg_train_error, avg_val_error=None):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
    
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_train_error': avg_train_error,
    }
    
    if avg_val_error is not None:
        checkpoint_data['avg_val_error'] = avg_val_error
    
    torch.save(checkpoint_data, model_path)
    
    print(f'Saved model to: {model_path}')

def save_best_val_model(model, optimizer, epoch, avg_train_error, avg_val_error):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, 'epoch_best_val.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_train_error': avg_train_error,
        'avg_val_error': avg_val_error,
    }, model_path)
    
    print(f'Saved best validation model to: {model_path} (epoch {epoch}, val_error: {avg_val_error:.6f})')

def load_model(model, optimizer, lr_scheduler, model_path):
    checkpoint = torch.load(model_path)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    # ReduceLROnPlateau doesn't have last_epoch attribute like other schedulers
    # It tracks metrics internally, so we don't need to manually step it
    if hasattr(lr_scheduler, 'last_epoch'):
        lr_scheduler.last_epoch = checkpoint["epoch"] - 1
        lr_scheduler.step()
    
    return model, optimizer, lr_scheduler

def train():
    wandb.init(
        project="prednet-mnist",
        name=f"prednet_debug",
        config={
            "epochs": nb_epoch,
            "batch_size": batch_size,
            "learning_rate": init_lr,
            "patience": patience,
            "n_channels": n_channels,
            "im_height": im_height,
            "im_width": im_width,
            "A_stack_sizes": A_stack_sizes,
            "R_stack_sizes": R_stack_sizes,
            "A_filter_sizes": A_filter_sizes,
            "Ahat_filter_sizes": Ahat_filter_sizes,
            "R_filter_sizes": R_filter_sizes,
            "layer_loss_weights": layer_loss_weights.cpu().numpy().tolist(),
            "nt": nt,
            "device": device
        }
    )
    # GENERATE AND SPLIT DATASET
    generator = MovingMnistDatasetGenerator(nt=nt, h=im_height, w=im_width)
    dataset, dataset_path = generator.generate_dataset(
        num_samples=num_samples, 
        n_digits=n_digits, 
        min_scale=min_scale, 
        max_scale=max_scale, 
        num_dilate_iterations=num_dilate_iterations,
    )
    
    train_path, val_path = split_custom_mnist_data(datapath=dataset_path, nt=nt)

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
    
    # LOAD MODEL
    model = Prednet(
        A_stack_sizes=A_stack_sizes, 
        R_stack_sizes=R_stack_sizes, 
        A_filter_sizes=A_filter_sizes, 
        R_filter_sizes=R_filter_sizes, 
        Ahat_filter_sizes=Ahat_filter_sizes,
        pixel_max=pixel_max,
        lstm_activation=lstm_activation, 
        A_activation=A_activation, 
        extrap_time=extrap_time, 
        output_type='error',
        device=device
    )
    model.to(device=device)
    
    early_stopping = EarlyStopping(patience=patience)
    
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    # lr_scheduler = optim.lr_scheduler.ReduceLROnPlateau(
    #     optimizer, 
    #     mode='min',
    #     factor=lr_scheduler_factor,
    #     patience=lr_scheduler_patience,
    #     min_lr=lr_scheduler_min_lr
    # )
    
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    
    if is_finetuning:
        model, optimizer, lr_scheduler = load_model(model, optimizer, lr_scheduler, model_checkpoint_path)

    global_step = 1
    # global_step = 300*96+1
    
    best_val_error = float('inf')
    
    for epoch in range(1, nb_epoch+1):
    # for epoch in range(301, nb_epoch+1):
        train_error, global_step = train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, input_shape, global_step, epoch)
        val_error = val_one_epoch(val_dataloader, model, input_shape, global_step, epoch)
        
        avg_train_error = train_error / len(train_dataloader) # average of cumulative error
        avg_val_error = val_error / len(val_dataloader) # average of cumulative error
        
        # Save best validation model
        if avg_val_error < best_val_error:
            best_val_error = avg_val_error
            save_best_val_model(model, optimizer, epoch, avg_train_error, avg_val_error)
        
        wandb.log({
            "epoch": epoch,
            "train_error_epoch": avg_train_error,
            "val_error_epoch": avg_val_error,
            "best_val_error": best_val_error,
            "learning_rate_epoch": optimizer.param_groups[0]['lr']
        }, step=global_step - 1)
        
        print(f'Epoch: {epoch} global step: {global_step - 1} | Train Error: {avg_train_error:3f} | Val Error: {avg_val_error:3f} | Learning Rate: {optimizer.param_groups[0]["lr"]:.6f}')
        
        # Save model every num_save epochs or at the end
        if epoch % num_save == 0 or epoch == nb_epoch:
            save_model(model, optimizer, epoch, avg_train_error, avg_val_error)
        
        # Step the learning rate scheduler with validation error
        # lr_scheduler.step(avg_val_error)
        
        lr_scheduler.step()
        
        torch.cuda.empty_cache()
        
        # TODO: Uncomment this when we want to use early stopping
        early_stopping(val_loss=avg_val_error)
        
        if early_stopping.early_stop:
            print('Early stopping triggered - stopping training')
            break
        
    wandb.finish()


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, input_shape, global_step, epoch):
    total_error = 0.0
    print(f'Starting epoch: {epoch}')
    
    random_plot_step = random.randint(1, len(train_dataloader))

    for step, frames in enumerate(train_dataloader, start=1):
        initial_states = model.get_initial_states(input_shape)
        
        output_list, hidden_states_list = model(frames.to(device), initial_states)
        error = 0.0
        
        for t, output in enumerate(output_list):
            weighted_layer_error = torch.matmul(output, layer_loss_weights)
            error += time_loss_weights[t] * torch.mean(weighted_layer_error)
        
        total_error += error
        
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        
        # Log metrics
        avg_error_so_far = total_error / step
    
        print(f"Epoch: {epoch} step: {step} | Train Error: {error:3f} | Train Avg Error: {avg_error_so_far:3f}")
        wandb.log({
            "train_step_error": error.item(),
            "train_step_avg_error": avg_error_so_far.item(),
            "learning_rate_step": optimizer.param_groups[0]['lr']
        }, step=global_step)
        
        if epoch % num_plot == 0 and step == random_plot_step:
            plot_hidden_states_list(hidden_states_list, frames, epoch, 'train', debug_images_dir)
        
        global_step += 1
        
        del initial_states, output_list, error, hidden_states_list
        torch.cuda.empty_cache()  # Clear GPU cache
        
    return total_error, global_step


def val_one_epoch(val_dataloader, model, input_shape, global_step, epoch):
    total_error = 0.0
    print('Starting validation')
    
    random_plot_step = random.randint(1, len(val_dataloader))
    
    with torch.no_grad(): 
        for step, frames in enumerate(val_dataloader, start=1):
            initial_states = model.get_initial_states(input_shape)
            
            output_list, hidden_states_list = model(frames.to(device), initial_states)
            error = 0.0
            
            for t, output in enumerate(output_list):
                weighted_layer_error = torch.matmul(output, layer_loss_weights)
                error += time_loss_weights[t] * torch.mean(weighted_layer_error)
            
            total_error += error
            
            if epoch % num_plot == 0 and step == random_plot_step:
                plot_hidden_states_list(hidden_states_list, frames, epoch, 'val', debug_images_dir)
            
            del initial_states, output_list, error, hidden_states_list
            torch.cuda.empty_cache()  # Clear GPU cache
            
    return total_error


if __name__ == '__main__':
    train()
