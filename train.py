import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import time

from kitti_data_utils import KittiDataloader
from early_stopping import EarlyStopping
from prednet import Prednet

# DATA DIR
DATA_DIR = './kitti_data'

# Device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# Training parameters
nb_epoch = 150
batch_size = 4
samples_per_epoch = 500
N_seq_val = 100  # number of sequences to use for validation
num_workers = 1
patience = 5

# Model Checkpointing
num_save = 10
checkpoint_dir = './checkpoints'

# Model parameters
n_channels, im_height, im_width = (3, 128, 160)
input_shape = (batch_size, n_channels, im_height, im_width)
A_stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = A_stack_sizes
A_filter_sizes = (3, 3, 3)
Ahat_filter_sizes = (3, 3, 3, 3)
R_filter_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0.1, 0.1, 0.1]) # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = torch.tensor(np.expand_dims(layer_loss_weights, 1), device=device, dtype=torch.float32)
nt = 10  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones(nt)  # equally weight all timesteps except the first
time_loss_weights[0] = 0

# LR scheduler
lr_lambda = lambda epoch: 0.001 if epoch < 75 else 0.0001

def save_model(model, optimizer, epoch, train_error):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'train_error': train_error,
    }, model_path)
    
    print(f'saved model to: {model_path}')
    
def load_model(model, optimizer, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def train():
    # Create TensorBoard writer
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/prednet_training_{timestamp}"
    writer = SummaryWriter(log_dir)
    
    train_file = os.path.join(DATA_DIR, 'X_train.hkl')
    train_sources = os.path.join(DATA_DIR, 'sources_train.hkl')
    val_file = os.path.join(DATA_DIR, 'X_val.hkl')
    val_sources = os.path.join(DATA_DIR, 'sources_val.hkl')

    train_dataloader = KittiDataloader(
        data_path=train_file,
        source_path=train_sources,
        nt=nt,
        batch_size=batch_size, 
        sequence_start_mode='all',
        output_mode='error',
        num_workers=num_workers
    ).dataloader()
    
    val_dataloader = KittiDataloader(
        data_path=val_file,
        source_path=val_sources,
        nt=nt,
        batch_size=batch_size, 
        sequence_start_mode='all',
        output_mode='error',
        num_workers=num_workers
    ).dataloader()
    
    model = Prednet(
        A_stack_sizes=A_stack_sizes, 
        R_stack_sizes=R_stack_sizes, 
        A_filter_sizes=A_filter_sizes, 
        R_filter_sizes=R_filter_sizes, 
        Ahat_filter_sizes=Ahat_filter_sizes,
        pixel_max=1,
        lstm_activation='relu', 
        A_activation='relu', 
        extrap_time=None, 
        output_type='error',
        device=device
    )
    model.to(device=device)
    
    early_stopping = EarlyStopping(patience=patience)
    
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    train_error_list, val_error_list = [], []

    global_step = 1
    
    for epoch in range(1, nb_epoch+1):
        print(f'training epoch {epoch}')
        train_error, global_step = train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, input_shape, writer, global_step, epoch)
        val_error = val_one_epoch(val_dataloader, model, input_shape, writer, global_step, epoch)
        
        if epoch % num_save == 0:
            save_model(model, optimizer, epoch, train_error)
        
        train_error_list.append(train_error)
        val_error_list.append(val_error)
        
        writer.add_scalar('Loss/Train_Epoch', np.mean(train_error), epoch)
        writer.add_scalar('Loss/Val_Epoch', np.mean(val_error), epoch)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], epoch)
        
        print(f'Epoch {epoch:2d} | Train Error: {np.mean(train_error):3f} | Val Error: {np.mean(val_error):3f}')
        
        torch.cuda.empty_cache()
        
        early_stopping(val_loss=val_error)
        
        if early_stopping.early_stop:
            print('Stop training')
            break
    
    writer.close()
    print(f"Training completed. TensorBoard logs saved to: {log_dir}")


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, input_shape, writer, global_step, epoch):
    total_error = 0.0
    step_count = 0
    print(len(train_dataloader))
    for step, (frames, _) in enumerate(train_dataloader):
        print(f'training step {step}')
        initial_states = model.get_initial_states(input_shape)
        
        output_list = model(frames.to(device), initial_states)
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
        
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        
        writer.add_scalar('Loss/Train_Step', error.item(), global_step)
        writer.add_scalar('Loss/Train_Step_Avg', total_error.item() / (step + 1), global_step)
        writer.add_scalar('Learning_Rate_Step', optimizer.param_groups[0]['lr'], global_step)
        
        global_step += 1
        step_count += 1
        
        # Explicitly delete variables to free memory
        del initial_states, output_list, error
        torch.cuda.empty_cache()  # Clear GPU cache
        
    lr_scheduler.step()
        
    return total_error, global_step


def val_one_epoch(val_dataloader, model, input_shape, writer, global_step, epoch):
    total_error = 0.0
    step_count = 0
    
    with torch.no_grad():  # Disable gradient computation for validation
        for step, (frames, _) in enumerate(val_dataloader):
            # Create fresh initial states for each batch
            initial_states = model.get_initial_states(input_shape)
            
            output_list = model(frames.to(device), initial_states)
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
            step_count += 1
            
            writer.add_scalar('Loss/Val_Step', error.item(), global_step + step)
            writer.add_scalar('Loss/Val_Step_Avg', total_error.item() / step_count, global_step + step)
            
            del initial_states, output_list, error
            
    return total_error


if __name__ == '__main__':
    train()
    
    # n_channels = 3
    # img_height = 128
    # img_width  = 160

    # A_stack_sizes = (n_channels, 48, 96, 192)
    # R_stack_sizes = A_stack_sizes
    # A_filter_sizes = (3, 3, 3)
    # Ahat_filter_sizes = (3, 3, 3, 3)
    # R_filter_sizes = (3, 3, 3, 3)
    
    
    # prednet = Prednet(
    #     A_stack_sizes=A_stack_sizes, 
    #     R_stack_sizes=R_stack_sizes, 
    #     A_filter_sizes=A_filter_sizes, 
    #     R_filter_sizes=R_filter_sizes, 
    #     Ahat_filter_sizes=Ahat_filter_sizes,
    #     pixel_max=1,
    #     lstm_activation='relu', 
    #     A_activation='relu', 
    #     extrap_time=None, 
    #     output_type='all'
    # )
    # optimizer = optim.Adam(prednet.parameters())
    
    # save_model(prednet, optimizer, 99, 0)
    # model, optimizer = load_model(prednet, optimizer, model_path='./checkpoints/epoch_99.pth')
    # print(model)
