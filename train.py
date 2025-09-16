import os
import numpy as np
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from dotenv import load_dotenv
import time
import logging
import wandb

from utils import KittiDataloader
from utils import EarlyStopping
from prednet import Prednet
from utils.mnist import MNISTDataloader, split_mnist_data

if os.path.exists('.env'):
    load_dotenv()
    wandb_key = os.getenv("WANDB_API_KEY")

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# DATA DIR
MNIST_DATA_DIR = './mnist_data/mnist_test_seq.npy'

# Device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# Training parameters
nb_epoch = 150
batch_size = 4
N_seq_val = 100  # number of sequences to use for validation
num_workers = 2
patience = 5
init_lr = 0.001
latter_lr = 0.0001

# Model Checkpointing
num_save = 1
checkpoint_dir = './checkpoints'

# Model parameters
n_channels, im_height, im_width = (1, 64, 64)
input_shape = (batch_size, n_channels, im_height, im_width)
A_stack_sizes = (n_channels, 48, 96, 192)
R_stack_sizes = A_stack_sizes
A_filter_sizes = (3, 3, 3)
Ahat_filter_sizes = (3, 3, 3, 3)
R_filter_sizes = (3, 3, 3, 3)
layer_loss_weights = np.array([1., 0.1, 0.1, 0.1]) # weighting for each layer in final loss; "L_0" model:  [1, 0, 0, 0], "L_all": [1, 0.1, 0.1, 0.1]
layer_loss_weights = torch.tensor(np.expand_dims(layer_loss_weights, 1), device=device, dtype=torch.float32)
nt = 5  # number of timesteps used for sequences in training
time_loss_weights = 1./ (nt - 1) * np.ones(nt)  # equally weight all timesteps except the first
time_loss_weights[0] = 0

# LR scheduler
lr_lambda = lambda epoch: 1.0 if epoch < 75 else (latter_lr / init_lr)

def save_model(model, optimizer, epoch, avg_train_error):
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_train_error': avg_train_error,
    }, model_path)
    
    print(f'Saved model to: {model_path}')
    
def load_model(model, optimizer, model_path):
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return model, optimizer

def train():
    # Create TensorBoard and WandB writers
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    log_dir = f"runs/prednet_training_{timestamp}"
    writer = SummaryWriter(log_dir)
    
    wandb.init(
        project="prednet-kitti",
        name=f"prednet_training_{timestamp}",
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
    train_path, val_path = split_mnist_data(datapath=MNIST_DATA_DIR, nt=nt)
    train_file = os.path.join(train_path)
    val_file = os.path.join(val_path)

    train_dataloader = MNISTDataloader(
        data_path=train_file,
        batch_size=batch_size, 
        num_workers=num_workers
    ).dataloader()
    
    val_dataloader = MNISTDataloader(
        data_path=val_file,
        batch_size=batch_size, 
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
    
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    
    # model, optimizer = load_model(model, optimizer, './checkpoints/epoch_19.pth')
    
    lr_scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)
    train_error_list, val_error_list = [], []

    global_step = 1
    
    for epoch in range(1, nb_epoch+1):
        train_error, global_step = train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, input_shape, writer, global_step, epoch)
        val_error = val_one_epoch(val_dataloader, model, input_shape, writer, global_step, epoch)
        
        avg_train_error = train_error / len(train_dataloader)
        avg_val_error = val_error / len(val_dataloader)
        
        train_error_list.append(avg_train_error)
        val_error_list.append(avg_val_error)
        
        # Log epoch-level metrics
        writer.add_scalar('Loss/Train_Epoch', avg_train_error, global_step - 1)
        writer.add_scalar('Loss/Val_Epoch', avg_val_error, global_step - 1)
        writer.add_scalar('Learning_Rate', optimizer.param_groups[0]['lr'], global_step - 1)
        wandb.log({
            "epoch": epoch,
            "train_error_epoch": avg_train_error,
            "val_error_epoch": avg_val_error,
            "learning_rate_epoch": optimizer.param_groups[0]['lr']
        }, step=global_step - 1)
        
        print(f'Epoch: {epoch} global step: {global_step - 1} | Train Error: {avg_train_error:3f} | Val Error: {avg_val_error:3f}')
        
        # save model
        if epoch % num_save == 0:
            save_model(model, optimizer, epoch, avg_train_error)
        
        torch.cuda.empty_cache()
        
        early_stopping(val_loss=avg_val_error)
        
        if early_stopping.early_stop:
            print('Early stopping triggered - stopping training')
            break
    
    writer.close()
    wandb.finish()
    print(f"Training completed. TensorBoard logs saved to: {log_dir}")
    print("Wandb logging completed")


def train_one_epoch(train_dataloader, model, optimizer, lr_scheduler, input_shape, writer, global_step, epoch):
    total_error = 0.0
    print(f'Starting epoch: {epoch}')
    
    for step, frames in enumerate(train_dataloader, start=1):
        # if step > 30: break
        print(f"Epoch: {epoch} step: {step} global step: {global_step}")
        
        initial_states = model.get_initial_states(input_shape)
        
        output_list = model(frames.to(device), initial_states)
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
        
        optimizer.zero_grad()
        error.backward()
        optimizer.step()
        
        # Log metrics
        avg_error_so_far = total_error / step
        writer.add_scalar('Loss/Train_Step_Avg', avg_error_so_far.item(), global_step)
        writer.add_scalar('Learning_Rate_Step', optimizer.param_groups[0]['lr'], global_step)
    
        wandb.log({
            "train_step_error": error.item(),
            "train_step_avg_error": avg_error_so_far.item(),
            "learning_rate_step": optimizer.param_groups[0]['lr']
        }, step=global_step)
        
        # if global_step % num_save == 0 or step == len(train_dataloader)-1:
        #     save_model(model, optimizer, epoch, global_step, avg_error_so_far.item())
        
        global_step += 1
        
        del initial_states, output_list, error
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
