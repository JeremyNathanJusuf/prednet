import os
import numpy as np
import torch
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

from prednet import Prednet
from utils.mnist import MNISTDataloader, split_mnist_data

# Device
device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

# Debug training parameters
nb_epoch = 40
batch_size = 64
num_workers = 4
init_lr = 0.001

# Data preprocessing
background_level = 0.0  # Subtle gray background (0.0=black, 0.05-0.1=subtle gray)
                         # Helps with gradient flow through ReLU in dark regions

# Darkness penalty configuration
# ============================================
# EXPERIMENT CONFIGS - Try different combinations:
# ============================================

# Current working config (MSE + Brightness)
use_mse_loss = True  # Add MSE reconstruction loss (most effective)
mse_weight = 3.0  # Weight for MSE loss (increased to dominate)

use_brightness_penalty = True  # Penalize low brightness
brightness_weight = 2.0  # Weight for brightness penalty (MUCH higher - same as MSE)

use_gradient_loss = False  # Match gradient magnitude (edges)
gradient_weight = 0.5  # Weight for gradient loss

# ============================================
# TO TEST: "Does brightness penalty help or is MSE enough?"
# Try: use_brightness_penalty = False, keep use_mse_loss = True
# ============================================

# ============================================
# TO TEST: "Should we use MSE for PredNet errors?"
# Go to prednet.py line 200-210 and uncomment Option 2
# ============================================

# Model Checkpointing
checkpoint_dir = './debug_checkpoints'
debug_images_dir = './debug_images'

# Model parameters (same as train.py)
n_channels, im_height, im_width = (1, 64, 64)
input_shape = (batch_size, n_channels, im_height, im_width)
A_stack_sizes = (n_channels, 48, 96)
R_stack_sizes = A_stack_sizes
A_filter_sizes = (3, 3)
Ahat_filter_sizes = (3, 3, 3)
R_filter_sizes = (3, 3, 3)
layer_loss_weights = np.array([1., .1, .1])
layer_loss_weights = torch.tensor(np.expand_dims(layer_loss_weights, 1), device=device, dtype=torch.float32)
nt = 5
time_loss_weights = 1./ (nt - 1) * np.ones(nt)
time_loss_weights[0] = 0

# MNIST Data paths
MNIST_DATA_DIR = './mnist_data/mnist_test_seq.npy'


def compute_additional_losses(predictions, targets):
    """
    Compute additional losses to combat dark outputs.
    
    Args:
        predictions: List of predicted frames (tensors)
        targets: Ground truth frames (batch, nt, c, h, w)
    
    Returns:
        Dictionary of additional losses
    """
    losses = {}
    
    # Stack predictions: (nt, batch, c, h, w)
    pred_stack = torch.stack(predictions)
    
    # Transpose targets to match: (batch, nt, c, h, w) -> (nt, batch, c, h, w)
    targets_t = targets.transpose(0, 1)
    
    # 1. MSE Loss (most common and effective)
    if use_mse_loss:
        mse_loss = torch.nn.functional.mse_loss(pred_stack, targets_t)
        losses['mse'] = mse_weight * mse_loss
    
    # 2. Brightness Penalty (penalize predictions darker than targets)
    if use_brightness_penalty:
        pred_mean = pred_stack.mean()
        target_mean = targets_t.mean()
        # Only penalize if prediction is darker
        brightness_loss = torch.nn.functional.relu(target_mean - pred_mean)
        losses['brightness'] = brightness_weight * brightness_loss
    
    # 3. Gradient Magnitude Loss (ensure edges are preserved)
    if use_gradient_loss:
        # Compute gradients using Sobel-like filters
        pred_grad_x = pred_stack[:, :, :, :, 1:] - pred_stack[:, :, :, :, :-1]
        pred_grad_y = pred_stack[:, :, :, 1:, :] - pred_stack[:, :, :, :-1, :]
        target_grad_x = targets_t[:, :, :, :, 1:] - targets_t[:, :, :, :, :-1]
        target_grad_y = targets_t[:, :, :, 1:, :] - targets_t[:, :, :, :-1, :]
        
        grad_loss = (torch.nn.functional.mse_loss(pred_grad_x, target_grad_x) + 
                     torch.nn.functional.mse_loss(pred_grad_y, target_grad_y))
        losses['gradient'] = gradient_weight * grad_loss
    
    return losses


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
    return model_path


def train_debug():
    """Train for 20 epochs with batch size 32"""
    print(f"\n{'='*60}")
    print(f"DEBUG TRAINING: {nb_epoch} epochs with batch size {batch_size}")
    print(f"{'='*60}\n")
    
    # Prepare data
    train_path, val_path = split_mnist_data(datapath=MNIST_DATA_DIR, nt=nt)
    
    train_dataloader = MNISTDataloader(
        data_path=train_path,
        batch_size=batch_size, 
        num_workers=num_workers,
        background_level=background_level
    ).dataloader()
    
    val_dataloader = MNISTDataloader(
        data_path=val_path,
        batch_size=batch_size, 
        num_workers=num_workers,
        background_level=background_level
    ).dataloader()
    
    # Initialize model
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
        output_type='both',  # Return both predictions and errors!
        device=device
    )
    model.to(device=device)
    
    optimizer = optim.Adam(model.parameters(), lr=init_lr)
    
    model_path = None
    
    # Training loop
    for epoch in range(1, nb_epoch + 1):
        print(f'\n--- Epoch {epoch}/{nb_epoch} ---')
        train_error = train_one_epoch(train_dataloader, model, optimizer, input_shape, epoch)
        val_error = val_one_epoch(val_dataloader, model, input_shape, epoch)
        
        avg_train_error = train_error / len(train_dataloader)
        avg_val_error = val_error / len(val_dataloader)
        
        print(f'Epoch {epoch} | Train Error: {avg_train_error:.6f} | Val Error: {avg_val_error:.6f}')
        
        # Save model after each epoch
        model_path = save_model(model, optimizer, epoch, avg_train_error)
        
        torch.cuda.empty_cache()
    
    print(f"\n{'='*60}")
    print("DEBUG TRAINING COMPLETE")
    print(f"{'='*60}\n")
    
    return model_path


def train_one_epoch(train_dataloader, model, optimizer, input_shape, epoch):
    model.train()
    total_error = 0.0
    total_mse = 0.0
    total_brightness = 0.0
    total_gradient = 0.0
    
    for step, frames in enumerate(train_dataloader, start=1):
        print(f"  Epoch {epoch}, Step {step}/{len(train_dataloader)}", end='\r')
        
        frames_device = frames.to(device)
        # Use actual batch size from current batch
        actual_batch_size = frames_device.shape[0]
        input_shape_batch = (actual_batch_size, n_channels, im_height, im_width)
        initial_states = model.get_initial_states(input_shape_batch)
        
        # Single forward pass returns (predictions, errors) tuple
        output_list = model(frames_device, initial_states)
        
        # Unpack predictions and errors from tuple
        predictions = [out[0] for out in output_list]  # List of predictions per timestep
        errors = [out[1] for out in output_list]  # List of errors per timestep
        
        # Compute standard PredNet error loss
        error = 0.0
        for t, error_output in enumerate(errors):
            weighted_layer_error = torch.matmul(error_output, layer_loss_weights)
            error += time_loss_weights[t] * torch.mean(weighted_layer_error)
        
        # Compute additional losses (MSE + brightness) on predictions
        additional_losses = compute_additional_losses(predictions, frames_device)
        
        # Combine all losses
        total_loss = error
        for loss_name, loss_value in additional_losses.items():
            total_loss = total_loss + loss_value
            if loss_name == 'mse':
                total_mse += loss_value.item()
            elif loss_name == 'brightness':
                total_brightness += loss_value.item()
            elif loss_name == 'gradient':
                total_gradient += loss_value.item()
        
        total_error += error.item()
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        del initial_states, output_list, predictions, errors, error, total_loss
        torch.cuda.empty_cache()
    
    print()  # New line after progress
    print(f"    Error: {total_error/len(train_dataloader):.6f}", end='')
    if use_mse_loss:
        print(f" | MSE: {total_mse/len(train_dataloader):.6f}", end='')
    if use_brightness_penalty:
        print(f" | Brightness: {total_brightness/len(train_dataloader):.6f}", end='')
    if use_gradient_loss:
        print(f" | Gradient: {total_gradient/len(train_dataloader):.6f}", end='')
    print()
    
    return total_error


def val_one_epoch(val_dataloader, model, input_shape, epoch):
    model.eval()
    total_error = 0.0
    
    with torch.no_grad(): 
        for step, frames in enumerate(val_dataloader, start=1):
            frames_device = frames.to(device)
            # Use actual batch size
            actual_batch_size = frames_device.shape[0]
            input_shape_batch = (actual_batch_size, n_channels, im_height, im_width)
            initial_states = model.get_initial_states(input_shape_batch)
            
            output_list = model(frames_device, initial_states)
            
            # Unpack tuple (predictions, errors) - we only need errors for validation
            errors = [out[1] for out in output_list]
            
            error = 0.0
            for t, error_output in enumerate(errors):
                weighted_layer_error = torch.matmul(error_output, layer_loss_weights)
                error += time_loss_weights[t] * torch.mean(weighted_layer_error)
            
            total_error += error
            del initial_states, output_list, errors, error
            
    return total_error


def extract_layer_outputs(model, frames):
    """
    Extract R, E, A, and Ahat outputs from all layers during forward pass.
    Returns dict with layer outputs for each timestep.
    """
    model.eval()
    
    # Prepare input
    frames_device = frames.to(device)
    frames_transposed = frames_device.transpose(0, 1)  # (nt, batch, c, h, w)
    
    batch_size = frames.shape[0]
    input_shape_local = (batch_size, n_channels, im_height, im_width)
    initial_states = model.get_initial_states(input_shape_local)
    
    timestep_outputs = []
    
    with torch.no_grad():
        hidden_states = initial_states
        
        for t in range(nt):
            A = frames_transposed[t, ...]
            
            # We need to manually run through the step to capture A and Ahat
            R_current = hidden_states['R']
            E_current = hidden_states['E']
            c_current = hidden_states['c']
            
            R_list, E_list, c_list = [], [], []
            A_list, Ahat_list = [], []
            R_upper = None
            
            # Top down pass (compute R)
            for l in reversed(range(model.nb_layers)):
                inputs = [R_current[l], E_current[l]]
                if model.is_not_top_layer(l):
                    inputs.append(R_upper)
                    
                inputs_torch = torch.cat(inputs, dim=-3)
                
                in_gate = model.hard_sigmoid(model.conv_layers['i'][l](inputs_torch))
                forget_gate = model.hard_sigmoid(model.conv_layers['f'][l](inputs_torch))
                out_gate = model.hard_sigmoid(model.conv_layers['o'][l](inputs_torch))
                cell_state = model.tanh(model.conv_layers['c'][l](inputs_torch))
                
                c_next = forget_gate * c_current[l] + in_gate * cell_state
                lstm_act = model.get_activation(model.lstm_activation)
                R_next = out_gate * lstm_act(c_next)
                
                c_list.insert(0, c_next)
                R_list.insert(0, R_next)
                
                if l > 0:
                    R_upper = model.upsample(R_next)
            
            # Bottom up pass (compute Ahat, E, A)
            A_current = A
            for l in range(model.nb_layers):
                # Compute Ahat from R
                Ahat = model.conv_layers['Ahat'][2*l](R_list[l])
                Ahat = model.conv_layers['Ahat'][2*l+1](Ahat)
                
                if l == 0:
                    Ahat = torch.clamp(Ahat, max=model.pixel_max)
                    frame_prediction = Ahat
                
                Ahat_list.append(Ahat)
                A_list.append(A_current)
                
                # Compute error
                E_pos = model.relu(Ahat - A_current)
                E_neg = model.relu(A_current - Ahat)
                E = torch.cat([E_neg, E_pos], dim=-3)
                E_list.append(E)
                
                # Compute next A
                if model.is_not_top_layer(l):
                    A_current = model.conv_layers['A'][2*l](E_list[l])
                    A_current = model.conv_layers['A'][2*l+1](A_current)
                    A_current = model.pool(A_current)
            
            # Store outputs for this timestep
            timestep_outputs.append({
                'R': [r.cpu().numpy() for r in R_list],
                'E': [e.cpu().numpy() for e in E_list],
                'A': [a.cpu().numpy() for a in A_list],
                'Ahat': [ahat.cpu().numpy() for ahat in Ahat_list],
                'prediction': frame_prediction.cpu().numpy()
            })
            
            # Update hidden states
            hidden_states = {
                'R': R_list,
                'c': c_list,
                'E': E_list,
            }
    
    return timestep_outputs


def plot_layer_outputs(original_frames, timestep_outputs, sample_idx, data_split):
    """
    Plot R, E, A, and Ahat for all layers and timesteps.
    Creates separate plots for each output type showing all layers.
    """
    os.makedirs(debug_images_dir, exist_ok=True)
    
    nb_layers = len(A_stack_sizes)
    
    # Plot each output type (R, E, A, Ahat) across all layers
    output_types = ['R', 'E', 'A', 'Ahat']
    colormaps = {'R': 'viridis', 'E': 'hot', 'A': 'gray', 'Ahat': 'gray'}
    
    for output_type in output_types:
        # Create figure with rows = layers, cols = timesteps
        fig = plt.figure(figsize=(nt * 3, nb_layers * 2.5))
        gs = GridSpec(nb_layers, nt, figure=fig, hspace=0.4, wspace=0.3)
        
        for layer_idx in range(nb_layers):
            for t in range(nt):
                outputs = timestep_outputs[t]
                
                ax = fig.add_subplot(gs[layer_idx, t])
                
                # Get the output for this layer and timestep
                output_img = outputs[output_type][layer_idx][0]  # First batch item
                
                # For visualization, take mean across channels
                if output_img.ndim == 3:
                    output_img = np.mean(output_img, axis=0)
                
                # Get min/max for normalization
                img_min, img_max = output_img.min(), output_img.max()
                
                # Normalize for visualization
                if img_max > img_min:
                    output_img_norm = (output_img - img_min) / (img_max - img_min)
                else:
                    output_img_norm = output_img
                
                ax.imshow(output_img_norm, cmap=colormaps[output_type])
                ax.set_title(f'Layer {layer_idx}, t={t}\nmin={img_min:.3f}, max={img_max:.3f}', fontsize=9)
                ax.axis('off')
        
        fig.suptitle(f'{output_type} across all layers - {data_split} sample {sample_idx}', fontsize=14, y=0.99)
        
        filename = f'{data_split}_sample{sample_idx}_{output_type}.png'
        filepath = os.path.join(debug_images_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        print(f'  Saved: {filename}')


def save_prediction_plot(original, predictions, sample_idx, data_split):
    """Plot original vs predicted frames (similar to eval.py)"""
    fig, axes = plt.subplots(2, nt, figsize=(nt * 2, 4))
    
    for t in range(nt):
        # Original
        orig_img = original[t, 0, :, :] if original.shape[1] == 1 else original[t].transpose(1, 2, 0)
        orig_img = np.clip(orig_img, 0, 1)
        orig_mean = orig_img.mean()
        axes[0, t].imshow(orig_img, cmap='gray', vmin=0, vmax=1)
        axes[0, t].set_title(f'Original t={t}\nmean={orig_mean:.3f}')
        axes[0, t].axis('off')
        
        # Predicted
        pred_img = predictions[t]['prediction'][0]  # First batch item
        pred_img = pred_img[0, :, :] if pred_img.shape[0] == 1 else pred_img.transpose(1, 2, 0)
        pred_img = np.clip(pred_img, 0, 1)
        pred_mean = pred_img.mean()
        pred_min = pred_img.min()
        pred_max = pred_img.max()
        axes[1, t].imshow(pred_img, cmap='gray', vmin=0, vmax=1)
        axes[1, t].set_title(f'Predicted t={t}\nmean={pred_mean:.3f}\nmin={pred_min:.3f} max={pred_max:.3f}', fontsize=8)
        axes[1, t].axis('off')
    
    plt.tight_layout()
    filename = f'{data_split}_predictions_sample{sample_idx}.png'
    filepath = os.path.join(debug_images_dir, filename)
    plt.savefig(filepath, dpi=150, bbox_inches='tight')
    plt.close()
    print(f'  Saved: {filename}')
    print(f'    Original mean: {original.mean():.4f}, Prediction mean: {pred_img.mean():.4f}')


def evaluate_and_visualize(model_path, n_samples=1):
    """Evaluate model and create visualizations"""
    print(f"\n{'='*60}")
    print("GENERATING DEBUG VISUALIZATIONS")
    print(f"{'='*60}\n")
    
    os.makedirs(debug_images_dir, exist_ok=True)
    
    # Load model
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
    
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    # Only evaluate on train split (since we're overfitting one video)
    data_split = 'train'
    print(f"\nProcessing {data_split} split...")
    
    data_path = f'./mnist_data/mnist_{data_split}.npy'
    dataloader = MNISTDataloader(
        data_path=data_path,
        batch_size=1,  # Process one at a time for visualization
        num_workers=2,
        background_level=background_level
    ).dataloader()
    
    samples_collected = 0
    
    for batch_idx, frames in enumerate(dataloader):
        if samples_collected >= n_samples:
            break
        
        print(f"  Visualizing sample {samples_collected + 1}/{n_samples}")
        
        # DIAGNOSTIC: Test direct model prediction
        with torch.no_grad():
            frames_device = frames.to(device)
            batch_size_vis = frames.shape[0]
            input_shape_vis = (batch_size_vis, n_channels, im_height, im_width)
            init_states = model.get_initial_states(input_shape_vis)
            direct_preds = model(frames_device, init_states)
            direct_pred_mean = torch.stack(direct_preds).mean().item()
            direct_pred_max = torch.stack(direct_preds).max().item()
            print(f"    Direct model prediction: mean={direct_pred_mean:.4f}, max={direct_pred_max:.4f}")
        
        # Extract layer outputs
        timestep_outputs = extract_layer_outputs(model, frames)
        
        # Get original frames
        original = frames.squeeze(0).cpu().numpy()  # (nt, c, h, w)
        print(f"    Original frames: mean={original.mean():.4f}, max={original.max():.4f}")
        
        # Save prediction comparison
        save_prediction_plot(original, timestep_outputs, samples_collected + 1, data_split)
        
        # Save layer outputs
        plot_layer_outputs(original, timestep_outputs, samples_collected + 1, data_split)
        
        samples_collected += 1
    
    print(f"\nCompleted: {samples_collected} sample visualized")
    print(f"\n{'='*60}")
    print(f"All debug images saved to: {debug_images_dir}")
    print(f"{'='*60}\n")


if __name__ == '__main__':
    # Step 1: Train for 2 epochs
    model_path = train_debug()
    
    # Step 2: Evaluate and create visualizations (only 1 train sample since overfitting)
    evaluate_and_visualize(model_path, n_samples=1)
    
    print("\n✓ Debug script complete!")
    print(f"  - Model checkpoints: {checkpoint_dir}")
    print(f"  - Debug visualizations: {debug_images_dir}")

