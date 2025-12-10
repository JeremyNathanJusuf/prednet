import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.mnist import MNISTDataloader
from prednet import Prednet
import config

# Import parameters from config
device = config.device
batch_size = config.batch_size
nt = config.nt

n_channels = config.n_channels
im_height = config.im_height
im_width = config.im_width
input_shape = config.input_shape
A_stack_sizes = config.A_stack_sizes
R_stack_sizes = config.R_stack_sizes
A_filter_sizes = config.A_filter_sizes
Ahat_filter_sizes = config.Ahat_filter_sizes
R_filter_sizes = config.R_filter_sizes
pixel_max = config.pixel_max
lstm_activation = config.lstm_activation
A_activation = config.A_activation

model_path = config.model_path

def load_model(model, model_path):
    """Load model weights from checkpoint."""
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_model_and_dataloader(data_path, extrap_time=None):
    """
    Initialize model and dataloader for MNIST dataset.
    
    Args:
        data_path: Path to MNIST .npy file
        extrap_time: Number of extrapolation steps
    
    Returns:
        model, dataloader
    """
    # Use "custom_mnist" dataset type for the custom MNIST format
    dataloader = MNISTDataloader(
        data_path=data_path,
        batch_size=batch_size, 
        num_workers=2
    ).dataloader(mnist_dataset_type="custom_mnist")
    
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
        output_type='prediction',
        device=device
    )
    model.to(device=device)
    model = load_model(model, model_path)
    model.eval()
    
    return model, dataloader

def plot_predictions(gt_frames, pred_frames, batch_idx, step, save_dir='./eval_plots_mnist'):
    """
    Plot ground truth vs model predictions.
    
    Args:
        gt_frames: Ground truth frames (batch, nt, C, H, W)
        pred_frames: Model predictions (batch, nt, C, H, W)
        batch_idx: Which batch item to visualize
        step: Step number for filename
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy if needed
    if isinstance(gt_frames, torch.Tensor):
        gt_frames = gt_frames.cpu().numpy()
    if isinstance(pred_frames, torch.Tensor):
        pred_frames = pred_frames.cpu().numpy()
    
    # Select single batch item
    gt = gt_frames[batch_idx]  # (nt, C, H, W)
    pred = pred_frames[batch_idx]  # (nt, C, H, W)
    
    nt = gt.shape[0]
    
    # Create figure with 2 rows: ground truth and predictions
    fig, axes = plt.subplots(2, nt, figsize=(2*nt, 4))
    
    for t in range(nt):
        # Ground truth
        img_gt = np.transpose(gt[t], (1, 2, 0))  # (H, W, C)
        if img_gt.shape[2] == 1:  # Grayscale
            img_gt = img_gt.squeeze(-1)
            axes[0, t].imshow(img_gt, cmap='gray')
        else:  # RGB
            axes[0, t].imshow(img_gt)
        axes[0, t].axis('off')
        axes[0, t].set_title(f'GT t={t}', fontsize=10)
        
        # Model prediction
        img_pred = np.transpose(pred[t], (1, 2, 0))  # (H, W, C)
        if img_pred.shape[2] == 1:  # Grayscale
            img_pred = img_pred.squeeze(-1)
            axes[1, t].imshow(img_pred, cmap='gray')
        else:  # RGB
            axes[1, t].imshow(img_pred)
        axes[1, t].axis('off')
        axes[1, t].set_title(f'Pred t={t}', fontsize=10)
    
    # Row labels
    axes[0, 0].set_ylabel('Ground Truth', fontsize=12, rotation=0, ha='right', va='center')
    axes[1, 0].set_ylabel('Model Pred', fontsize=12, rotation=0, ha='right', va='center')
    
    plt.suptitle(f'MNIST Evaluation - Batch {batch_idx}, Step {step}', fontsize=14)
    plt.tight_layout()
    
    save_path = os.path.join(save_dir, f'mnist_step{step}_batch{batch_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved plot to {save_path}")

def evaluate_and_plot(data_path, extrap_time=4, num_samples=5):
    """
    Evaluate model and create visualization plots.
    
    Args:
        data_path: Path to MNIST .npy file
        extrap_time: Number of extrapolation steps
        num_samples: Number of samples to plot
    """
    print(f"\nEvaluating MNIST dataset:")
    print(f"  Data path: {data_path}")
    print(f"  Model extrapolation time: {extrap_time}")
    print(f"  Number of samples to plot: {num_samples}")
    print("="*60)
    
    model, dataloader = get_model_and_dataloader(data_path, extrap_time=extrap_time)
    
    plot_count = 0
    
    with torch.no_grad():
        for step, frames in enumerate(dataloader, start=1):
            batch_size_actual = frames.shape[0]
            
            # Get model predictions
            initial_states = model.get_initial_states(input_shape)
            output_list, _ = model(frames.to(device), initial_states)
            pred_frames = torch.stack(output_list, dim=1)  # (batch, nt, C, H, W)
            pred_frames = pred_frames.detach().cpu()
            
            # Generate plots for first few samples
            if plot_count < num_samples:
                for batch_idx in range(min(2, batch_size_actual)):
                    if plot_count >= num_samples:
                        break
                    plot_predictions(
                        frames, 
                        pred_frames,
                        batch_idx,
                        step,
                        save_dir='./eval_plots_mnist'
                    )
                    plot_count += 1
            
            # Print progress
            if step % 5 == 0:
                print(f"Processed {step} batches...")
            
            if plot_count >= num_samples:
                break
            
            del initial_states, output_list
    
    print("\n" + "="*60)
    print(f"Evaluation complete! Generated {plot_count} plots.")
    print(f"Plots saved to: ./eval_plots_mnist/")
    print("="*60)

if __name__ == '__main__':
    # You can change this to your MNIST data path
    # For now, using the val data path from config
    data_path = config.val_path
    
    print(f"Using data from: {data_path}")
    print("Note: This script uses 'custom_mnist' dataset type")
    print("If you need to use a different data file, update the data_path variable.\n")
    
    # Evaluate and plot
    evaluate_and_plot(
        data_path=data_path,
        extrap_time=4,
        num_samples=5
    )

