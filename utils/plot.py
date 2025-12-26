import numpy as np
import matplotlib.pyplot as plt
import os
import torch


def plot_comparison(gt_frames, model_pred_frames, naive_pred_frames, batch_idx, step, extrap_time=None, save_dir='./eval_plots_mnist'):
    """
    Plot ground truth vs model predictions vs naive predictions.
    
    Args:
        gt_frames: Ground truth frames (batch, nt, C, H, W)
        model_pred_frames: Model predictions (batch, nt, C, H, W)
        naive_pred_frames: Naive predictions - last frame repeated (batch, nt, C, H, W)
        batch_idx: Which batch item to visualize
        step: Step number for filename
        extrap_time: Time step where extrapolation begins (optional, for visual marker)
        save_dir: Directory to save plots
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy if needed
    if isinstance(gt_frames, torch.Tensor):
        gt_frames = gt_frames.cpu().numpy()
    if isinstance(model_pred_frames, torch.Tensor):
        model_pred_frames = model_pred_frames.cpu().numpy()
    if isinstance(naive_pred_frames, torch.Tensor):
        naive_pred_frames = naive_pred_frames.cpu().numpy()
    
    # Select single batch item
    gt = gt_frames[batch_idx]  # (nt, C, H, W)
    model_pred = model_pred_frames[batch_idx]  # (nt, C, H, W)
    naive_pred = naive_pred_frames[batch_idx]  # (nt, C, H, W)
    
    nt = gt.shape[0]
    
    # Create figure with 3 rows: ground truth, model predictions, naive predictions
    fig, axes = plt.subplots(3, nt, figsize=(2.5*nt, 7.5))
    
    row_data = [
        (gt, 'Ground Truth', 'GT'),
        (model_pred, 'Model Pred', 'Model'),
        (naive_pred, 'Naive Pred', 'Naive'),
    ]
    
    for row_idx, (frames, row_label, title_prefix) in enumerate(row_data):
        for t in range(nt):
            img = np.transpose(frames[t], (1, 2, 0))  # (H, W, C)
            img = np.clip(img, 0, 1)
            
            # Handle grayscale images
            if img.shape[2] == 1:
                img = img.squeeze(-1)
                axes[row_idx, t].imshow(img, cmap='gray', vmin=0, vmax=1)
            elif img.shape[2] == 3:
                if np.allclose(img[:,:,0], img[:,:,1], atol=0.1) and np.allclose(img[:,:,1], img[:,:,2], atol=0.1):
                    axes[row_idx, t].imshow(img[:,:,0], cmap='gray', vmin=0, vmax=1)
                else:
                    axes[row_idx, t].imshow(img)
            else:
                axes[row_idx, t].imshow(img)
            
            axes[row_idx, t].axis('off')
            
            # Add title with extrapolation marker
            title = f'{title_prefix} t={t}'
            if extrap_time is not None and t >= extrap_time:
                title = f'{title_prefix} t={t}*'
                # Add red border for extrapolation frames
                for spine in axes[row_idx, t].spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(2)
                axes[row_idx, t].spines['top'].set_visible(True)
                axes[row_idx, t].spines['bottom'].set_visible(True)
                axes[row_idx, t].spines['left'].set_visible(True)
                axes[row_idx, t].spines['right'].set_visible(True)
            
            axes[row_idx, t].set_title(title, fontsize=9)
        
        # Row labels
        axes[row_idx, 0].set_ylabel(row_label, fontsize=11, rotation=0, ha='right', va='center')
    
    # Add legend for extrapolation marker
    if extrap_time is not None:
        fig.text(0.5, 0.02, f'* Extrapolation frames (t >= {extrap_time})', 
                 ha='center', fontsize=10, color='red')
    
    plt.suptitle(f'Comparison - Batch {batch_idx}, Step {step}', fontsize=14)
    plt.tight_layout(rect=[0, 0.03, 1, 0.97])
    
    save_path = os.path.join(save_dir, f'comparison_step{step}_batch{batch_idx}.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


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
        img_gt = np.clip(img_gt, 0, 1)  # Clip to valid range
        if img_gt.shape[2] == 1:  # Grayscale
            img_gt = img_gt.squeeze(-1)
            axes[0, t].imshow(img_gt, cmap='gray', vmin=0, vmax=1)
        elif img_gt.shape[2] == 3:
            # Check if it's actually grayscale (all channels same) - display as grayscale
            if np.allclose(img_gt[:,:,0], img_gt[:,:,1]) and np.allclose(img_gt[:,:,1], img_gt[:,:,2]):
                axes[0, t].imshow(img_gt[:,:,0], cmap='gray', vmin=0, vmax=1)
            else:
                axes[0, t].imshow(img_gt)
        else:
            axes[0, t].imshow(img_gt)
        axes[0, t].axis('off')
        axes[0, t].set_title(f'GT t={t}', fontsize=10)
        
        # Model prediction
        img_pred = np.transpose(pred[t], (1, 2, 0))  # (H, W, C)
        img_pred = np.clip(img_pred, 0, 1)  # Clip to valid range
        if img_pred.shape[2] == 1:  # Grayscale
            img_pred = img_pred.squeeze(-1)
            axes[1, t].imshow(img_pred, cmap='gray', vmin=0, vmax=1)
        elif img_pred.shape[2] == 3:
            # Check if it's actually grayscale (all channels same) - display as grayscale
            if np.allclose(img_pred[:,:,0], img_pred[:,:,1], atol=0.1) and np.allclose(img_pred[:,:,1], img_pred[:,:,2], atol=0.1):
                axes[1, t].imshow(img_pred[:,:,0], cmap='gray', vmin=0, vmax=1)
            else:
                # Show as RGB but also print channel stats for debugging
                axes[1, t].imshow(img_pred)
        else:
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


def plot_layer(hidden_states_list, frames, layer_name, output_dir, batch_idx):
    num_timesteps = len(hidden_states_list)
    num_layers = len(hidden_states_list[0][layer_name])
    
    fig, axes = plt.subplots(num_layers, num_timesteps, figsize=(num_timesteps * 3, num_layers * 3))
    if num_layers == 1:
        axes = axes.reshape(1, -1)
    for layer in range(num_layers):
        for t in range(num_timesteps):
            layer_data = hidden_states_list[t][layer_name][layer][batch_idx].detach().cpu().numpy()

            layer_mean = np.mean(layer_data, axis=0)
            axes[layer, t].imshow(layer_mean, cmap='gray')
            
            min_val, max_val, mean_val = layer_data.min(), layer_data.max(), layer_data.mean()
            axes[layer, t].set_title(f'L{layer} t={t}\nmin:{min_val:.3f} max:{max_val:.3f}\nmean:{mean_val:.3f}', fontsize=8)
            axes[layer, t].axis('off')
            
    plt.suptitle(f'{layer_name}', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, str(batch_idx) ,f'{layer_name}.png'), dpi=150, bbox_inches='tight')
    plt.close()
    
def plot_input_vs_prediction(hidden_states_list, frames, output_dir, batch_idx):
    num_timesteps = len(hidden_states_list)
    fig, axes = plt.subplots(2, num_timesteps, figsize=(num_timesteps * 3, 6))
    
    for t in range(num_timesteps):
        input_img = frames[batch_idx, t].detach().cpu().numpy() 
        input_img = np.transpose(input_img, (1, 2, 0))  # Convert to [H, W, C]
        axes[0, t].imshow(input_img, vmin=0, vmax=1)
        
        min_val, max_val, mean_val = input_img.min(), input_img.max(), input_img.mean()
        axes[0, t].set_title(f'Input t={t}\nmin:{min_val:.3f} max:{max_val:.3f}\nmean:{mean_val:.3f}', fontsize=8)
        axes[0, t].axis('off')
        
        pred_img = hidden_states_list[t]['Ahat'][0][batch_idx].detach().cpu().numpy()
        pred_img = np.transpose(pred_img, (1, 2, 0))  # Convert to [H, W, C]
        
        axes[1, t].imshow(pred_img, vmin=0, vmax=1)
        
        min_val, max_val, mean_val = pred_img.min(), pred_img.max(), pred_img.mean()
        axes[1, t].set_title(f'Prediction t={t}\nmin:{min_val:.3f} max:{max_val:.3f}\nmean:{mean_val:.3f}', fontsize=8)
        axes[1, t].axis('off')
        
    plt.suptitle('Input vs Prediction (Layer 0)', fontsize=16)
    plt.savefig(os.path.join(output_dir, str(batch_idx), 'input_vs_prediction.png'), dpi=150, bbox_inches='tight')
    plt.close()

def plot_hidden_states_list(hidden_states_list, frames, epoch, data_split='train', debug_images_dir='./debug_layer_images'):
    
    output_dir = f'{debug_images_dir}/{data_split}/{epoch}'
    os.makedirs(output_dir, exist_ok=True)
    
    num_timesteps = len(hidden_states_list)
    num_layers = len(hidden_states_list[0]['R'])

    for batch_idx in range(min(frames.shape[0], 8)):
        os.makedirs(os.path.join(output_dir, str(batch_idx)), exist_ok=True)
        plot_layer(hidden_states_list, frames, 'R', output_dir, batch_idx)
        plot_layer(hidden_states_list, frames, 'A', output_dir, batch_idx)
        plot_layer(hidden_states_list, frames, 'Ahat', output_dir, batch_idx)
        plot_layer(hidden_states_list, frames, 'E', output_dir, batch_idx)
        
        plot_input_vs_prediction(hidden_states_list, frames, output_dir, batch_idx)