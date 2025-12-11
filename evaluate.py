import os
import numpy as np
import torch

from utils.mnist import MNISTDataloader
from utils.model import load_model
from utils.plot import plot_predictions, plot_comparison
from prednet import Prednet
import config
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

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


def get_model_and_dataloader(data_path, extrap_time=None):
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
    model, _, _ = load_model(model, model_path, device)
    model.eval()
    
    return model, dataloader


def evaluate_and_plot(data_path, extrap_time=4, num_samples=5):
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
            
            if plot_count >= num_samples:
                break
            
            del initial_states, output_list
    
    print("\n" + "="*60)
    print(f"Evaluation complete! Generated {plot_count} plots.")
    print(f"Plots saved to: ./eval_plots_mnist/")
    print("="*60)
    
def evaluate_and_compare_to_baseline(data_path, extrap_time=4, num_samples=5):
    model, dataloader = get_model_and_dataloader(data_path, extrap_time=extrap_time)
    
    plot_count = 0
    total_naive_l1_error = 0
    total_naive_l2_error = 0
    total_naive_psnr = 0
    total_naive_ssim = 0
    total_model_l1_error = 0
    total_model_l2_error = 0
    total_model_psnr = 0
    total_model_ssim = 0
    
    total_steps = len(dataloader)
    
    with torch.no_grad():
        for step, frames in tqdm(enumerate(dataloader, start=1), total=total_steps):
            batch_size_actual = frames.shape[0]
            nt_actual = frames.shape[1]
            
            # Get model predictions
            initial_states = model.get_initial_states(input_shape)
            output_list, _ = model(frames.to(device), initial_states)
            model_pred_frames = torch.stack(output_list, dim=1)  # (batch, nt, C, H, W)
            model_pred_frames = model_pred_frames.detach().cpu()
            
            # Create naive predictions: repeat the last frame before extrapolation for all timesteps
            # For t < extrap_time: use actual frames, for t >= extrap_time: repeat frame at extrap_time-1
            naive_pred_frames = frames.clone()
            last_frame = frames[:, extrap_time-1:extrap_time, ...]  # (batch, 1, C, H, W)
            naive_pred_frames[:, extrap_time:, ...] = last_frame.expand(-1, nt_actual - extrap_time, -1, -1, -1)
            
            # For metrics, compare extrapolated frames only
            gt_extrap = frames[:, extrap_time:, ...]
            model_extrap = model_pred_frames[:, extrap_time:, ...]
            naive_extrap = naive_pred_frames[:, extrap_time:, ...]
            
            # Reshape from (B, T, C, H, W) to (B*T, C, H, W) for PSNR/SSIM
            b, t, c, h, w = gt_extrap.shape
            gt_extrap_flat = gt_extrap.reshape(b * t, c, h, w)
            model_extrap_flat = model_extrap.reshape(b * t, c, h, w)
            naive_extrap_flat = naive_extrap.reshape(b * t, c, h, w)
            
            # Model metrics
            model_l1 = torch.mean(torch.abs(model_extrap - gt_extrap))
            model_l2 = torch.mean(torch.pow(model_extrap - gt_extrap, 2))
            model_psnr = peak_signal_noise_ratio(model_extrap_flat, gt_extrap_flat, data_range=1.0)
            model_ssim = structural_similarity_index_measure(model_extrap_flat, gt_extrap_flat, data_range=1.0)
            
            # Naive baseline metrics
            naive_l1 = torch.mean(torch.abs(naive_extrap - gt_extrap))
            naive_l2 = torch.mean(torch.pow(naive_extrap - gt_extrap, 2))
            naive_psnr = peak_signal_noise_ratio(naive_extrap_flat, gt_extrap_flat, data_range=1.0)
            naive_ssim = structural_similarity_index_measure(naive_extrap_flat, gt_extrap_flat, data_range=1.0)
            
            total_model_l1_error += model_l1.item()
            total_model_l2_error += model_l2.item()
            total_model_psnr += model_psnr.item()
            total_model_ssim += model_ssim.item()
            total_naive_l1_error += naive_l1.item()
            total_naive_l2_error += naive_l2.item()
            total_naive_psnr += naive_psnr.item()
            total_naive_ssim += naive_ssim.item()
            
            # Generate comparison plots for first few samples
            if plot_count < num_samples:
                for batch_idx in range(min(2, batch_size_actual)):
                    if plot_count >= num_samples:
                        break
                    plot_comparison(
                        frames,
                        model_pred_frames,
                        naive_pred_frames,
                        batch_idx,
                        step,
                        extrap_time=extrap_time,
                        save_dir='./eval_plots_mnist'
                    )
                    plot_count += 1
            
            del initial_states, output_list

    print(f"Total model L1 error: {total_model_l1_error / total_steps:.4f}")
    print(f"Total model L2 error: {total_model_l2_error / total_steps:.4f}")
    print(f"Total model PSNR: {total_model_psnr / total_steps:.4f}")
    print(f"Total model SSIM: {total_model_ssim / total_steps:.4f}")
    print(f"Total naive L1 error: {total_naive_l1_error / total_steps:.4f}")
    print(f"Total naive L2 error: {total_naive_l2_error / total_steps:.4f}")
    print(f"Total naive PSNR: {total_naive_psnr / total_steps:.4f}")
    print(f"Total naive SSIM: {total_naive_ssim / total_steps:.4f}")
    
if __name__ == '__main__':
    data_path = config.val_path

    evaluate_and_compare_to_baseline(
        data_path=data_path,
        extrap_time=4,
        num_samples=5
    )
