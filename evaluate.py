import os
import torch

from utils.mnist import MNISTDataloader, split_mnist_data
from utils.model import load_model
from utils.plot import plot_predictions, plot_comparison, plot_disruption_comparison
from prednet import Prednet
import config
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
from tqdm import tqdm

from utils.dataset_generator import DisruptDatasetGenerator

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


def get_model_and_dataloader(data_path, model_path, extrap_time=None):
    dataloader = MNISTDataloader(
        data_path=data_path,
        batch_size=batch_size, 
        num_workers=2,
        target_h=im_height,
        target_w=im_width,
        target_channels=n_channels
    ).dataloader(shuffle=False)
    
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


def evaluate_and_plot(data_path, model_path, extrap_time=4, num_samples=5):
    print(f"\nEvaluating MNIST dataset:")
    print(f"  Data path: {data_path}")
    print(f"  Model path: {model_path}")
    print(f"  Model extrapolation time: {extrap_time}")
    print(f"  Number of samples to plot: {num_samples}")
    print("="*60)
    
    model, dataloader = get_model_and_dataloader(data_path, model_path, extrap_time=extrap_time)
    
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


def evaluate_and_compare_to_baseline(data_path, model_path, extrap_time=4, num_samples=5):
    """
    Evaluate model against naive baseline (repeating last frame before extrapolation).
    
    Args:
        data_path: Path to evaluation data
        model_path: Path to model checkpoint
        extrap_time: Timestep where extrapolation begins
        num_samples: Number of samples to plot
    """
    model, dataloader = get_model_and_dataloader(data_path, model_path, extrap_time=extrap_time)
    
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
            # IMPORTANT: frames is the REAL ground truth from dataloader (not modified)
            gt_extrap = frames[:, extrap_time:, ...].clone()  # Real GT for extrapolation period
            model_extrap = model_pred_frames[:, extrap_time:, ...]  # Model predictions for extrapolation
            naive_extrap = naive_pred_frames[:, extrap_time:, ...]  # Naive predictions for extrapolation
            
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
                        save_dir='./eval_plots_model_vs_baseline'
                    )
                    plot_count += 1
            
            del initial_states, output_list

    print("\n" + "="*60)
    print("Naive Baseline Method")
    print(f"  L1 error: {total_naive_l1_error / total_steps:.4f}")
    print(f"  L2 error: {total_naive_l2_error / total_steps:.4f}")
    print(f"  PSNR: {total_naive_psnr / total_steps:.4f}")
    print(f"  SSIM: {total_naive_ssim / total_steps:.4f}")
    
    print("\nModel")
    print(f"  L1 error: {total_model_l1_error / total_steps:.4f}")
    print(f"  L2 error: {total_model_l2_error / total_steps:.4f}")
    print(f"  PSNR: {total_model_psnr / total_steps:.4f}")
    print(f"  SSIM: {total_model_ssim / total_steps:.4f}")
    print("="*60)
    

def evaluate_disruption(disruption_path, original_path, disruption_time, model_path, model_path_finetuned_on_disrupt=None, num_samples=5, save_dir='./eval_plots_disruption'):
    model, disruption_dataloader = get_model_and_dataloader(disruption_path, model_path, extrap_time=None)
    _, original_dataloader = get_model_and_dataloader(original_path, model_path, extrap_time=None)
    
    # Load finetuned model if provided
    finetuned_model = None
    if model_path_finetuned_on_disrupt is not None:
        finetuned_model, _ = get_model_and_dataloader(disruption_path, model_path_finetuned_on_disrupt, extrap_time=None)
        finetuned_model = finetuned_model  # The first element is the model
    
    plot_count = 0
    total_disrupt_pred_l1 = 0
    total_disrupt_pred_l2 = 0
    total_disrupt_pred_psnr = 0
    total_disrupt_pred_ssim = 0
    total_original_pred_l1 = 0
    total_original_pred_l2 = 0
    total_original_pred_psnr = 0
    total_original_pred_ssim = 0
    total_naive_l1 = 0
    total_naive_l2 = 0
    total_naive_psnr = 0
    total_naive_ssim = 0
    total_finetuned_pred_l1 = 0
    total_finetuned_pred_l2 = 0
    total_finetuned_pred_psnr = 0
    total_finetuned_pred_ssim = 0
    
    total_steps = len(disruption_dataloader)
    
    os.makedirs(save_dir, exist_ok=True)
    
    # evaluate from the disruption time + 1
    eval_start = disruption_time + 1
    
    with torch.no_grad():
        for step, (disrupt_frames, original_frames) in enumerate(zip(disruption_dataloader, original_dataloader), start=1):
            batch_size_actual = disrupt_frames.shape[0]
            nt_actual = disrupt_frames.shape[1]
            
            initial_states = model.get_initial_states(input_shape)
            
            output_list, _ = model(disrupt_frames.to(device), initial_states)
            disrupt_pred = torch.stack(output_list, dim=1).detach().cpu()
            
            initial_states = model.get_initial_states(input_shape)
            output_list, _ = model(original_frames.to(device), initial_states)
            original_pred = torch.stack(output_list, dim=1).detach().cpu()
            
            # Finetuned model predictions (if available)
            finetuned_pred = None
            if finetuned_model is not None:
                initial_states = finetuned_model.get_initial_states(input_shape)
                output_list, _ = finetuned_model(disrupt_frames.to(device), initial_states)
                finetuned_pred = torch.stack(output_list, dim=1).detach().cpu()
            
            naive_pred = original_frames.clone()
            disrupt_frame_at_t = disrupt_frames[:, disruption_time:disruption_time+1, ...]
            naive_pred[:, disruption_time:, ...] = disrupt_frame_at_t.expand(-1, nt_actual - disruption_time, -1, -1, -1)
            
            gt_eval = disrupt_frames[:, eval_start:, ...]
            disrupt_pred_eval = disrupt_pred[:, eval_start:, ...]
            original_pred_eval = original_pred[:, eval_start:, ...]
            naive_pred_eval = naive_pred[:, eval_start:, ...]
            
            b, t, c, h, w = gt_eval.shape
            gt_flat = gt_eval.reshape(b * t, c, h, w)
            disrupt_pred_flat = disrupt_pred_eval.reshape(b * t, c, h, w)
            original_pred_flat = original_pred_eval.reshape(b * t, c, h, w)
            naive_pred_flat = naive_pred_eval.reshape(b * t, c, h, w)
            
            disrupt_l1 = torch.mean(torch.abs(disrupt_pred_eval - gt_eval))
            disrupt_l2 = torch.mean(torch.pow(disrupt_pred_eval - gt_eval, 2))
            disrupt_psnr = peak_signal_noise_ratio(disrupt_pred_flat, gt_flat, data_range=1.0)
            disrupt_ssim = structural_similarity_index_measure(disrupt_pred_flat, gt_flat, data_range=1.0)
            
            original_l1 = torch.mean(torch.abs(original_pred_eval - gt_eval))
            original_l2 = torch.mean(torch.pow(original_pred_eval - gt_eval, 2))
            original_psnr = peak_signal_noise_ratio(original_pred_flat, gt_flat, data_range=1.0)
            original_ssim = structural_similarity_index_measure(original_pred_flat, gt_flat, data_range=1.0)
            
            naive_l1 = torch.mean(torch.abs(naive_pred_eval - gt_eval))
            naive_l2 = torch.mean(torch.pow(naive_pred_eval - gt_eval, 2))
            naive_psnr = peak_signal_noise_ratio(naive_pred_flat, gt_flat, data_range=1.0)
            naive_ssim = structural_similarity_index_measure(naive_pred_flat, gt_flat, data_range=1.0)
            
            # Finetuned model metrics (if available)
            if finetuned_pred is not None:
                finetuned_pred_eval = finetuned_pred[:, eval_start:, ...]
                finetuned_pred_flat = finetuned_pred_eval.reshape(b * t, c, h, w)
                finetuned_l1 = torch.mean(torch.abs(finetuned_pred_eval - gt_eval))
                finetuned_l2 = torch.mean(torch.pow(finetuned_pred_eval - gt_eval, 2))
                finetuned_psnr = peak_signal_noise_ratio(finetuned_pred_flat, gt_flat, data_range=1.0)
                finetuned_ssim = structural_similarity_index_measure(finetuned_pred_flat, gt_flat, data_range=1.0)
                total_finetuned_pred_l1 += finetuned_l1.item()
                total_finetuned_pred_l2 += finetuned_l2.item()
                total_finetuned_pred_psnr += finetuned_psnr.item()
                total_finetuned_pred_ssim += finetuned_ssim.item()
            
            total_disrupt_pred_l1 += disrupt_l1.item()
            total_disrupt_pred_l2 += disrupt_l2.item()
            total_disrupt_pred_psnr += disrupt_psnr.item()
            total_disrupt_pred_ssim += disrupt_ssim.item()
            total_original_pred_l1 += original_l1.item()
            total_original_pred_l2 += original_l2.item()
            total_original_pred_psnr += original_psnr.item()
            total_original_pred_ssim += original_ssim.item()
            total_naive_l1 += naive_l1.item()
            total_naive_l2 += naive_l2.item()
            total_naive_psnr += naive_psnr.item()
            total_naive_ssim += naive_ssim.item()
            
            if plot_count < num_samples:
                for batch_idx in range(min(2, batch_size_actual)):
                    if plot_count >= num_samples:
                        break
                    
                    plot_disruption_comparison(
                        disrupt_frames, disrupt_pred, original_pred, naive_pred, finetuned_pred,
                        batch_idx, step, disruption_time, eval_start, save_dir
                    )
                    plot_count += 1
            
            del initial_states, output_list
    
    print(f"Disrupt Pred L1: {total_disrupt_pred_l1 / total_steps:.4f}")
    print(f"Disrupt Pred L2: {total_disrupt_pred_l2 / total_steps:.4f}")
    print(f"Disrupt Pred PSNR: {total_disrupt_pred_psnr / total_steps:.4f}")
    print(f"Disrupt Pred SSIM: {total_disrupt_pred_ssim / total_steps:.4f}")
    print(f"Original Pred L1: {total_original_pred_l1 / total_steps:.4f}")
    print(f"Original Pred L2: {total_original_pred_l2 / total_steps:.4f}")
    print(f"Original Pred PSNR: {total_original_pred_psnr / total_steps:.4f}")
    print(f"Original Pred SSIM: {total_original_pred_ssim / total_steps:.4f}")
    print(f"Naive L1: {total_naive_l1 / total_steps:.4f}")
    print(f"Naive L2: {total_naive_l2 / total_steps:.4f}")
    print(f"Naive PSNR: {total_naive_psnr / total_steps:.4f}")
    print(f"Naive SSIM: {total_naive_ssim / total_steps:.4f}")
    if finetuned_model is not None:
        print(f"Finetuned Pred L1: {total_finetuned_pred_l1 / total_steps:.4f}")
        print(f"Finetuned Pred L2: {total_finetuned_pred_l2 / total_steps:.4f}")
        print(f"Finetuned Pred PSNR: {total_finetuned_pred_psnr / total_steps:.4f}")
        print(f"Finetuned Pred SSIM: {total_finetuned_pred_ssim / total_steps:.4f}")
    
if __name__ == '__main__':
    disrupt_sudden_appear_path = config.disrupt_sudden_appear_path
    disrupt_sudden_transform_path = config.disrupt_sudden_transform_path
    disrupt_sudden_disappear_path = config.disrupt_sudden_disappear_path
    disrupt_base_path = config.disrupt_base_path
    
    # Compare model vs naive baseline
    # evaluate_and_compare_to_baseline(
    #     data_path='./data/mnist_val_multi.npy',
    #     model_path=model_path,
    #     extrap_time=8,
    #     num_samples=5
    # )
    
    # # Simple evaluation with plots
    # evaluate_and_plot(
    #     data_path=original_path,
    #     model_path=model_path,
    #     extrap_time=8,
    #     num_samples=5
    # )
    
    # generator = DisruptDatasetGenerator(
    #     base_data_path='./data/mnist_val.npy',
    #     disruption_time=8,
    #     max_iou=0.1,
    #     target_h=config.im_height,
    #     target_w=config.im_width
    # )
    # print(f"Loaded {generator.num_base_videos} base videos")
    # print(f"Disruption occurs at timestep {generator.disruption_time}")
    # print(f"Max IOU threshold: {generator.max_iou}")
    # print(f"Target dimensions: {config.im_height}x{config.im_width}")

    # datasets = generator.generate_dataset(
    #     num_samples=2000,
    #     min_scale=2.0,
    #     max_scale=2.5,
    #     nt=config.nt,
    #     min_digits=2,  # Base has 2 digits, total will be 2-5
    #     max_digits=3,  # Including disruption digit and additional digits
    #     h=config.im_height,
    #     w=config.im_width
    # )
    # print("Generated datasets:", list(datasets.keys()))
    # print("\nDisruption datasets:")
    # print(f"  - Sudden appear: {datasets['sudden_appear'][1]}")
    # print(f"  - Transform: {datasets['transform'][1]}")
    # print(f"  - Disappear: {datasets['disappear'][1]}")
    # print("\nBase validation dataset (shared by all disruption types):")
    # print(f"  - Base val: {datasets['base_val'][1]}")
    
    print(f"Evaluating sudden appear disruption...")
    evaluate_disruption(
        disruption_path=disrupt_sudden_appear_path,
        original_path=disrupt_base_path,
        disruption_time=8,
        model_path=model_path,
        num_samples=5,
        save_dir='./eval_plots_sudden_appear_'
    )
    
    print(f"Evaluating sudden transform disruption...")
    evaluate_disruption(
        disruption_path=disrupt_sudden_transform_path,
        original_path=disrupt_base_path,
        disruption_time=8,
        model_path=model_path,
        num_samples=5,
        save_dir='./eval_plots_sudden_transform_'
    )
    
    print(f"Evaluating sudden disappear disruption...")
    evaluate_disruption(
        disruption_path=disrupt_sudden_disappear_path,
        original_path=disrupt_base_path,
        disruption_time=8,
        model_path=model_path,
        num_samples=5,
        save_dir='./eval_plots_sudden_disappear_'
    )