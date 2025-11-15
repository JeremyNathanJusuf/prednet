import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.mnist import MNISTDataloader
from utils.plot import plot_hidden_states_list
from prednet import Prednet
import cv2
import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure
import pandas as pd
import config

# Import parameters from config
device = config.device
n_samples = config.n_samples_eval
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
debug_images_dir = config.debug_images_dir

def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_model_and_dataloader(data_split, output_type='error', extrap_time=None):
    # Choose the appropriate data file
    if data_split == 'train':
        data_file = config.train_path
    else:
        data_file = config.val_path

    dataloader = MNISTDataloader(
        data_path=data_file,
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
        output_type=output_type,
        device=device
    )
    model.to(device=device)
    model = load_model(model, model_path)
    model.eval()
    
    return model, dataloader

def evaluate_dataset_with_extrapolation(data_split):
    model, dataloader = get_model_and_dataloader(data_split, output_type='error', extrap_time=4)
    
    samples_collected = 0
    
    with torch.no_grad(): 
        for step, frames in enumerate(dataloader, start=1):
            if step > 3: break
            initial_states = model.get_initial_states(input_shape)
            output_list, hidden_states_list = model(frames.to(device), initial_states)
            plot_hidden_states_list(hidden_states_list, frames, step, 'prediction', debug_images_dir)
            del initial_states, output_list, hidden_states_list
            

def calculate_model_metrics(data_split, start_frame_idx, end_frame_idx, extrap_time=None):
    model, dataloader = get_model_and_dataloader(data_split, output_type='prediction', extrap_time=extrap_time)
    mse_values, psnr_values, ssim_values, l1_values = [], [], [], []

    for step, frames in enumerate(dataloader, start=1):
        initial_states = model.get_initial_states(input_shape)
        output_list, _ = model(frames.to(device), initial_states)
        prediction_frames = torch.stack(output_list, dim=1)  # (batch, nt, ...)
        prediction_frames = prediction_frames.detach().cpu().numpy() 
        
        mse_value = _calculate_mse(prediction_frames, frames, start_frame_idx, end_frame_idx)
        psnr_value = _calculate_psnr(prediction_frames, frames, start_frame_idx, end_frame_idx)
        ssim_value = _calculate_ssim(prediction_frames, frames, start_frame_idx, end_frame_idx)
        l1_value = _calculate_l1(prediction_frames, frames, start_frame_idx, end_frame_idx)
        
        mse_values.append(mse_value)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        l1_values.append(l1_value)
        
        del initial_states, output_list
        
    mse_mean = np.mean(mse_values)
    psnr_mean = np.mean(psnr_values)
    ssim_mean = np.mean(ssim_values)
    l1_mean = np.mean(l1_values)
    
    return mse_mean, psnr_mean, ssim_mean, l1_mean

def calculate_naive_prev_frame_metrics(data_split, start_frame_idx, end_frame_idx):
    _, dataloader = get_model_and_dataloader(data_split)
    mse_values, psnr_values, ssim_values, l1_values = [], [], [], []
    
    for step, frames in enumerate(dataloader, start=1):
        prediction_frames = np.zeros_like(frames)
        prediction_frames[:, 1:, ...] = frames[:, :-1, ...].numpy()
        
        mse_value = _calculate_mse(prediction_frames, frames, start_frame_idx, end_frame_idx)
        psnr_value = _calculate_psnr(prediction_frames, frames, start_frame_idx, end_frame_idx)
        ssim_value = _calculate_ssim(prediction_frames, frames, start_frame_idx, end_frame_idx)
        l1_value = _calculate_l1(prediction_frames, frames, start_frame_idx, end_frame_idx)
        
        mse_values.append(mse_value)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        l1_values.append(l1_value)
        
    mse_mean = np.mean(mse_values)
    psnr_mean = np.mean(psnr_values)
    ssim_mean = np.mean(ssim_values)
    l1_mean = np.mean(l1_values)
    
    return mse_mean, psnr_mean, ssim_mean, l1_mean

def predict_next_frame_optical_flow(prev_frame, curr_frame):
    # Convert from float [0,1] to uint8 [0,255] if needed
    if prev_frame.dtype != np.uint8:
        prev_frame = (prev_frame.cpu().numpy() * 255).astype(np.uint8).transpose(1,2,0)
    if curr_frame.dtype != np.uint8:
        curr_frame = (curr_frame.cpu().numpy() * 255).astype(np.uint8).transpose(1,2,0)
    
    # Convert to grayscale
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
    
    # Calculate optical flow (Farneback method)
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray, None,
        pyr_scale=0.5, levels=3, winsize=15,
        iterations=3, poly_n=5, poly_sigma=1.2, flags=0
    )
    
    # Create mesh grid for warping
    h, w = curr_frame.shape[:2]
    flow_map = np.zeros((h, w, 2), dtype=np.float32)
    flow_map[:, :, 0] = np.arange(w)  # x coordinates
    flow_map[:, :, 1] = np.arange(h)[:, np.newaxis]  # y coordinates
    
    # Add flow to get new positions
    flow_map += flow
    
    # Warp current frame to predict next frame
    predicted_frame = cv2.remap(
        curr_frame, 
        flow_map[:, :, 0], 
        flow_map[:, :, 1],
        cv2.INTER_LINEAR
    )
    
    # Convert back to float [0,1] for consistency with input
    predicted_frame = predicted_frame.astype(np.float32) / 255.0
    
    return predicted_frame.transpose(2, 0, 1)

def calculate_optical_flow_metrics(data_split, start_frame_idx, end_frame_idx):
    _, dataloader = get_model_and_dataloader(data_split)
    mse_values, psnr_values, ssim_values, l1_values = [], [], [], []

    for step, frames in enumerate(dataloader, start=1):
        prediction_frames = np.zeros_like(frames)
        batch_size, nt = frames.shape[0], frames.shape[1]
        
        for batch_idx in range(batch_size):
            video = frames[batch_idx]
            for t in range(2, nt):
                pred = predict_next_frame_optical_flow(video[t-2], video[t-1])
                prediction_frames[batch_idx, t, ...] = pred
            
        mse_value = _calculate_mse(prediction_frames, frames, start_frame_idx, end_frame_idx)
        psnr_value = _calculate_psnr(prediction_frames, frames, start_frame_idx, end_frame_idx)
        ssim_value = _calculate_ssim(prediction_frames, frames, start_frame_idx, end_frame_idx)
        l1_value = _calculate_l1(prediction_frames, frames, start_frame_idx, end_frame_idx)
        
        mse_values.append(mse_value)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        l1_values.append(l1_value)
        
    mse_mean = np.mean(mse_values)
    psnr_mean = np.mean(psnr_values)
    ssim_mean = np.mean(ssim_values)
    l1_mean = np.mean(l1_values)
    
    return mse_mean, psnr_mean, ssim_mean, l1_mean

def _ensure_torch(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data
    raise TypeError("data must be numpy or torch datatype")
        
def _take_eval_frames_and_flatten_batch_nt(prediction_frames, gt_frames, start_frame_idx=1, end_frame_idx=-1):
    nt = gt_frames.shape[1]
    prediction_frames = _ensure_torch(prediction_frames)
    gt_frames = _ensure_torch(gt_frames)
    
    if end_frame_idx == -1:
        end_frame_idx = nt - 1
    
    prediction_frames_ = prediction_frames[:, start_frame_idx:end_frame_idx+1,...]
    gt_frames_ = gt_frames[:, start_frame_idx:end_frame_idx+1,...]
    
    b, t, c, h, w = prediction_frames_.shape
    prediction_frames_ = prediction_frames_.reshape((b*t, c, h, w))
    gt_frames_ = gt_frames_.reshape((b*t, c, h, w))
    
    return prediction_frames_, gt_frames_

def _calculate_mse(prediction_frames, gt_frames, start_frame_idx=1, end_frame_idx=-1):
    prediction_frames_, gt_frames_ = _take_eval_frames_and_flatten_batch_nt(prediction_frames, gt_frames, start_frame_idx, end_frame_idx)
    mse_value = torch.mean((prediction_frames_ - gt_frames_)**2)
    
    return mse_value

def _calculate_psnr(prediction_frames, gt_frames, start_frame_idx=1, end_frame_idx=-1):
    prediction_frames_, gt_frames_ = _take_eval_frames_and_flatten_batch_nt(prediction_frames, gt_frames, start_frame_idx, end_frame_idx)
    psnr_value = peak_signal_noise_ratio(prediction_frames_, gt_frames_, data_range=1.0)
    
    return psnr_value
    
def _calculate_ssim(prediction_frames, gt_frames, start_frame_idx=1, end_frame_idx=-1):
    prediction_frames_, gt_frames_ = _take_eval_frames_and_flatten_batch_nt(prediction_frames, gt_frames, start_frame_idx, end_frame_idx)
    ssim_value = structural_similarity_index_measure(prediction_frames_, gt_frames_, data_range=1.0)
    
    return ssim_value

def _calculate_l1(prediction_frames, gt_frames, start_frame_idx=1, end_frame_idx=-1):
    prediction_frames_, gt_frames_ = _take_eval_frames_and_flatten_batch_nt(prediction_frames, gt_frames, start_frame_idx, end_frame_idx)
    l1_value = torch.mean(torch.abs(prediction_frames_ - gt_frames_))
    
    return l1_value 

def evaluate():
    evaluate_dataset('train')
    evaluate_dataset('val')

def capture_results_for_configs():
    """
    Captures results for all configurations and returns a pandas DataFrame.
    
    Configurations:
    1) 4->4: extrap_time=4, start=4, end=4
    2) 4->7: extrap_time=4, start=4, end=7
    3) 1->1: extrap_time=1, start=1, end=1 (skip optical flow)
    4) 1->4: extrap_time=1, start=1, end=4 (skip optical flow)
    
    For each config, computes metrics for:
    - Naive prev frame baseline
    - Optical flow baseline (only for configs 1 and 2)
    - Model with extrapolation
    - Model without extrapolation
    
    Returns:
        pd.DataFrame: Results with columns [Experiment, Method, MSE, PSNR, SSIM, L1]
    """
    data_split = 'val'
    
    configs = [
        {'name': '4->4', 'extrap_time': 4, 'start': 4, 'end': 4, 'use_optical_flow': True},
        {'name': '4->7', 'extrap_time': 4, 'start': 4, 'end': 7, 'use_optical_flow': True},
        {'name': '1->1', 'extrap_time': 1, 'start': 1, 'end': 1, 'use_optical_flow': False},
        {'name': '1->4', 'extrap_time': 1, 'start': 1, 'end': 4, 'use_optical_flow': False},
    ]
    
    results = []
    
    for config in configs:
        print(f"\n{'='*60}")
        print(f"Running config: {config['name']}")
        print(f"{'='*60}")
        
        name = config['name']
        extrap_time = config['extrap_time']
        start_frame_idx = config['start']
        end_frame_idx = config['end']
        use_optical_flow = config['use_optical_flow']
        
        # Naive prev frame baseline
        print(f"[{name}] Computing naive prev frame metrics...")
        mse, psnr, ssim, l1 = calculate_naive_prev_frame_metrics(data_split, start_frame_idx, end_frame_idx)
        results.append({
            'Experiment': name,
            'Method': 'Naive Prev Frame',
            'MSE': float(mse),
            'PSNR': float(psnr),
            'SSIM': float(ssim),
            'L1': float(l1)
        })
        print(f"[{name}] Naive Prev Frame - MSE: {mse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, L1: {l1:.4f}")
        
        # Optical flow baseline (only for configs 1 and 2)
        if use_optical_flow:
            print(f"[{name}] Computing optical flow metrics...")
            mse, psnr, ssim, l1 = calculate_optical_flow_metrics(data_split, start_frame_idx, end_frame_idx)
            results.append({
                'Experiment': name,
                'Method': 'Optical Flow',
                'MSE': float(mse),
                'PSNR': float(psnr),
                'SSIM': float(ssim),
                'L1': float(l1)
            })
            print(f"[{name}] Optical Flow - MSE: {mse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, L1: {l1:.4f}")
        
        # Model with extrapolation
        print(f"[{name}] Computing model metrics with extrapolation (extrap_time={extrap_time})...")
        mse, psnr, ssim, l1 = calculate_model_metrics(data_split, start_frame_idx, end_frame_idx, extrap_time)
        results.append({
            'Experiment': name,
            'Method': f'Model (extrap={extrap_time})',
            'MSE': float(mse),
            'PSNR': float(psnr),
            'SSIM': float(ssim),
            'L1': float(l1)
        })
        print(f"[{name}] Model (extrap={extrap_time}) - MSE: {mse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, L1: {l1:.4f}")
        
        # Model without extrapolation
        print(f"[{name}] Computing model metrics without extrapolation...")
        mse, psnr, ssim, l1 = calculate_model_metrics(data_split, start_frame_idx, end_frame_idx, None)
        results.append({
            'Experiment': name,
            'Method': 'Model (no extrap)',
            'MSE': float(mse),
            'PSNR': float(psnr),
            'SSIM': float(ssim),
            'L1': float(l1)
        })
        print(f"[{name}] Model (no extrap) - MSE: {mse:.4f}, PSNR: {psnr:.4f}, SSIM: {ssim:.4f}, L1: {l1:.4f}")
    
    df = pd.DataFrame(results)
    
    # Save to CSV
    os.makedirs('./eval_results', exist_ok=True)
    df.to_csv('./eval_results/metrics.csv', index=False)
    
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    print(df.to_string(index=False))
    print(f"\nResults saved to ./eval_results/metrics.csv")
    
    return df

if __name__ == '__main__':
    # Run all configured experiments and get results as DataFrame
    df = capture_results_for_configs()
