import os
import numpy as np
import torch
import matplotlib.pyplot as plt

from utils.mnist import MNISTDataloader
from utils.plot import plot_hidden_states_list
from prednet import Prednet

import torch
from torchmetrics.functional.image import peak_signal_noise_ratio, structural_similarity_index_measure

device = 'cuda' if torch.cuda.is_available() else 'mps' if torch.mps.is_available() else 'cpu'

n_samples = 10
batch_size = 8
nt = 8

n_channels, im_height, im_width = (3, 64, 64)  # RGB images
input_shape = (batch_size, n_channels, im_height, im_width)
A_stack_sizes = (n_channels, 32, 64, 128, 256)
R_stack_sizes = A_stack_sizes
A_filter_sizes = (3, 3, 3, 3)
Ahat_filter_sizes = (3, 3, 3, 3, 3)
R_filter_sizes = (3, 3, 3, 3, 3)

model_path = './checkpoints_14_nov_1/epoch_100.pth'  # TODO: Update to appropriate checkpoint

def load_model(model, model_path):
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def get_model_and_dataloader(data_split, output_type='error'):
    train_path, val_path = "./custom_dataset/mnist_train.npy", "./custom_dataset/mnist_val.npy"
    
    # Choose the appropriate data file
    if data_split == 'train':
        data_file = train_path
    else:
        data_file = val_path

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
        pixel_max=1.0,
        lstm_activation='relu', 
        A_activation='relu', 
        extrap_time=None, 
        output_type=output_type,
        device=device
    )
    model.to(device=device)
    model = load_model(model, model_path)
    model.eval()
    
    return model, dataloader

def evaluate_dataset(data_split):
    model, dataloader = get_model_and_dataloader(data_split)
    
    samples_collected = 0
    
    with torch.no_grad(): 
        for step, frames in enumerate(dataloader, start=1):
            # if step > 3: break
            initial_states = model.get_initial_states(input_shape)
            
            output_list, hidden_states_list = model(frames.to(device), initial_states)
            
        
            plot_hidden_states_list(hidden_states_list, frames, step, 'val')
            
            del initial_states, output_list, hidden_states_list

def calculate_model_metrics(data_split):
    model, dataloader = get_model_and_dataloader(data_split, output_type='prediction')
    mse_values, psnr_values, ssim_values = [], [], []

    for step, frames in enumerate(dataloader, start=1):
        initial_states = model.get_initial_states(input_shape)
        output_list, _ = model(frames.to(device), initial_states)
        prediction_frames = torch.stack(output_list, dim=1)  # (batch, nt, ...)
        prediction_frames = prediction_frames.detach().cpu().numpy() 
        
        mse_value = _calculate_mse(prediction_frames, frames)
        psnr_value = _calculate_psnr(prediction_frames, frames)
        ssim_value = _calculate_ssim(prediction_frames, frames)
        
        mse_values.append(mse_value)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        
        del initial_states, output_list
        
    mse_mean = np.mean(mse_values)
    psnr_mean = np.mean(psnr_values)
    ssim_mean = np.mean(ssim_values)
    
    return mse_mean, psnr_mean, ssim_mean

def calculate_naive_prev_frame(data_split):
    _, dataloader = get_model_and_dataloader(data_split, output_type='prediction')
    mse_values, psnr_values, ssim_values = [], [], []
    
    for step, frames in enumerate(dataloader, start=1):
        prediction_frames = np.zeros_like(frames)
        prediction_frames[:, 1:, ...] = frames[:, :-1, ...]
        
        mse_value = _calculate_mse(prediction_frames, frames)
        psnr_value = _calculate_psnr(prediction_frames, frames)
        ssim_value = _calculate_ssim(prediction_frames, frames)
        
        mse_values.append(mse_value)
        psnr_values.append(psnr_value)
        ssim_values.append(ssim_value)
        
    mse_mean = np.mean(mse_values)
    psnr_mean = np.mean(psnr_values)
    ssim_mean = np.mean(ssim_values)
    
    return mse_mean, psnr_mean, ssim_mean

def _ensure_torch(data):
    if isinstance(data, np.ndarray):
        return torch.from_numpy(data)
    elif isinstance(data, torch.Tensor):
        return data
    raise TypeError("data must be numpy or torch datatype")
        
def _take_eval_frames_and_flatten_batch_nt(prediction_frames, gt_frames, start_frame_idx=1):
    nt = gt_frames.shape[1]
    prediction_frames_ = prediction_frames[:, start_frame_idx:,...]
    gt_frames_ = gt_frames[:, start_frame_idx:,...]
    
    b, t, c, h, w = prediction_frames_.shape
    prediction_frames_ = prediction_frames_.reshape((b*t, c, h, w))
    gt_frames_ = gt_frames_.reshape((b*t, c, h, w))
    
    prediction_frames_ = _ensure_torch(prediction_frames_)
    gt_frames_ = _ensure_torch(gt_frames_)
    
    return prediction_frames_, gt_frames_

def _calculate_mse(prediction_frames, gt_frames, start_frame_idx=1):
    prediction_frames_, gt_frames_ = _take_eval_frames_and_flatten_batch_nt(prediction_frames, gt_frames, start_frame_idx)
    mse_value = torch.mean((prediction_frames_ - gt_frames_)**2)
    
    return mse_value

def _calculate_psnr(prediction_frames, gt_frames, start_frame_idx=1):
    prediction_frames_, gt_frames_ = _take_eval_frames_and_flatten_batch_nt(prediction_frames, gt_frames, start_frame_idx)
    psnr_value = peak_signal_noise_ratio(prediction_frames_, gt_frames_, data_range=1.0)
    
    return psnr_value
    
def _calculate_ssim(prediction_frames, gt_frames, start_frame_idx=1):
    prediction_frames_, gt_frames_ = _take_eval_frames_and_flatten_batch_nt(prediction_frames, gt_frames, start_frame_idx)
    ssim_value = structural_similarity_index_measure(prediction_frames_, gt_frames_, data_range=1.0)
    
    return ssim_value 

def evaluate():
    evaluate_dataset('train')
    evaluate_dataset('val')

if __name__ == '__main__':
    print(calculate_naive_prev_frame('val'))
    print(calculate_model_metrics('val'))