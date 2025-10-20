import numpy as np
import matplotlib.pyplot as plt
import os

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

def plot_hidden_states_list(hidden_states_list, frames, epoch, data_split='train'):
    
    output_dir = f'./debug_layer_images/{data_split}/{epoch}'
    os.makedirs(output_dir, exist_ok=True)
    
    num_timesteps = len(hidden_states_list)
    num_layers = len(hidden_states_list[0]['R'])

    for batch_idx in range(min(frames.shape[0], 5)):
        os.makedirs(os.path.join(output_dir, str(batch_idx)), exist_ok=True)
        plot_layer(hidden_states_list, frames, 'R', output_dir, batch_idx)
        plot_layer(hidden_states_list, frames, 'A', output_dir, batch_idx)
        plot_layer(hidden_states_list, frames, 'Ahat', output_dir, batch_idx)
        plot_layer(hidden_states_list, frames, 'E', output_dir, batch_idx)
        
        plot_input_vs_prediction(hidden_states_list, frames, output_dir, batch_idx)
    