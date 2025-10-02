"""Quick diagnostic to check if model is actually learning"""
import torch
import numpy as np
from prednet import Prednet

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Model parameters
n_channels, im_height, im_width = (1, 64, 64)
batch_size = 1
A_stack_sizes = (n_channels, 48, 96)
R_stack_sizes = A_stack_sizes
A_filter_sizes = (3, 3)
Ahat_filter_sizes = (3, 3, 3)
R_filter_sizes = (3, 3, 3)

# Create model
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
model.to(device)
model.eval()

# Create dummy input
dummy_input = torch.ones(1, 5, 1, 64, 64).to(device) * 0.5  # Gray image

# Get prediction
input_shape = (1, n_channels, im_height, im_width)
initial_states = model.get_initial_states(input_shape)

with torch.no_grad():
    predictions = model(dummy_input, initial_states)
    pred_stack = torch.stack(predictions)
    
    print("=" * 60)
    print("MODEL DIAGNOSTIC")
    print("=" * 60)
    print(f"Input: mean={dummy_input.mean():.4f}, std={dummy_input.std():.4f}")
    print(f"Prediction: mean={pred_stack.mean():.4f}, std={pred_stack.std():.4f}")
    print(f"Prediction: min={pred_stack.min():.4f}, max={pred_stack.max():.4f}")
    print(f"Prediction shape: {pred_stack.shape}")
    print()
    
    # Check if predictions are constant
    unique_values = torch.unique(pred_stack)
    print(f"Number of unique values in prediction: {len(unique_values)}")
    if len(unique_values) < 10:
        print(f"Unique values: {unique_values[:10]}")
        print("⚠️  WARNING: Very few unique values - model might be stuck!")
    
    # Check if predictions have any variation
    if pred_stack.std() < 0.001:
        print("⚠️  WARNING: Predictions are nearly constant!")
        print("   Model is not producing meaningful output.")
    else:
        print("✓ Predictions have variation (good)")
    
    print("=" * 60)
    
# Now check if loaded model works
print("\nChecking trained model from checkpoint...")
try:
    checkpoint = torch.load('./debug_checkpoints/epoch_20.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    with torch.no_grad():
        predictions = model(dummy_input, initial_states)
        pred_stack = torch.stack(predictions)
        
        print(f"Loaded model prediction: mean={pred_stack.mean():.4f}, std={pred_stack.std():.4f}")
        print(f"Loaded model prediction: min={pred_stack.min():.4f}, max={pred_stack.max():.4f}")
        
        if pred_stack.std() < 0.001:
            print("⚠️  PROBLEM: Trained model still outputs constant values!")
            print("   Training is not working properly.")
        else:
            print("✓ Trained model produces varying predictions")
            
except FileNotFoundError:
    print("Checkpoint not found - train the model first")
    
print("\n" + "=" * 60)

