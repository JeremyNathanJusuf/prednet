import os
import torch


def save_model(model, optimizer, epoch, avg_train_error, checkpoint_dir, avg_val_error=None):
    """Save model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number
        avg_train_error: Average training error
        checkpoint_dir: Directory to save checkpoints
        avg_val_error: Optional average validation error
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, f'epoch_{epoch}.pth')
    
    checkpoint_data = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_train_error': avg_train_error,
    }
    
    if avg_val_error is not None:
        checkpoint_data['avg_val_error'] = avg_val_error
    
    torch.save(checkpoint_data, model_path)
    
    print(f'Saved model to: {model_path}')


def save_best_val_model(model, optimizer, epoch, avg_train_error, avg_val_error, checkpoint_dir):
    """Save the best validation model checkpoint.
    
    Args:
        model: The model to save
        optimizer: The optimizer state to save
        epoch: Current epoch number
        avg_train_error: Average training error
        avg_val_error: Average validation error
        checkpoint_dir: Directory to save checkpoints
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    model_path = os.path.join(checkpoint_dir, 'epoch_best_val.pth')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'avg_train_error': avg_train_error,
        'avg_val_error': avg_val_error,
    }, model_path)
    
    print(f'Saved best validation model to: {model_path} (epoch {epoch}, val_error: {avg_val_error:.6f})')


def load_model(model, model_path, device, optimizer=None, lr_scheduler=None, load_optimizer=True):
    """Load model weights from checkpoint or pretrained weights.
    
    Supports two formats:
    1. Checkpoint dict with 'model_state_dict' key (from previous training)
    2. Direct state_dict (from pretrained weights .pkl file)
    
    Args:
        model: The model to load weights into
        model_path: Path to the checkpoint file
        device: Device to load the model onto
        optimizer: Optional optimizer to load state into
        lr_scheduler: Optional learning rate scheduler to update
        load_optimizer: Whether to load optimizer state (default True)
    
    Returns:
        model, optimizer, lr_scheduler (optimizer and lr_scheduler may be None)
    """
    checkpoint = torch.load(model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if load_optimizer and optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            
            if lr_scheduler is not None and hasattr(lr_scheduler, 'last_epoch'):
                lr_scheduler.last_epoch = checkpoint["epoch"] - 1
                lr_scheduler.step()
        else:
            print(f"Loading model weights only (not optimizer state) from {model_path}")
    else:
        model.load_state_dict(checkpoint)
        print(f"Loaded pretrained weights from {model_path}")
    
    return model, optimizer, lr_scheduler

