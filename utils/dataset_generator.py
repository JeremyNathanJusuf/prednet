import cv2
import numpy as np 
import torchvision.transforms as transforms
from torchvision import datasets
import os
import matplotlib.pyplot as plt

class MovingMnistDatasetGenerator():
    def __init__(self, nt, h, w):
        # load the mnist dataset
        mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        self.mnist_dataset = np.stack([img for img, _ in mnist_dataset])
        self.nt = nt
        self.h = h
        self.w = w
    
    def get_random_digit(self):
        frame = self.mnist_dataset[np.random.randint(0, len(self.mnist_dataset))]
        return frame

    def generate_random_video(self, n_digits, min_scale=0.8, max_scale=1.2, num_dilate_iterations=2):
        # generate a random video with n_digits moving in random directions and random colors
        nt = self.nt
        h = self.h
        w = self.w
        
        iou = 1
        coverage = 0
        
        while iou > 0.2 or coverage < 0.15:
            iou = 0
            coverage = 0
            
            digits = [self.get_random_digit() for _ in range(n_digits)]
            original_x_dim, original_y_dim = digits[0].shape[1], digits[0].shape[2]
            
            video = np.zeros((nt, 3, h, w))
            directions_with_speed = np.random.randint(-10, 10, size=(n_digits, 2))
            scales = np.random.uniform(low=min_scale, high=max_scale, size=(n_digits,))
            init_x_arr, init_y_arr = [], []
            
            for i in range(n_digits):
                # Calculate scaled dimensions
                x_dim = int(original_x_dim * scales[i])
                y_dim = int(original_y_dim * scales[i])
                
                # Ensure at least 2/3 of digit is within frame (at most 1/3 can be cut off)
                # x must be in [-x_dim/3, w - 2*x_dim/3] to keep 2/3 visible
                # y must be in [-y_dim/3, h - 2*y_dim/3] to keep 2/3 visible
                init_x = int(np.random.uniform(-x_dim/3, w - 2*x_dim/3))
                init_y = int(np.random.uniform(-y_dim/3, h - 2*y_dim/3))
                
                init_x_arr.append(init_x)
                init_y_arr.append(init_y)
                
            colors = np.random.rand(n_digits, 3)
            
            avg_color = np.mean(colors, axis=0)
            if np.mean(avg_color) > 0.05:
                background_color = np.expand_dims(np.random.rand(3) * 0.2, axis=(1, 2))
            else:
                background_color = np.expand_dims(0.8 + np.random.rand(3) * 0.2, axis=(1, 2))
            
            for t in range(nt):
                frame = np.zeros((3, h, w)) + background_color
                for i in range(n_digits):
                    x_, y_  = init_x_arr[i], init_y_arr[i]
                    x, y = x_ + t * directions_with_speed[i][0], y_ + t * directions_with_speed[i][1]
                    
                    x_dim, y_dim = int(original_x_dim * scales[i]), int(original_y_dim * scales[i])
                    assert x_dim > 28 and y_dim > 28, f"bruh {x_dim} {y_dim}"
                    color = np.expand_dims(colors[i], axis=(1, 2))
                        
                    input_digit = np.repeat(digits[i], 3, axis=0)
                    
                    # Resize the digit with smooth interpolation for nice edges
                    input_digit = cv2.resize(input_digit.transpose(1, 2, 0), (x_dim, y_dim), interpolation=cv2.INTER_LINEAR).transpose(2, 0, 1)
                    
                    # Dilate the digit
                    kernel = np.ones((3, 3), np.uint8)
                    input_digit = cv2.dilate(input_digit, kernel, iterations=num_dilate_iterations)
                    
                    # Hard threshold to remove texture while keeping smooth edges
                    input_digit = np.where(input_digit > 0.1, 1, 0)

                    # Calculate the overlapping region between digit and frame
                    # Digit coordinates
                    digit_x_start = max(0, -x)
                    digit_y_start = max(0, -y)
                    digit_x_end = min(x_dim, w - x)
                    digit_y_end = min(y_dim, h - y)
                    
                    # Frame coordinates
                    frame_x_start = max(0, x)
                    frame_y_start = max(0, y)
                    frame_x_end = min(w, x + x_dim)
                    frame_y_end = min(h, y + y_dim)
                    
                    # Check if there's any overlap
                    if digit_x_end <= digit_x_start or digit_y_end <= digit_y_start:
                        continue
                    if frame_x_end <= frame_x_start or frame_y_end <= frame_y_start:
                        continue
                    
                    # Extract the visible part of the digit
                    visible_digit = input_digit[:, digit_y_start:digit_y_end, digit_x_start:digit_x_end]
                    
                    frame_mask = (frame[:, frame_y_start:frame_y_end, frame_x_start:frame_x_end] > background_color + 0.01)
                    frame[:, frame_y_start:frame_y_end, frame_x_start:frame_x_end] = np.clip(
                        frame[:, frame_y_start:frame_y_end, frame_x_start:frame_x_end] + visible_digit * color, 0, 1)
                    digit_mask = visible_digit > 0
                    
                    intersection = (frame_mask & digit_mask).sum()
                    union = (frame_mask | digit_mask).sum()
                    iou = max(intersection / union if union > 0 else 0, iou)
                    
                    coverage = max(frame_mask.sum() / (h * w), coverage)

                video[t] = frame
            
        return video
        
    def generate_dataset(self, num_samples, n_digits, min_scale=0.8, max_scale=1.2, num_dilate_iterations=2):
        save_path = f'./custom_dataset/mnist_dataset_{num_samples}_{self.nt}.npy'
        if os.path.exists(save_path):
            print(f"dataset already exists at {save_path}")
            return np.load(save_path), save_path
        
        dataset = []
        for _ in range(num_samples):
            video = self.generate_random_video(n_digits, min_scale, max_scale, num_dilate_iterations)
            dataset.append(video)
            
        dataset = np.stack(dataset)
        
        # save dataset as npy file
        os.makedirs('./custom_dataset', exist_ok=True)
        np.save(save_path, dataset)
        print("dataset.shape", dataset.shape)
        return dataset, save_path
    
if __name__ == '__main__':
    generator = MovingMnistDatasetGenerator(nt=5, h=64, w=64)
    dataset = generator.generate_dataset(num_samples=10000, n_digits=3, max_scale=1.2)
    print(dataset.shape)