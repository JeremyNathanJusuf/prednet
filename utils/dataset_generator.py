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

    def generate_random_video(self, n_digits, max_scale=1.2):
        # generate a random video with n_digits moving in random directions and random colors
        nt = self.nt
        h = self.h
        w = self.w
        
        iou = 1
        coverage = 0
        
        while iou > 0.2 or coverage < 0.1:
            iou = 0
            coverage = 0
            
            digits = [self.get_random_digit() for _ in range(n_digits)]
            original_x_dim, original_y_dim = digits[0].shape[1], digits[0].shape[2]
            
            video = np.zeros((nt, 3, h, w))
            directions_with_speed = np.random.randint(0, 10, size=(n_digits, 2))
            init_w_arr, init_h_arr = np.random.randint(0, w - original_x_dim, size=(n_digits,)), np.random.randint(0, h - original_y_dim, size=(n_digits,))
            colors = np.random.rand(n_digits, 3)
            scales = np.random.uniform(low=1, high=max_scale, size=(n_digits,))
            
            avg_color = np.mean(colors, axis=0)
            if np.mean(avg_color) > 0.05:
                background_color = np.expand_dims(np.random.rand(3) * 0.2, axis=(1, 2))
            else:
                background_color = np.expand_dims(0.8 + np.random.rand(3) * 0.2, axis=(1, 2))
            
            for t in range(nt):
                frame = np.zeros((3, h, w)) + background_color
                for i in range(n_digits):
                    x_, y_  = init_w_arr[i], init_h_arr[i]
                    x, y = x_ + t * directions_with_speed[i][0], y_ + t * directions_with_speed[i][1]
                    x_dim, y_dim = int(original_x_dim * scales[i]), int(original_y_dim * scales[i])

                    color = np.expand_dims(colors[i], axis=(1, 2))
                        
                    input_digit = np.repeat(digits[i], 3, axis=0)
                    input_digit = cv2.resize(input_digit.transpose(1, 2, 0), (x_dim, y_dim), interpolation=cv2.INTER_CUBIC).transpose(2, 0, 1)

                    if x > w or y > h:
                        continue
                    if x+x_dim > w:
                        input_digit = input_digit[:, :, :w-x]
                    if y+y_dim > h:
                        input_digit = input_digit[:, :h-y, :]
                    if x < 0:
                        input_digit = input_digit[:, :, -x:]
                    if y < 0:
                        input_digit = input_digit[:, -y:, :]

                    frame_mask = (frame[:, y:y+y_dim, x:x+x_dim] > background_color + 0.01)
                    frame[:, y:y+y_dim, x:x+x_dim] = np.clip(frame[:, y:y+y_dim, x:x+x_dim] + input_digit * color, 0, 1)
                    digit_mask = input_digit > 0
                    
                    intersection = (frame_mask & digit_mask).sum()
                    union = (frame_mask | digit_mask).sum()
                    iou = max(intersection / union if union > 0 else 0, iou)
                    
                    coverage = max(frame_mask.sum() / (h * w), coverage)

                video[t] = frame
                
            
        return video
        
    def generate_dataset(self, num_samples, n_digits, max_scale=1.2):
        save_path = f'./custom_dataset/mnist_dataset_{num_samples}_{self.nt}.npy'
        if os.path.exists(save_path):
            return np.load(save_path)
        
        dataset = []
        for _ in range(num_samples):
            video = self.generate_random_video(n_digits, max_scale)
            dataset.append(video)
            
        dataset = np.stack(dataset)
        
        # save dataset as npy file
        os.makedirs('./custom_dataset', exist_ok=True)
        np.save(save_path, dataset)
        print("dataset.shape", dataset.shape)
        return dataset
    
if __name__ == '__main__':
    generator = MovingMnistDatasetGenerator(nt=5, h=64, w=64)
    dataset = generator.generate_dataset(num_samples=10000, n_digits=3, max_scale=1.2)
    print(dataset.shape)