import cv2
import numpy as np 
import torchvision.transforms as transforms
from torchvision import datasets
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from utils.mnist import split_custom_mnist_data


class GrayscaleMovingMnistGenerator:
    """
    Simple grayscale Moving MNIST generator that preserves original digit textures.
    - Single channel (grayscale)
    - Original MNIST texture preserved (not solid colors)
    - Simple compositing: digits overlay on black background
    - Output shape: (nt, 1, h, w)
    - Digits are dispersed across the frame (not clustered)
    """
    
    def __init__(self, nt, h, w):
        # Load the MNIST dataset
        mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        self.mnist_dataset = [img.numpy().squeeze() for img, _ in mnist_dataset]  # Store as 2D arrays
        self.nt = nt
        self.h = h
        self.w = w
    
    def generate_random_video(self, n_digits, min_scale=1.0, max_scale=2.0):
        """
        Generate a grayscale moving MNIST video with dispersed digits.
        
        Args:
            n_digits: Number of digits in the video
            min_scale, max_scale: Scale range for digits
        
        Returns:
            video: numpy array of shape (nt, 1, h, w)
        """
        nt, h, w = self.nt, self.h, self.w
        
        digits = [self.mnist_dataset[np.random.randint(0, len(self.mnist_dataset))] for _ in range(n_digits)]
        original_size = 28  # MNIST digits are 28x28
        
        # Random scales and velocities
        scales = np.random.uniform(min_scale, max_scale, size=n_digits)
        velocities = np.random.randint(-4, 5, size=(n_digits, 2))  # (vx, vy) for each digit
        
        # Initial positions with mild dispersion (not too clustered, not too sparse)
        positions = []
        min_distance = 15  # Minimum distance between digit centers
        
        for i in range(n_digits):
            digit_size = int(original_size * scales[i])
            
            # Try to find a position that's not too close to existing digits
            for attempt in range(20):
                x = np.random.randint(0, max(1, w - digit_size))
                y = np.random.randint(0, max(1, h - digit_size))
                
                # Check distance from other digits
                too_close = False
                for px, py in positions:
                    dist = np.sqrt((x - px)**2 + (y - py)**2)
                    if dist < min_distance:
                        too_close = True
                        break
                
                if not too_close or attempt == 19:  # Accept if far enough or last attempt
                    break
            
            positions.append([x, y])
        
        positions = np.array(positions, dtype=float)
        
        # Generate video frames
        video = np.zeros((nt, 1, h, w), dtype=np.float32)
        
        for t in range(nt):
            frame = np.zeros((h, w), dtype=np.float32)
            
            for i in range(n_digits):
                # Current position
                x, y = int(positions[i, 0]), int(positions[i, 1])
                
                # Resize digit
                digit_size = int(original_size * scales[i])
                digit = cv2.resize(digits[i], (digit_size, digit_size), interpolation=cv2.INTER_LINEAR)
                
                # Calculate visible region
                x_start_frame = max(0, x)
                y_start_frame = max(0, y)
                x_end_frame = min(w, x + digit_size)
                y_end_frame = min(h, y + digit_size)
                
                x_start_digit = max(0, -x)
                y_start_digit = max(0, -y)
                x_end_digit = x_start_digit + (x_end_frame - x_start_frame)
                y_end_digit = y_start_digit + (y_end_frame - y_start_frame)
                
                # Skip if no overlap
                if x_end_frame <= x_start_frame or y_end_frame <= y_start_frame:
                    continue
                
                # Get visible part of digit
                visible_digit = digit[y_start_digit:y_end_digit, x_start_digit:x_end_digit]
                
                # Simple max compositing (digit on top of what's already there)
                current_region = frame[y_start_frame:y_end_frame, x_start_frame:x_end_frame]
                frame[y_start_frame:y_end_frame, x_start_frame:x_end_frame] = np.maximum(current_region, visible_digit)
            
            video[t, 0] = frame
            
            # Update positions for next frame
            positions += velocities
        
        return video
    
    def generate_dataset(self, num_samples, n_digits, min_scale=1.0, max_scale=2.0):
        """
        Generate a dataset of grayscale moving MNIST videos.
        Saves with 1 channel. Use load_dataset_with_channel_check() to load and expand to 3 channels.
        
        Args:
            num_samples: Number of videos to generate
            n_digits: Number of digits per video
            min_scale, max_scale: Scale range for digits
        
        Returns:
            dataset: numpy array of shape (num_samples, nt, 1, h, w)
            save_path: Path where dataset was saved
        """
        save_path = f'./custom_dataset/mnist_grayscale_{num_samples}_{self.nt}.npy'
        if os.path.exists(save_path):
            print(f"Dataset already exists at {save_path}")
            return np.load(save_path), save_path
        
        dataset = []
        for _ in tqdm(range(num_samples), desc="Generating videos"):
            video = self.generate_random_video(n_digits, min_scale, max_scale)
            dataset.append(video)
        
        dataset = np.stack(dataset)
        
        # Save dataset as npy file (with 1 channel)
        os.makedirs('./custom_dataset', exist_ok=True)
        np.save(save_path, dataset)
        print(f"Dataset saved at {save_path}")
        print(f"Dataset shape: {dataset.shape}")
        return dataset, save_path


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
            # Use -5 to 6 to include both -5 and 5 (numpy excludes upper bound)
            directions_with_speed = np.random.randint(-5, 6, size=(n_digits, 2))
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
                    color = np.expand_dims(colors[i], axis=(1, 2))
                        
                    input_digit = np.repeat(digits[i], 3, axis=0)
                    
                    # STEP 1: Use LANCZOS4 interpolation for best quality upscaling
                    input_digit = cv2.resize(input_digit.transpose(1, 2, 0), (x_dim, y_dim), 
                                            interpolation=cv2.INTER_LANCZOS4).transpose(2, 0, 1)
                    
                    # STEP 2: Apply Gaussian blur for anti-aliasing and smooth edges
                    # Kernel size proportional to scale for consistent smoothness
                    blur_kernel = max(3, int(scales[i] * 1.5) | 1)  # Ensure odd number
                    input_digit = cv2.GaussianBlur(input_digit.transpose(1, 2, 0), 
                                                (blur_kernel, blur_kernel), 0).transpose(2, 0, 1)
                    
                    # STEP 3: Optional dilation with elliptical kernel (smoother than square)
                    if num_dilate_iterations > 0:
                        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
                        input_digit = cv2.morphologyEx(input_digit.transpose(1, 2, 0), 
                                                    cv2.MORPH_DILATE, kernel, 
                                                    iterations=num_dilate_iterations).transpose(2, 0, 1)
                    
                    # STEP 4: Soft threshold with smooth sigmoid transition instead of hard cutoff
                    # This maintains the anti-aliased edges from interpolation
                    threshold = 0.1
                    transition_width = 0.08
                    input_digit = 1 / (1 + np.exp(-(input_digit - threshold) / transition_width * 10))

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
                    
                    # Use visible_digit as alpha mask - digits now layer on top instead of blending
                    frame_mask = (frame[:, frame_y_start:frame_y_end, frame_x_start:frame_x_end] > background_color + 0.01)
                    
                    # Alpha compositing: where digit is visible, show digit color; otherwise keep background
                    digit_alpha = visible_digit  # Smooth alpha from 0 to 1
                    current_frame_region = frame[:, frame_y_start:frame_y_end, frame_x_start:frame_x_end]
                    frame[:, frame_y_start:frame_y_end, frame_x_start:frame_x_end] = (
                        digit_alpha * color + (1 - digit_alpha) * current_frame_region
                    )
                    
                    digit_mask = visible_digit > 0.1  # Adjusted threshold for soft edges
                    
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
        for _ in tqdm(range(num_samples), desc="Generating videos"):
            video = self.generate_random_video(n_digits, min_scale, max_scale, num_dilate_iterations)
            dataset.append(video)
            
        dataset = np.stack(dataset)
        
        # save dataset as npy file
        os.makedirs('./custom_dataset', exist_ok=True)
        np.save(save_path, dataset)
        print("dataset.shape", dataset.shape)
        return dataset, save_path
    
class DisruptDatasetGenerator:
    def __init__(self, base_data_path, disruption_time, max_iou=0.3, target_h=128, target_w=160):
        self.base_videos = np.load(base_data_path)
        if len(self.base_videos.shape) == 4:
            if self.base_videos.shape[0] == 20:
                self.base_videos = np.transpose(self.base_videos, [1, 0, 2, 3])
            self.num_base_videos, self.nt, original_h, original_w = self.base_videos.shape
            self.base_videos = self.base_videos[:, :, np.newaxis, :, :]
        else:
            self.num_base_videos, self.nt, _, original_h, original_w = self.base_videos.shape
        
        # Set target dimensions (will resize base videos to these dimensions)
        self.target_h = target_h
        self.target_w = target_w
        self.h = target_h  # Use target dimensions for all operations
        self.w = target_w
        
        # Check if resizing is needed
        self.needs_resize = (original_h != target_h) or (original_w != target_w)
        
        self.disruption_time = disruption_time
        self.max_iou = max_iou
        
        mnist_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transforms.ToTensor())
        self.mnist_digits = [img.numpy().squeeze() for img, _ in mnist_dataset]
        
        print(f"Loaded {self.num_base_videos} base videos with shape: {self.base_videos.shape}")
        if self.needs_resize:
            print(f"Base videos will be resized from ({original_h}, {original_w}) to ({target_h}, {target_w})")

        
    def _get_random_digit(self):
        return self.mnist_digits[np.random.randint(0, len(self.mnist_digits))]
    
    def _get_base_video(self, idx):
        video = self.base_videos[idx % self.num_base_videos]  # (nt, 1, h, w) uint8
        video = video[:, 0, :, :].astype(np.float32) / 255.0  # (nt, h, w) float32 [0,1]
        
        # Resize to target dimensions if needed
        if self.needs_resize:
            nt = video.shape[0]
            resized_video = np.zeros((nt, self.target_h, self.target_w), dtype=np.float32)
            for t in range(nt):
                resized_video[t] = cv2.resize(video[t], (self.target_w, self.target_h), interpolation=cv2.INTER_LINEAR)
            video = resized_video
        
        return video  # (nt, target_h, target_w) float32 [0,1]
    
    def _get_digit_mask(self, digit, scale, pos, h=None, w=None):
        """Get a binary mask for the digit at given position."""
        x, y = int(pos[0]), int(pos[1])
        digit_size = int(28 * scale)
        resized = cv2.resize(digit, (digit_size, digit_size), interpolation=cv2.INTER_LINEAR)
        
        mask_h = h if h is not None else self.h
        mask_w = w if w is not None else self.w
        mask = np.zeros((mask_h, mask_w), dtype=np.float32)
        
        x_start_frame = max(0, x)
        y_start_frame = max(0, y)
        x_end_frame = min(mask_w, x + digit_size)
        y_end_frame = min(mask_h, y + digit_size)
        
        x_start_digit = max(0, -x)
        y_start_digit = max(0, -y)
        x_end_digit = x_start_digit + (x_end_frame - x_start_frame)
        y_end_digit = y_start_digit + (y_end_frame - y_start_frame)
        
        if x_end_frame <= x_start_frame or y_end_frame <= y_start_frame:
            return mask
        
        visible = resized[y_start_digit:y_end_digit, x_start_digit:x_end_digit]
        mask[y_start_frame:y_end_frame, x_start_frame:x_end_frame] = visible
        
        return mask > 0.1  # Binary mask with threshold
    
    def _compute_iou(self, mask1, mask2):
        """Compute Intersection over Union between two binary masks."""
        intersection = np.logical_and(mask1, mask2).sum()
        union = np.logical_or(mask1, mask2).sum()
        if union == 0:
            return 0.0
        return intersection / union
    
    def _check_trajectory_iou(self, base_video, digit, scale, velocity, init_pos, start_time, end_time):
        """Check if IOU stays below threshold for entire trajectory."""
        for t in range(start_time, end_time):
            current_pos = init_pos + velocity * (t - start_time)
            digit_mask = self._get_digit_mask(digit, scale, current_pos)
            base_mask = base_video[t] > 0.1  # Binary mask of base frame
            
            iou = self._compute_iou(digit_mask, base_mask)
            if iou > self.max_iou:
                return False, iou
        return True, 0.0
    
    def _init_new_digit_with_iou_check(self, base_video, min_scale, max_scale, start_time, end_time, max_attempts=50):
        """Initialize a new digit with position that maintains low IOU throughout trajectory."""
        for attempt in range(max_attempts):
            digit = self._get_random_digit()
            scale = np.random.uniform(min_scale, max_scale)
            velocity = np.random.randint(-4, 5, size=2)
            digit_size = int(28 * scale)
            
            # Random initial position
            pos = np.array([np.random.randint(0, max(1, self.w - digit_size)),
                           np.random.randint(0, max(1, self.h - digit_size))], dtype=float)
            
            # Check IOU for entire trajectory
            valid, max_iou_found = self._check_trajectory_iou(
                base_video, digit, scale, velocity, pos, start_time, end_time
            )
            
            if valid:
                return digit, scale, velocity, pos
        
        # If we can't find a good position, return the last attempt anyway
        # (with a warning if needed)
        return digit, scale, velocity, pos
    
    def _init_new_digit(self, min_scale, max_scale):
        """Legacy method without IOU check (for backwards compatibility)."""
        digit = self._get_random_digit()
        scale = np.random.uniform(min_scale, max_scale)
        velocity = np.random.randint(-4, 5, size=2)
        digit_size = int(28 * scale)
        pos = np.array([np.random.randint(0, max(1, self.w - digit_size)),
                       np.random.randint(0, max(1, self.h - digit_size))], dtype=float)
        return digit, scale, velocity, pos
    
    def _overlay_digit(self, frame, digit, scale, pos, h=None, w=None):
        x, y = int(pos[0]), int(pos[1])
        digit_size = int(28 * scale)
        resized = cv2.resize(digit, (digit_size, digit_size), interpolation=cv2.INTER_LINEAR)
        
        frame_h = h if h is not None else self.h
        frame_w = w if w is not None else self.w
        
        x_start_frame = max(0, x)
        y_start_frame = max(0, y)
        x_end_frame = min(frame_w, x + digit_size)
        y_end_frame = min(frame_h, y + digit_size)
        
        x_start_digit = max(0, -x)
        y_start_digit = max(0, -y)
        x_end_digit = x_start_digit + (x_end_frame - x_start_frame)
        y_end_digit = y_start_digit + (y_end_frame - y_start_frame)
        
        if x_end_frame <= x_start_frame or y_end_frame <= y_start_frame:
            return frame
        
        visible = resized[y_start_digit:y_end_digit, x_start_digit:x_end_digit]
        frame[y_start_frame:y_end_frame, x_start_frame:x_end_frame] = np.maximum(
            frame[y_start_frame:y_end_frame, x_start_frame:x_end_frame], visible)
        return frame
    
    def generate_sudden_appear(self, video_idx, min_scale=1.5, max_scale=2.5, nt=None, return_params=False):
        base = self._get_base_video(video_idx)
        if nt is None:
            nt = self.nt
        else:
            base = base[:nt]
        
        digit, scale, velocity, pos = self._init_new_digit_with_iou_check(
            base, min_scale, max_scale, 
            start_time=self.disruption_time, end_time=nt
        )
        
        video = np.zeros((nt, 1, self.h, self.w), dtype=np.float32)
        for t in range(nt):
            frame = base[t].copy()
            if t >= self.disruption_time:
                current_pos = pos + velocity * (t - self.disruption_time)
                frame = self._overlay_digit(frame, digit, scale, current_pos)
            video[t, 0] = frame
        
        if return_params:
            return video, (digit, scale, velocity, pos)
        return video
    
    def generate_transform(self, video_idx, min_scale=1.5, max_scale=2.5, nt=None, return_params=False):
        base = self._get_base_video(video_idx)
        if nt is None:
            nt = self.nt
        else:
            base = base[:nt]
        
        digit1, scale, velocity, pos = self._init_new_digit_with_iou_check(
            base, min_scale, max_scale,
            start_time=0, end_time=nt
        )
        digit2 = self._get_random_digit()
        
        video = np.zeros((nt, 1, self.h, self.w), dtype=np.float32)
        for t in range(nt):
            frame = base[t].copy()
            current_pos = pos + velocity * t
            digit = digit2 if t >= self.disruption_time else digit1
            frame = self._overlay_digit(frame, digit, scale, current_pos)
            video[t, 0] = frame
        
        if return_params:
            return video, (digit1, scale, velocity, pos)
        return video
    
    def generate_disappear(self, video_idx, min_scale=1.5, max_scale=2.5, nt=None, return_params=False):
        base = self._get_base_video(video_idx)
        if nt is None:
            nt = self.nt
        else:
            base = base[:nt]
        
        digit, scale, velocity, pos = self._init_new_digit_with_iou_check(
            base, min_scale, max_scale,
            start_time=0, end_time=self.disruption_time
        )
        
        video = np.zeros((nt, 1, self.h, self.w), dtype=np.float32)
        for t in range(nt):
            frame = base[t].copy()
            if t < self.disruption_time:
                current_pos = pos + velocity * t
                frame = self._overlay_digit(frame, digit, scale, current_pos)
            video[t, 0] = frame
        
        if return_params:
            return video, (digit, scale, velocity, pos)
        return video
    
    def generate_sudden_appear_val(self, video_idx, digit, scale, velocity, pos, min_scale=1.5, max_scale=2.5, nt=None):
        """Generate validation version: digit appears from t=0 (no disruption).
        
        Args:
            pos: Position at disruption_time (from disrupt version)
            For val version, we need to back-calculate initial position so digit
            is at pos at disruption_time, meaning initial_pos = pos - velocity * disruption_time
        """
        base = self._get_base_video(video_idx)
        if nt is None:
            nt = self.nt
        else:
            base = base[:nt]
        
        # Back-calculate initial position so digit is at pos at disruption_time
        initial_pos = pos - velocity * self.disruption_time
        
        video = np.zeros((nt, 1, self.h, self.w), dtype=np.float32)
        for t in range(nt):
            frame = base[t].copy()
            # Digit is present from t=0 (no disruption)
            current_pos = initial_pos + velocity * t
            frame = self._overlay_digit(frame, digit, scale, current_pos)
            video[t, 0] = frame
        return video
    
    def generate_transform_val(self, video_idx, digit1, scale, velocity, pos, min_scale=1.5, max_scale=2.5, nt=None):
        """Generate validation version: digit1 stays throughout (no transformation)."""
        base = self._get_base_video(video_idx)
        if nt is None:
            nt = self.nt
        else:
            base = base[:nt]
        
        video = np.zeros((nt, 1, self.h, self.w), dtype=np.float32)
        for t in range(nt):
            frame = base[t].copy()
            # Use digit1 throughout (no transformation)
            current_pos = pos + velocity * t
            frame = self._overlay_digit(frame, digit1, scale, current_pos)
            video[t, 0] = frame
        return video
    
    def generate_disappear_val(self, video_idx, digit, scale, velocity, pos, min_scale=1.5, max_scale=2.5, nt=None):
        """Generate validation version: digit stays throughout (no disappearance)."""
        base = self._get_base_video(video_idx)
        if nt is None:
            nt = self.nt
        else:
            base = base[:nt]
        
        video = np.zeros((nt, 1, self.h, self.w), dtype=np.float32)
        for t in range(nt):
            frame = base[t].copy()
            # Digit stays throughout (no disappearance)
            current_pos = pos + velocity * t
            frame = self._overlay_digit(frame, digit, scale, current_pos)
            video[t, 0] = frame
        return video
    
    def generate_dataset(self, num_samples, min_scale=1.5, max_scale=2.5, nt=None, 
                        min_digits=2, max_digits=5, h=None, w=None):
        """
        Generate disruption datasets and a single base validation set.
        All disruption types use the same base videos and additional digits.
        
        Args:
            num_samples: Number of videos to generate
            min_scale, max_scale: Scale range for disruption digits
            nt: Number of timesteps
            min_digits, max_digits: Total number of digits (base has 2, so adds 0-3 more)
            h, w: Target height and width (defaults to self.h, self.w)
        """
        if nt is None:
            nt = self.nt
        target_h = h if h is not None else self.h
        target_w = w if w is not None else self.w
        
        os.makedirs('./data/adapt', exist_ok=True)
        
        # Step 1: Generate base validation videos with additional digits (no disruptions)
        # This will be shared by all disruption types
        print("Step 1: Generating base validation videos with additional digits...")
        base_val_videos = []
        disruption_params = []  # Store disruption parameters for each video
        
        for i in tqdm(range(num_samples), desc="Generating base videos"):
            base_video = self._get_base_video(i)
            if nt is not None and base_video.shape[0] > nt:
                base_video = base_video[:nt]
            
            # Resize base if needed
            if target_h != self.h or target_w != self.w:
                resized_base = np.zeros((nt, target_h, target_w), dtype=np.float32)
                for t in range(nt):
                    resized_base[t] = cv2.resize(base_video[t], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                base_video = resized_base
            
            # Calculate how many digits we can add beyond the base 2 digits
            # Base has 2 digits, max_digits is total, so we can add (max_digits - 2) digits total
            # We need: transform digit1 (1) + disappear digit (1) = 2 disruption digits minimum
            # So if max_digits - 2 < 2, we have a problem. Let's use the same digit for both.
            # Actually, let's make transform and disappear use the SAME digit to save on digit count
            # This way: base (2) + shared disruption digit (1) = 3 digits total (when max_digits=3)
            
            # Generate disruption parameters for each type FIRST (before creating base_val)
            # For appear: digit appears at disruption_time (NOT in base_val)
            digit_appear, scale_appear, velocity_appear, pos_appear = self._init_new_digit_with_iou_check(
                base_video, min_scale, max_scale,
                start_time=self.disruption_time, end_time=nt
            )
            
            # For transform and disappear: use the SAME digit to save on total digit count
            # This digit will be in base_val throughout (full trajectory)
            # For transform: it transforms to digit2 at disruption_time
            # For disappear: it disappears at disruption_time
            shared_disrupt_digit, shared_scale, shared_velocity, shared_pos = self._init_new_digit_with_iou_check(
                base_video, min_scale, max_scale,
                start_time=0, end_time=nt
            )
            digit2_transform = self._get_random_digit()  # For transform disruption
            
            # Calculate additional digits (beyond the shared disruption digit)
            # Total = base (2) + shared disruption digit (1) + additional (n_additional)
            # To get max_digits: 3 + n_additional = max_digits
            #                   n_additional = max_digits - 3
            n_additional = max(0, max_digits - 3)  # Additional digits beyond base + shared disruption digit
            
            # Generate additional digits (if any)
            additional_digits = []
            # Create a combined base with shared disruption digit for IOU checking
            temp_base_with_disruption = base_video.copy()
            for t in range(nt):
                current_pos_shared = shared_pos + shared_velocity * t
                temp_base_with_disruption[t] = self._overlay_digit(
                    temp_base_with_disruption[t].copy(), 
                    shared_disrupt_digit, shared_scale, current_pos_shared, 
                    h=target_h, w=target_w
                )
            
            for _ in range(n_additional):
                digit_add, scale_add, velocity_add, pos_add = self._init_new_digit_with_multi_iou_check(
                    temp_base_with_disruption, additional_digits, min_scale, max_scale,
                    start_time=0, end_time=nt, h=target_h, w=target_w
                )
                additional_digits.append((digit_add, scale_add, velocity_add, pos_add))
            
            # Create base validation video with additional digits + transform/disappear digits (full trajectories)
            val_video = np.zeros((nt, 1, target_h, target_w), dtype=np.float32)
            for t in range(nt):
                frame = base_video[t].copy()
                # Add all additional digits
                for digit_add, scale_add, velocity_add, pos_add in additional_digits:
                    current_pos = pos_add + velocity_add * t
                    frame = self._overlay_digit(frame, digit_add, scale_add, current_pos, h=target_h, w=target_w)
                
                # Add shared disruption digit (full trajectory, no disruption)
                # This digit is used for both transform and disappear disruptions
                current_pos_shared = shared_pos + shared_velocity * t
                frame = self._overlay_digit(frame, shared_disrupt_digit, shared_scale, current_pos_shared, h=target_h, w=target_w)
                
                val_video[t, 0] = frame
            
            base_val_videos.append(val_video)
            
            disruption_params.append({
                'appear': (digit_appear, scale_appear, velocity_appear, pos_appear),
                'transform': (shared_disrupt_digit, digit2_transform, shared_scale, shared_velocity, shared_pos),
                'disappear': (shared_disrupt_digit, shared_scale, shared_velocity, shared_pos),
                'additional_digits': additional_digits
            })
        
        # Save base validation dataset
        base_val_dataset = np.stack(base_val_videos)
        base_val_dataset = np.clip(base_val_dataset * 255.0, 0, 255).astype(np.uint8)
        base_val_path = './data/adapt/mnist_val_disrupt_base.npy'
        np.save(base_val_path, base_val_dataset)
        print(f"Saved base validation: {base_val_dataset.shape} to {base_val_path}")
        
        # Step 2: Generate disruption datasets using the same base videos and additional digits
        print("\nStep 2: Generating disruption datasets...")
        
        # Generate sudden appear datasets
        appear_videos = []
        for i in tqdm(range(num_samples), desc="Generating sudden appear"):
            # Start from base validation video (has base + additional digits)
            video = base_val_videos[i].copy()
            digit, scale, velocity, pos = disruption_params[i]['appear']
            additional_digits = disruption_params[i]['additional_digits']
            
            # Add disruption: digit appears at disruption_time
            for t in range(nt):
                if t >= self.disruption_time:
                    current_pos = pos + velocity * (t - self.disruption_time)
                    video[t, 0] = self._overlay_digit(video[t, 0].copy(), digit, scale, current_pos, h=target_h, w=target_w)
            
            appear_videos.append(video)
        
        appear_dataset = np.stack(appear_videos)
        appear_dataset = np.clip(appear_dataset * 255.0, 0, 255).astype(np.uint8)
        appear_path = f'./data/adapt/sudden_appear_{num_samples}_{nt}_t{self.disruption_time}.npy'
        np.save(appear_path, appear_dataset)
        print(f"Saved: {appear_dataset.shape} to {appear_path}")
        
        # Generate transform datasets
        transform_videos = []
        for i in tqdm(range(num_samples), desc="Generating transform"):
            # Start from base validation video (already has shared disruption digit throughout)
            video = base_val_videos[i].copy()
            shared_digit, digit2, scale, velocity, pos = disruption_params[i]['transform']
            additional_digits = disruption_params[i]['additional_digits']
            
            # Apply disruption: replace shared digit with digit2 at disruption_time
            # We need to rebuild frames from disruption_time onwards to properly replace the digit
            # Get the base video (without any added digits)
            base_video = self._get_base_video(i)
            if nt is not None and base_video.shape[0] > nt:
                base_video = base_video[:nt]
            
            if target_h != self.h or target_w != self.w:
                resized_base = np.zeros((nt, target_h, target_w), dtype=np.float32)
                for t in range(nt):
                    resized_base[t] = cv2.resize(base_video[t], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                base_video = resized_base
            
            # Rebuild frames from disruption_time onwards with digit2 instead of shared_digit
            for t in range(self.disruption_time, nt):
                frame = base_video[t].copy()
                # Add all additional digits
                for digit_add, scale_add, velocity_add, pos_add in additional_digits:
                    current_pos = pos_add + velocity_add * t
                    frame = self._overlay_digit(frame, digit_add, scale_add, current_pos, h=target_h, w=target_w)
                # Add digit2 instead of shared_digit (transformation happens here)
                current_pos = pos + velocity * t
                frame = self._overlay_digit(frame, digit2, scale, current_pos, h=target_h, w=target_w)
                video[t, 0] = frame
            
            transform_videos.append(video)
        
        transform_dataset = np.stack(transform_videos)
        transform_dataset = np.clip(transform_dataset * 255.0, 0, 255).astype(np.uint8)
        transform_path = f'./data/adapt/sudden_transform_{num_samples}_{nt}_t{self.disruption_time}.npy'
        np.save(transform_path, transform_dataset)
        print(f"Saved: {transform_dataset.shape} to {transform_path}")
        
        # Generate disappear datasets
        disappear_videos = []
        for i in tqdm(range(num_samples), desc="Generating disappear"):
            # Start from base validation video (already has shared disruption digit throughout)
            video = base_val_videos[i].copy()
            shared_digit, scale, velocity, pos = disruption_params[i]['disappear']
            
            # Apply disruption: remove shared digit at disruption_time
            # We need to recreate frames from t=disruption_time onwards without the shared digit
            # Get the base video (without any added digits)
            base_video = self._get_base_video(i)
            if nt is not None and base_video.shape[0] > nt:
                base_video = base_video[:nt]
            
            if target_h != self.h or target_w != self.w:
                resized_base = np.zeros((nt, target_h, target_w), dtype=np.float32)
                for t in range(nt):
                    resized_base[t] = cv2.resize(base_video[t], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
                base_video = resized_base
            
            additional_digits = disruption_params[i]['additional_digits']
            
            # Rebuild frames from disruption_time onwards without the shared disruption digit
            for t in range(self.disruption_time, nt):
                frame = base_video[t].copy()
                # Add all additional digits
                for digit_add, scale_add, velocity_add, pos_add in additional_digits:
                    current_pos = pos_add + velocity_add * t
                    frame = self._overlay_digit(frame, digit_add, scale_add, current_pos, h=target_h, w=target_w)
                # Do NOT add shared disruption digit (it disappears at disruption_time)
                video[t, 0] = frame
            
            disappear_videos.append(video)
        
        disappear_dataset = np.stack(disappear_videos)
        disappear_dataset = np.clip(disappear_dataset * 255.0, 0, 255).astype(np.uint8)
        disappear_path = f'./data/adapt/sudden_disappear_{num_samples}_{nt}_t{self.disruption_time}.npy'
        np.save(disappear_path, disappear_dataset)
        print(f"Saved: {disappear_dataset.shape} to {disappear_path}")
        
        return {
            'sudden_appear': (appear_dataset, appear_path),
            'transform': (transform_dataset, transform_path),
            'disappear': (disappear_dataset, disappear_path),
            'base_val': (base_val_dataset, base_val_path)
        }
    
    def _check_trajectory_iou_multi(self, base_video, existing_digits, digit, scale, velocity, init_pos, start_time, end_time, h=None, w=None):
        mask_h = h if h is not None else self.h
        mask_w = w if w is not None else self.w
        
        for t in range(start_time, end_time):
            current_pos = init_pos + velocity * (t - start_time)
            digit_mask = self._get_digit_mask(digit, scale, current_pos, h=mask_h, w=mask_w)
            base_mask = base_video[t] > 0.1
            
            combined_mask = base_mask.copy()
            for existing_digit, existing_scale, existing_velocity, existing_pos in existing_digits:
                existing_current_pos = existing_pos + existing_velocity * (t - start_time)
                existing_mask = self._get_digit_mask(existing_digit, existing_scale, existing_current_pos, h=mask_h, w=mask_w)
                combined_mask = np.logical_or(combined_mask, existing_mask)
            
            iou = self._compute_iou(digit_mask, combined_mask)
            if iou > self.max_iou:
                return False, iou
        return True, 0.0
    
    def _init_new_digit_with_multi_iou_check(self, base_video, existing_digits, min_scale, max_scale, start_time, end_time, h=None, w=None, max_attempts=50):
        mask_h = h if h is not None else self.h
        mask_w = w if w is not None else self.w
        
        for attempt in range(max_attempts):
            digit = self._get_random_digit()
            scale = np.random.uniform(min_scale, max_scale)
            velocity = np.random.randint(-4, 5, size=2)
            digit_size = int(28 * scale)
            
            pos = np.array([np.random.randint(0, max(1, mask_w - digit_size)),
                           np.random.randint(0, max(1, mask_h - digit_size))], dtype=float)
            
            valid, max_iou_found = self._check_trajectory_iou_multi(
                base_video, existing_digits, digit, scale, velocity, pos, start_time, end_time, h=mask_h, w=mask_w
            )
            
            if valid:
                return digit, scale, velocity, pos
        
        return digit, scale, velocity, pos
    
    def generate_multi_digit_video(self, video_idx, min_digits=2, max_digits=5, min_scale=1.5, max_scale=2.5, nt=None, h=None, w=None):
        base = self._get_base_video(video_idx)
        if nt is None:
            nt = self.nt
        else:
            base = base[:nt]
        
        target_h = h if h is not None else self.h
        target_w = w if w is not None else self.w
        
        if target_h != self.h or target_w != self.w:
            resized_base = np.zeros((nt, target_h, target_w), dtype=np.float32)
            for t in range(nt):
                resized_base[t] = cv2.resize(base[t], (target_w, target_h), interpolation=cv2.INTER_LINEAR)
            base = resized_base
        
        n_additional = np.random.randint(max(0, min_digits - 2), max_digits - 2 + 1)
        
        additional_digits = []
        for _ in range(n_additional):
            digit, scale, velocity, pos = self._init_new_digit_with_multi_iou_check(
                base, additional_digits, min_scale, max_scale, start_time=0, end_time=nt, h=target_h, w=target_w
            )
            additional_digits.append((digit, scale, velocity, pos))
        
        video = np.zeros((nt, 1, target_h, target_w), dtype=np.float32)
        for t in range(nt):
            frame = base[t].copy()
            
            for digit, scale, velocity, pos in additional_digits:
                current_pos = pos + velocity * t
                frame = self._overlay_digit(frame, digit, scale, current_pos, h=target_h, w=target_w)
            
            video[t, 0] = frame
        return video
    
    def generate_multi_digit_dataset(self, num_samples, min_digits=2, max_digits=5, min_scale=1.5, max_scale=2.5, nt=None, h=None, w=None):
        if nt is None:
            nt = self.nt
        target_h = h if h is not None else self.h
        target_w = w if w is not None else self.w
        
        os.makedirs('./data', exist_ok=True)
        
        videos = []
        for i in tqdm(range(num_samples), desc="Generating multi-digit videos"):
            video = self.generate_multi_digit_video(i, min_digits, max_digits, min_scale, max_scale, nt=nt, h=target_h, w=target_w)
            videos.append(video)
        
        dataset = np.stack(videos)
        dataset = np.clip(dataset * 255.0, 0, 255).astype(np.uint8)
        
        save_path = f'./data/multi_digit_{num_samples}_{nt}_d{min_digits}-{max_digits}.npy'
        np.save(save_path, dataset)
        print(f"Saved: {dataset.shape} to {save_path}")
        
        return dataset, save_path


if __name__ == '__main__':
    generator = GrayscaleMovingMnistGenerator(nt=10, h=128, w=160)
    dataset = generator.generate_dataset(num_samples=10000, n_digits=3, max_scale=2.5, min_scale=2.0)
    dataset_path = f'./custom_dataset/mnist_grayscale_{10000}_{10}.npy'
    train_path, val_path = split_custom_mnist_data(datapath=dataset_path, nt=10)
    print(train_path, val_path)