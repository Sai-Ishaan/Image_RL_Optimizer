##Custom OpenEnv observation transform functions
from collections import deque
import numpy as np
import torch

class AtariTransformPipeline:
    ##Handles preprocessing and frame-stacking for raw Atari frames: Converts raw env o/ps to optimized tensors for NatureCNN
    def __init__(self, stack_size:int =4, target_dim:tuple = (84,84)):
        self.stack_size = stack_size
        self.target_dim = target_dim
        self.frame_buffer = deque(maxlen=stack_size)

    def reset(self, initial_frame: np.ndarray)-> torch.Tensor:
        ##Resets the frame buffer and fills with by duplicating the initial frame.
        processed = self._preprocess_frame(initial_frame)
        for _ in range(self.stack_size):
            self.frame_buffer.append(processed)
        return self._get_stack_tensor()
    def process_step(self, frame: np.ndarray)-> torch.Tensor:
        ##Processes a single new frame and updates temporal stack
        processed = self._preprocess_frame(frame)
        self.frame_buffer.append(processed)
        return self._get_stack_tensor()
    def _preprocess_frame(self, frame: np.ndarray)->np.ndarray:
        ###Slices out the unwanted rendering space and downsamples to target dimensions: either flattened array wrapper or (210,160,3) RGB
        if frame.ndim == 1: #Ensures array structure
            if frame.size == 100800:
                frame = frame.reshape(210, 160, 3)
            else:
                raise ValueError(f"Unexpected flat frame size: {frame.size}")
        if frame.ndim < 2 or frame.shape[0] <194 or frame.shape[1]<84:
            raise ValueError(f"Malformed input dimensions:{frame.shape}. Expected at least (194,84)- Atari Spatial Structure")

        if frame.ndim == 3 and frame.shape[-1] == 3:
            ## Std lumninance weights for grayscale conversion
            frame = dot_grayscale(frame)
        cropped = frame[34:194, :]
        ###Downsample to 84x84 using uniform step slicing (fastest CPU approach)
        split_h = cropped.shape[0] // self.target_dim[0]
        split_w = cropped.shape[1] // self.target_dim[1]
        downsampled = cropped[::split_h, ::split_w][:self.target_dim[0], :self.target_dim[1]]

        return downsampled.astype(np.float32) / 255.0
    def _get_stack_tensor(self)-> torch.Tensor:
        ## Converts the buffer collection into a single PyTorch batch tensor
        stacked = np.stack(self.frame_buffer, axis=0) 
        tensor = torch.from_numpy(stacked).unsqueeze(0)
        return tensor

def dot_grayscale(rgb_frame: np.ndarray)-> np.ndarray:
    ##Fast matrix dot product for grayscale conversion
    return np.dot(rgb_frame[..., :3],[0.2989, 0.5870, 0.1140]).astype(np.uint8)    