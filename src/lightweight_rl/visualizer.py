## To visualize and compare the performance of the teacher and student models
import cv2
import numpy as np

class Visualizer: 
    ##Inspired by OpenAI Gym's rendering utilities, 
    ## as well as DeepMind visualization tools for RL agents. 
    ## Renders side-by-side visual comparison of Teacher vs Student agents
    ## WITH Real Time HUD overlays, Q-Values and latency metrics.
    def __init__(self, frame_size=(320, 320)):
        self.frame_size = frame_size

    def _prepare_frame(self, raw_frame: np.ndarray) -> np.ndarray:
        ### Converting observation matrices into std RGB renderable frames
        if raw_frame.ndim == 1:
            if raw_frame.size == 100800:
                raw_frame = raw_frame.reshape(210,160,3)
            else:
                dim = int(np.sqrt(raw_frame.size)) ##Handling flattened/single channel inputs
                raw_frame = raw_frame.reshape(dim,dim)
