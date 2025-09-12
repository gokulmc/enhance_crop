from .base_vsr_arch import BaseVSRArch
import torch
from queue import Queue
class VSRInferenceHelper:
    def __init__(self, model: BaseVSRArch):
        self.model = model
        self.num_cached_frames = model.num_cached_frames
        self.frame_cache = []
        self._first_inference = True

    def __call__(self, frame):
        # fill the queue to render with multiple frames for the model
        if self._first_inference:
            for i in self.num_cached_frames:
                self.frame_cache.append(frame)
                self._first_inference = False
        
        output = self.model(torch.stack(self.frame_cache))
        # remove frame from cache
        self.frame_cache.pop()
        self.frame_cache.append(frame)
        return output

            