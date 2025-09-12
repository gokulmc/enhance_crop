from .base_vsr_arch import BaseVSRArch
import torch
import sys
class VSRInferenceHelper:
    def __init__(self, model: BaseVSRArch):
        self.model = model.model
        self.num_cached_frames = model.num_cached_frames
        self.frame_cache = []
        self._first_inference = True

    def __call__(self, frame: torch.Tensor):
        # fill the queue to render with multiple frames for the model
        if self._first_inference:
            for i in range(self.num_cached_frames):
                self.frame_cache.append(frame)
                self._first_inference = False
        x = torch.cat(self.frame_cache, dim=0)
        #print(x.shape, file=sys.stderr)
        #sys.exit()
        output = self.model(x)
        # remove frame from cache
        """self.frame_cache.pop(0)
        self.frame_cache.append(frame)"""
        return output

            