from .base_vsr_arch import BaseVSRArch
import torch
import gc
class VSRInferenceHelper:
    def __init__(self, model: BaseVSRArch):
        self.scale = model.scale
        self.model = model.model
        self.num_cached_frames = model.num_cached_frames
        self.frame_cache = []
        self._first_inference = True

    def __call__(self, frame: torch.Tensor):
        
        
        # fill the queue to render with multiple frames for the model
        if self._first_inference:
            height, width = frame.shape[2:]
            self.state = frame.new_zeros(1, 64, height, width)
            self.out = frame.new_zeros(1, 3, height * self.scale, width * self.scale)
            for i in range(self.num_cached_frames):
                self.frame_cache.append(frame)
            self._first_inference = False
        x = torch.cat(self.frame_cache, dim=1)
        #print(x.shape, file=sys.stderr)
        #sys.exit()
        self.out, self.state = self.model(x, self.out, self.state)
        # remove frame from cache
        self.frame_cache.pop(0)
        
        self.frame_cache.append(frame)
        gc.collect()
        return self.out

            