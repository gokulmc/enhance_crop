from . import animesr_arch
from ..base_vsr_arch import BaseVSRArch
class AnimeSRArch(BaseVSRArch):
    """
    this class will get attributes from the model, like scale and dims.
    """
    def __init__(self):
        self.model = animesr_arch.AnimeSR()
        self.num_cached_frames = 3
        self.scale = 2
        
    
