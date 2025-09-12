from dataclasses import dataclass
import torch
@dataclass
class BaseVSRArch:
    """
    base vsr class for all vsr archs
    """
    num_cached_frames:int
    scale:int
    model: torch.nn.Module
        
        
    
