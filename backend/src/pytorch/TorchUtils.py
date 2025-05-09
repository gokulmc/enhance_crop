import torch
import torch.nn.functional as F
import sys
from ..utils.BackendChecks import (
    check_bfloat16_support,
    get_gpus_torch,
)
from ..constants import HAS_PYTORCH_CUDA
from ..utils.Util import (
    printAndLog,
    errorAndLog,
    warnAndLog,
    log,
)

class TorchUtils:
    # device and precision are in string formats, loaded straight from the command line arguments
    def __init__(self, width, height, hdr_mode=False, padding=None, ):
        self.width = width
        self.height = height
        self.hdr_mode = hdr_mode
        self.padding = padding

    @staticmethod
    def init_stream():
        if HAS_PYTORCH_CUDA:
            return torch.cuda.Stream()
        else:
            return torch.cpu.Stream()
        
    @staticmethod
    def run_stream(stream):
        if HAS_PYTORCH_CUDA:
            return torch.cuda.stream(stream)
        else:
            return torch.cpu.stream(stream)
        
    @staticmethod
    def sync_all_streams():
        if HAS_PYTORCH_CUDA:
            torch.cuda.synchronize()
        else:
            torch.cpu.synchronize()
    
    @staticmethod
    def handle_device(device, gpu_id: int = 0) -> torch.device:
        """
        returns device based on gpu id and device parameter
    """
        if device == "auto":
            torchdevice = torch.device("cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu")
        
        elif device == "cuda":
            torchdevice = torch.device(
                device, gpu_id
            )  # 0 is the device index, may have to change later
        else:
            torchdevice = torch.device(device)
    
        device = get_gpus_torch()[gpu_id]
        print("Using Device: " + str(device), file=sys.stderr)
        return torchdevice
    
    @staticmethod
    def handle_precision(precision) -> torch.dtype:
        if precision == "auto":
            return torch.float16 if check_bfloat16_support() else torch.float32
        if precision == "float32":
            return torch.float32
        if precision == "float16":
            return torch.float16
        if precision == "bfloat16":
            return torch.bfloat16
        return torch.float32

    @torch.inference_mode()
    def copy_tensor(self, tensorToCopy: torch.Tensor, tensorCopiedTo: torch.Tensor, stream: torch.Stream):
        with self.run_stream(stream):  # type: ignore
            tensorToCopy.copy_(tensorCopiedTo, non_blocking=True)
        stream.synchronize()
    

    @torch.inference_mode()
    def frame_to_tensor(self, frame, stream: torch.Stream, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        with self.run_stream(stream):  # type: ignore

            frame = torch.frombuffer(
                    frame,
                    dtype=torch.uint16 if self.hdr_mode else torch.uint8,
                ).to(device=device, non_blocking=True, dtype=torch.float32 if self.hdr_mode else dtype) # torch dies in hdr mode if we dont cast to float before half
            
            frame = (
                frame
                .reshape(self.height, self.width, 3)
                .permute(2, 0, 1)
                .unsqueeze(0)
                .div(65535.0 if self.hdr_mode else 255.0)
                .clamp_(0.0, 1.0)
                ).to(dtype=dtype, non_blocking=True)

            if self.padding:
                frame = F.pad(frame, self.padding)
                
            stream.synchronize()
        return frame
    
    @staticmethod
    def clear_cache():
        if HAS_PYTORCH_CUDA:
            torch.cuda.empty_cache()
    
    @torch.inference_mode()
    def tensor_to_frame(self, frame: torch.Tensor):
        # Choose conversion parameters based on hdr_mode flag

        return (
            frame.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(65535.0 if self.hdr_mode else 255.0)
            .to(torch.uint16 if self.hdr_mode else torch.uint8)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
    )