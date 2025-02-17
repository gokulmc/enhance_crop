
import torch
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from queue import Queue

from ..utils.SSIM import SSIM

# from backend.src.pytorch.InterpolateArchs.GIMM import GIMM
from .InterpolateArchs.DetectInterpolateArch import ArchDetect
from .UpscaleTorch import UpscalePytorch
import math
import os
import logging
import gc
import sys
from ..utils.Util import (
    printAndLog,
    errorAndLog,
    check_bfloat16_support,
    warnAndLog,
    log,
    get_gpus_torch,
)
from ..constants import HAS_SYSTEM_CUDA
from time import sleep

torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)

class DynamicScale:
    def __init__(self, possible_values: dict, CompareNet: SSIM):
        self.possible_values = possible_values
        self.CompareNet = CompareNet

    @torch.inference_mode()
    def dynamicScaleCalculation(self, frame0, frame1):
        ssim: torch.Tensor = self.CompareNet(frame0, frame1)
        closest_value = min(self.possible_values, key=lambda v: abs(ssim.item() - v))
        scale = self.possible_values[closest_value]
        return scale

    # limit gmfss scale to 1.0 max


class BaseInterpolate(metaclass=ABCMeta):
    @abstractmethod
    def _load(self):
        """Loads in the model"""
        self.device = torch.device("cuda")
        self.dtype = torch.float32
        self.width = 1920
        self.height = 1080
        self.padding = [0, 0, 0, 0]
        self.frame0 = None
        self.encode0 = None
        self.flownet = None
        self.encode = None
        self.tenFlow_div = None
        self.backwarp_tenGrid = None
        self.doEncodingOnFrame = False  # set this by default
        self.hdr_mode = False
        self.CompareNet = None

    @staticmethod
    def handleDevice(device: str, gpu_id: int = 0) -> torch.device:
        if device == "default":
            if torch.cuda.is_available():
                torchdevice = torch.device(
                    "cuda", gpu_id
                )  # 0 is the device index, may have to change later
            else:
                torchdevice = torch.device("cpu")
        else:
            torchdevice = torch.device(device)
        device = get_gpus_torch()[gpu_id]
        print("Using GPU: " + str(device), file=sys.stderr)
        return torchdevice

    def handlePrecision(self, precision) -> torch.dtype:
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
    def copyTensor(self, tensorToCopy: torch.Tensor, tensorCopiedTo: torch.Tensor, stream):
        with torch.cuda.stream(stream):  # type: ignore
            tensorToCopy.copy_(tensorCopiedTo, non_blocking=True)
        self.stream.synchronize()

    def hotUnload(self):
        self.flownet = None
        self.encode = None
        self.tenFlow_div = None
        self.backwarp_tenGrid = None
        self.f0encode = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.reset_max_memory_allocated()
        torch.cuda.reset_max_memory_cached()

    @torch.inference_mode()
    def hotReload(self):
        self._load()

    @abstractmethod
    @torch.inference_mode()
    def __call__(
        self,
        img1,
        writeQueue: Queue,
        transition=False,
        upscaleModel: UpscalePytorch = None,
    ):  # type: ignore
        """Perform processing"""

    def initLog(self):
        printAndLog("Using dtype: " + str(self.dtype))

    @torch.inference_mode()
    def norm(self, frame: torch.Tensor):
        return (
            frame
            .reshape(self.height, self.width, 3)
            .permute(2, 0, 1)
            .unsqueeze(0)
            .div_(65535.0 if self.hdr_mode else 255.0)
        )

    @torch.inference_mode()
    def frame_to_tensor(self, frame, stream: torch.cuda.Stream) -> torch.Tensor:
        with torch.cuda.stream(stream):  # type: ignore
            frame = self.norm(
                torch.frombuffer(
                    frame,
                    dtype=torch.uint16 if self.hdr_mode else torch.uint8,
                ).to(device=self.device, non_blocking=True, dtype=torch.float32 if self.hdr_mode else self.dtype) # torch dies in hdr mode if we dont cast to float before half
            ).to(dtype=self.dtype, non_blocking=True)
            frame = F.pad(frame, self.padding)
        stream.synchronize()
        return frame

    @torch.inference_mode()
    def uncacheFrame(self):
        self.f0encode = None
        self.img0 = None

    @torch.inference_mode()
    def tensor_to_frame(self, frame: torch.Tensor):
        # Choose conversion parameters based on hdr_mode flag

        return (
            frame.squeeze(0)
            .permute(1, 2, 0)
            .clamp(0, 1)
            .mul(65535.0 if self.hdr_mode else 255.0)
            .round()
            .to(torch.uint16 if self.hdr_mode else torch.uint8)
            .contiguous()
            .detach()
            .cpu()
            .numpy()
    )