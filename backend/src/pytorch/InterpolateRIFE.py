import torch
import torch.nn.functional as F
from abc import ABCMeta, abstractmethod
from queue import Queue

from ..utils.SSIM import SSIM

# from backend.src.pytorch.InterpolateArchs.GIMM import GIMM
from .BaseInterpolate import BaseInterpolate, DynamicScale
from .InterpolateArchs.DetectInterpolateArch import ArchDetect
from .UpscaleTorch import UpscalePytorch
import math
import os
import logging
import sys
from ..utils.Util import (
    errorAndLog,
)
from ..constants import HAS_SYSTEM_CUDA
from time import sleep

torch.set_float32_matmul_precision("medium")
torch.set_grad_enabled(False)
logging.basicConfig(level=logging.INFO)

class InterpolateRifeTorch(BaseInterpolate):
    @torch.inference_mode()
    def __init__(
        self,
        modelPath: str,
        ceilInterpolateFactor: int = 2,
        width: int = 1920,
        height: int = 1080,
        device: str = "default",
        dtype: str = "auto",
        backend: str = "pytorch",
        UHDMode: bool = False,
        ensemble: bool = False,
        dynamicScaledOpticalFlow: bool = False,
        gpu_id: int = 0,
        # trt options
        trt_optimization_level: int = 5,
        *args,
        **kwargs,
    ):
        self.interpolateModel = modelPath
        self.width = width
        self.height = height

        self.device: torch.device = self.handleDevice(device, gpu_id=gpu_id)
        self.dtype = self.handlePrecision(dtype)
        self.backend = backend
        self.ceilInterpolateFactor = ceilInterpolateFactor
        self.dynamicScaledOpticalFlow = dynamicScaledOpticalFlow
        self.CompareNet = None
        self.frame0 = None
        self.encode0 = None
        # set up streams for async processing
        self.scale = 1
        self.doEncodingOnFrame = True
        self.ensemble = ensemble

        self.trt_optimization_level = trt_optimization_level
        self.trt_cache_dir = os.path.dirname(
            modelPath
        )  # use the model directory as the cache directory
        self.UHDMode = UHDMode
        if self.UHDMode:
            self.scale = 0.5
        self._load()

    @torch.inference_mode()
    def _load(self):
        self.stream = torch.cuda.Stream()
        self.prepareStream = torch.cuda.Stream()
        self.copyStream = torch.cuda.Stream()
        self.f2tStream = torch.cuda.Stream()
        with torch.cuda.stream(self.prepareStream):  # type: ignore
            state_dict = torch.load(
                self.interpolateModel,
                map_location=self.device,
                weights_only=True,
                mmap=True,
            )
            # detect what rife arch to use

            ad = ArchDetect(self.interpolateModel)
            interpolateArch = ad.getArchName()
            _pad = 32
            num_ch_for_encode = 0
            match interpolateArch.lower():
                case "rife46":
                    from .InterpolateArchs.RIFE.rife46IFNET import IFNet

                    self.doEncodingOnFrame = False
                case "rife47":
                    from .InterpolateArchs.RIFE.rife47IFNET import IFNet

                    num_ch_for_encode = 4
                    self.encode = torch.nn.Sequential(
                        torch.nn.Conv2d(3, 16, 3, 2, 1),
                        torch.nn.ConvTranspose2d(16, 4, 4, 2, 1),
                    )
                case "rife413":
                    from .InterpolateArchs.RIFE.rife413IFNET import IFNet, Head

                    num_ch_for_encode = 8
                    self.encode = Head()
                case "rife420":
                    from .InterpolateArchs.RIFE.rife420IFNET import IFNet, Head

                    num_ch_for_encode = 8
                    self.encode = Head()
                case "rife421":
                    from .InterpolateArchs.RIFE.rife421IFNET import IFNet, Head

                    num_ch_for_encode = 8
                    self.encode = Head()
                case "rife422lite":
                    from .InterpolateArchs.RIFE.rife422_liteIFNET import IFNet, Head

                    self.encode = Head()
                    num_ch_for_encode = 4
                case "rife425":
                    from .InterpolateArchs.RIFE.rife425IFNET import IFNet, Head

                    _pad = 64
                    num_ch_for_encode = 4
                    self.encode = Head()

                case _:
                    errorAndLog("Invalid Interpolation Arch")
                    exit()

            # model unspecific setup
            if self.dynamicScaledOpticalFlow:
                tmp = max(
                    _pad, int(_pad / 0.25)
                )  # set pad to higher for better dynamic optical scale support
            else:
                tmp = max(_pad, int(_pad / self.scale))
            self.pw = math.ceil(self.width / tmp) * tmp
            self.ph = math.ceil(self.height / tmp) * tmp
            self.padding = (0, self.pw - self.width, 0, self.ph - self.height)
            # caching the timestep tensor in a dict with the timestep as a float for the key

            self.timestepDict = {}
            for n in range(self.ceilInterpolateFactor):
                timestep = n / (self.ceilInterpolateFactor)
                timestep_tens = torch.full(
                    (1, 1, self.ph, self.pw),
                    timestep,
                    dtype=self.dtype,
                    device=self.device,
                )
                self.timestepDict[timestep] = timestep_tens
            # rife specific setup
            self.set_rife_args()  # sets backwarp_tenGrid and tenFlow_div
            self.flownet = IFNet(
                scale=self.scale,
                ensemble=self.ensemble,
                dtype=self.dtype,
                device=self.device,
                width=self.width,
                height=self.height,
            )

            state_dict = {
                k.replace("module.", ""): v
                for k, v in state_dict.items()
                if "module." in k
            }
            head_state_dict = {
                k.replace("encode.", ""): v
                for k, v in state_dict.items()
                if "encode." in k
            }
            if self.doEncodingOnFrame:
                self.encode.load_state_dict(state_dict=head_state_dict, strict=True)
                self.encode.eval().to(device=self.device, dtype=self.dtype)
            self.flownet.load_state_dict(state_dict=state_dict, strict=False)
            self.flownet.eval().to(device=self.device, dtype=self.dtype)

            if self.dynamicScaledOpticalFlow:
                if self.backend == "tensorrt":
                    print(
                        "Dynamic Scaled Optical Flow does not work with TensorRT, disabling",
                        file=sys.stderr,
                    )

                elif self.UHDMode:
                    print(
                        "Dynamic Scaled Optical Flow does not work with UHD Mode, disabling",
                        file=sys.stderr,
                    )
                else:
                    from ..utils.SSIM import SSIM

                    CompareNet = SSIM().to(device=self.device, dtype=self.dtype)
                    possible_values = {
                        0.25: 0.25,
                        0.37: 0.5,
                        0.5: 1.0,
                        0.69: 1.5,
                        1.0: 2.0,
                    }  # closest_value:representative_scale
                    self.dynamicScale = DynamicScale(
                        possible_values=possible_values, CompareNet=CompareNet
                    )
                    print("Dynamic Scaled Optical Flow Enabled")

            if self.backend == "tensorrt":
                from .TensorRTHandler import TorchTensorRTHandler

                trtHandler = TorchTensorRTHandler(
                    trt_optimization_level=self.trt_optimization_level,
                    multi_precision_engine=True,
                )

                base_trt_engine_path = os.path.join(
                    os.path.realpath(self.trt_cache_dir),
                    (
                        f"{os.path.basename(self.interpolateModel)}"
                        + f"_{self.width}x{self.height}"
                        + f"_{'fp16' if self.dtype == torch.float16 else 'fp32'}"
                        + f"_scale-{self.scale}"
                        + f"_{torch.cuda.get_device_name(self.device)}"
                        + f"_trt-{trtHandler.tensorrt_version}"
                        + f"_ensemble-{self.ensemble}"
                        + f"_torch_tensorrt-{trtHandler.torch_tensorrt_version}"
                        + (
                            f"_level-{self.trt_optimization_level}"
                            if self.trt_optimization_level is not None
                            else ""
                        )
                    ),
                )
                trt_engine_path = base_trt_engine_path + ".dyn"
                encode_trt_engine_path = base_trt_engine_path + "_encode.dyn"

                # lay out inputs
                # load flow engine
                if not os.path.isfile(trt_engine_path):
                    if not self.doEncodingOnFrame:
                        exampleInput = [
                            torch.zeros(
                                [1, 3, self.ph, self.pw],
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros(
                                [1, 3, self.ph, self.pw],
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros(
                                [1, 1, self.ph, self.pw],
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros([2], dtype=torch.float, device=self.device),
                            torch.zeros(
                                [1, 2, self.ph, self.pw],
                                dtype=torch.float,
                                device=self.device,
                            ),
                        ]

                    else:
                        # if rife46
                        exampleInput = [
                            torch.zeros(
                                [1, 3, self.ph, self.pw],
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros(
                                [1, 3, self.ph, self.pw],
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros(
                                [1, 1, self.ph, self.pw],
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros([2], dtype=torch.float, device=self.device),
                            torch.zeros(
                                [1, 2, self.ph, self.pw],
                                dtype=torch.float,
                                device=self.device,
                            ),
                            torch.zeros(
                                (1, num_ch_for_encode, self.ph, self.pw),
                                dtype=self.dtype,
                                device=self.device,
                            ),
                            torch.zeros(
                                (1, num_ch_for_encode, self.ph, self.pw),
                                dtype=self.dtype,
                                device=self.device,
                            ),
                        ]

                        if not os.path.isfile(encode_trt_engine_path):
                            # build encode engine

                            encodedExampleInputs = [
                                torch.zeros(
                                    (1, 3, self.ph, self.pw),
                                    dtype=self.dtype,
                                    device=self.device,
                                ),
                            ]
                            trtHandler.build_engine(
                                model=self.encode,
                                dtype=self.dtype,
                                example_inputs=encodedExampleInputs,
                                device=self.device,
                                trt_engine_path=encode_trt_engine_path,
                            )

                        self.encode = trtHandler.load_engine(encode_trt_engine_path)

                    trtHandler.build_engine(
                        model=self.flownet,
                        dtype=self.dtype,
                        example_inputs=exampleInput,
                        device=self.device,
                        trt_engine_path=trt_engine_path,
                    )

                self.flownet = trtHandler.load_engine(trt_engine_path)
        torch.cuda.empty_cache()
        self.prepareStream.synchronize()

    @torch.inference_mode()
    def set_rife_args(self):
        self.tenFlow_div = torch.tensor(
            [(self.pw - 1.0) / 2.0, (self.ph - 1.0) / 2.0],
            dtype=torch.float32,
            device=self.device,
        )
        tenHorizontal = (
            torch.linspace(-1.0, 1.0, self.pw, dtype=torch.float32, device=self.device)
            .view(1, 1, 1, self.pw)
            .expand(-1, -1, self.ph, -1)
        ).to(dtype=torch.float32, device=self.device)
        tenVertical = (
            torch.linspace(-1.0, 1.0, self.ph, dtype=torch.float32, device=self.device)
            .view(1, 1, self.ph, 1)
            .expand(-1, -1, -1, self.pw)
        ).to(dtype=torch.float32, device=self.device)
        self.backwarp_tenGrid = torch.cat([tenHorizontal, tenVertical], 1)

    @torch.inference_mode()
    def __call__(
        self,
        img1,
        writeQueue: Queue,
        transition=False,
        upscaleModel: UpscalePytorch = None,
    ):  # type: ignore
        if self.frame0 is None:
                self.frame0 = self.frame_to_tensor(img1, self.prepareStream)
                if self.doEncodingOnFrame:
                    self.encode0 = self.encode_Frame(self.frame0, self.prepareStream)
                return

        frame1 = self.frame_to_tensor(img1, self.f2tStream)
        if self.doEncodingOnFrame:
            encode1 = self.encode_Frame(frame1, self.f2tStream)

        with torch.cuda.stream(self.stream):  # type: ignore
            

            if self.dynamicScaledOpticalFlow:
                closest_value = self.dynamicScale.dynamicScaleCalculation(
                    self.frame0, frame1
                )
            else:
                closest_value = None
            for n in range(self.ceilInterpolateFactor - 1):
                if not transition:
                    timestep = (n + 1) * 1.0 / (self.ceilInterpolateFactor)
                    while self.flownet is None:
                        sleep(1)
                    timestep = self.timestepDict[timestep]
                    if self.doEncodingOnFrame:
                        output = self.flownet(
                            self.frame0,
                            frame1,
                            timestep,
                            self.tenFlow_div,
                            self.backwarp_tenGrid,
                            self.encode0,
                            encode1,  # type: ignore
                            closest_value,
                        )
                    else:
                        output = self.flownet(
                            self.frame0,
                            frame1,
                            timestep,
                            self.tenFlow_div,
                            self.backwarp_tenGrid,
                            closest_value,
                        )
                    if upscaleModel is not None:
                        output = upscaleModel(
                            upscaleModel.frame_to_tensor(self.tensor_to_frame(output))
                        )
                    else:
                        output = self.tensor_to_frame(output)
                    writeQueue.put(output)
                else:
                    if upscaleModel is not None:
                        img1 = upscaleModel(frame1[:, :, : self.height, : self.width])
                    writeQueue.put(img1)

            self.copyTensor(self.frame0, frame1, self.copyStream)
            if self.doEncodingOnFrame:
                self.copyTensor(self.encode0, encode1, self.copyStream)  # type: ignore

        self.stream.synchronize()

    @torch.inference_mode()
    def encode_Frame(self, frame: torch.Tensor, stream: torch.cuda.Stream):
        while self.encode is None:
            sleep(1)
        with torch.cuda.stream(stream):  # type: ignore
            frame = self.encode(frame)
        stream.synchronize()
        return frame


class InterpolateRifeTensorRT(InterpolateRifeTorch):
    @torch.inference_mode()
    def __call__(
        self,
        img1,
        writeQueue: Queue,
        transition=False,
        upscaleModel: UpscalePytorch = None,
    ):  # type: ignore
        if self.frame0 is None:
            self.frame0 = self.frame_to_tensor(img1, self.prepareStream)
            if self.doEncodingOnFrame:
                self.encode0 = self.encode_Frame(self.frame0, self.prepareStream)
            return

        frame1 = self.frame_to_tensor(img1, self.f2tStream)
        if self.doEncodingOnFrame:
            encode1 = self.encode_Frame(frame1, self.f2tStream)
        with torch.cuda.stream(self.stream):  # type: ignore
            

            for n in range(self.ceilInterpolateFactor - 1):
                while self.flownet is None:
                    sleep(1)

                if not transition:
                    timestep = (n + 1) * 1.0 / (self.ceilInterpolateFactor)
                    timestep = self.timestepDict[timestep]

                    if self.doEncodingOnFrame:
                        output = self.flownet(
                            self.frame0,
                            frame1,
                            timestep,
                            self.tenFlow_div,
                            self.backwarp_tenGrid,
                            self.encode0,
                            encode1,  # type: ignore
                        )
                    else:
                        output = self.flownet(
                            self.frame0,
                            frame1,
                            timestep,
                            self.tenFlow_div,
                            self.backwarp_tenGrid,
                        )

                    if upscaleModel is not None:
                        output = upscaleModel(
                            upscaleModel.frame_to_tensor(self.tensor_to_frame(output))
                        )
                    else:
                        output = self.tensor_to_frame(output)

                    writeQueue.put(output)

                else:
                    if upscaleModel is not None:
                        img1 = upscaleModel(frame1[:, :, : self.height, : self.width])
                    writeQueue.put(img1)

            self.copyTensor(self.frame0, frame1, self.copyStream)
            if self.doEncodingOnFrame:
                self.copyTensor(self.encode0, encode1, self.copyStream)  # type: ignore

        self.stream.synchronize()