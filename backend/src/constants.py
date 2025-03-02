import os
import sys
global CWD

def checkForCUDA() -> bool:
    try:
        import torch
        import torchvision
        import cupy

        if cupy.cuda.get_cuda_path() == None:
            return False
    except Exception as e:
        return False
    return True


__version__ = "2.1.5"
IS_FLATPAK = "FLATPAK_ID" in os.environ
HOME_PATH = os.path.expanduser("~")
PLATFORM = sys.platform  # win32, darwin, linux


if IS_FLATPAK:
    CWD = (
        os.path.join(
            HOME_PATH, ".var", "app", "io.github.tntwise.REAL-Video-Enhancer"
        )
    )

CWD = os.getcwd()

def set_manual_cwd(cwd):
    global CWD
    CWD = cwd

FFMPEG_PATH = os.path.join(CWD, "bin", "ffmpeg")
FFMPEG_LOG_FILE = os.path.join(CWD, "ffmpeg_log.txt")
MODELS_DIRECTORY = os.path.join(CWD, "models")
HAS_SYSTEM_CUDA = checkForCUDA()
