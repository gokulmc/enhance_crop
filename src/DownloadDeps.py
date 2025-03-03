from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional
from PySide6.QtWidgets import QMessageBox

from numpy import extract

from .constants import (
    PLATFORM,
    PYTHON_DIRECTORY,
    PYTHON_EXECUTABLE_PATH,
    PYTHON_VERSION,
    FFMPEG_PATH,
    BACKEND_PATH,
    TEMP_DOWNLOAD_PATH,
    CWD,
    HAS_NETWORK_ON_STARTUP,
)
from .version import version, backend_dev_version
from .Util import (
    FileHandler,
    log,
    createDirectory,
    makeExecutable,
    move,
    extractTarGZ,
    downloadFile,
    removeFolder,
)
from .ui.QTcustom import (
    DownloadProgressPopup,
    DisplayCommandOutputPopup,
    RegularQTPopup,
    needs_network_else_exit,
)
import os
from platform import machine
import subprocess


def run_executable(exe_path):
    try:
        # Run the executable and wait for it to complete
        result = subprocess.run(exe_path, check=True, capture_output=True, text=True)

        # Print the output of the executable
        print("STDOUT:", result.stdout)

        # Print any error messages
        print("STDERR:", result.stderr)

        # Print the exit code
        print("Exit Code:", result.returncode)

    except subprocess.CalledProcessError as e:
        print("An error occurred while running the executable.")
        print("Exit Code:", e.returncode)
        print("Output:", e.output)
        print("Error:", e.stderr)
        return False
    except FileNotFoundError:
        print("The specified executable was not found.")
        return False
    except Exception as e:
        print("An unexpected error occurred:", str(e))
        return False
    return True

@dataclass
class Dependency(ABC):
    updatable: bool
    download_path:str
    installed_path:str

    def __init__(self):
        FileHandler.createDirectory(os.path.dirname(self.download_path))

    @abstractmethod
    def get_download_link(self) -> str: ...
        
    @abstractmethod
    def download(self) -> None: ...

    def get_if_update_available(self) -> bool: ...
    def update_if_updates_available(self) -> None: ...

class Backend(Dependency):
    updatable: bool = True
    is_update_available: bool
    download_path = os.path.join(CWD, "backend.tar.gz")
    installed_path = BACKEND_PATH


    def get_download_link(self) -> str:
        backend_url = f"https://github.com/TNTwise/REAL-Video-Enhancer/releases/download/RVE-{version}/backend-v{version}.tar.gz"
        return backend_url
    
    def download(self):
        download_link = self.get_download_link()
        DownloadProgressPopup(link=download_link, downloadLocation=self.download_path, title="Downloading Backend")
        extractTarGZ(self.download_path)
    
    def get_if_update_available(self) -> bool:
        try:
            output = subprocess.run([PYTHON_EXECUTABLE_PATH, os.path.join(BACKEND_PATH, "rve-backend.py"), "--version"], check=True, capture_output=True, text=True)
            output = output.stdout.strip() # this extracts the version number from the output
            log(f"Backend Version: {output}")
            update_available = not output == backend_dev_version
            self.is_update_available = update_available
            return update_available
        except subprocess.CalledProcessError: # if the backend is not found
            self.download()
            self.is_update_available = False
            return False
    
    def update_if_updates_available(self) -> None:
        if self.is_update_available:
            needs_network_else_exit()
            FileHandler.removeFolder(BACKEND_PATH) # remove the old backend directory
            self.download()


class Python(Dependency):
    download_path = os.path.join(CWD, "python", "python.tar.gz")
    installed_path = PYTHON_DIRECTORY
    is_update_available: bool

    def get_download_link(self) -> str:
        link = f"https://github.com/indygreg/python-build-standalone/releases/download/20250205/cpython-{PYTHON_VERSION}+20250205-"
       
        match PLATFORM:
            case "linux":
                link += "x86_64-unknown-linux-gnu-install_only.tar.gz"
            case "win32":
                link += "x86_64-pc-windows-msvc-install_only.tar.gz"
            case "darwin":
                if machine() == "arm64":
                    link += "aarch64-apple-darwin-install_only.tar.gz"
                else:
                    link += "x86_64-apple-darwin-install_only.tar.gz"

        return link

    def download(self):
        needs_network_else_exit()
        download_link = self.get_download_link()
        FileHandler.createDirectory(os.path.dirname(self.download_path))
        DownloadProgressPopup(link = download_link, downloadLocation=self.download_path, title = f"Downloading Python {PYTHON_VERSION}")
        extractTarGZ(self.download_path)
    
    def get_version(self):
        return subprocess.run([PYTHON_EXECUTABLE_PATH, "--version"], check=True, capture_output=True, text=True).stdout.strip().split(" ")[1] # this extracts the version number from the output
    
    def get_if_update_available(self) -> bool:
        try:
            output = self.get_version()
        except subprocess.CalledProcessError: # if python is not found
            self.download()
            return False
        
        is_update = not output == PYTHON_VERSION

        if is_update:
            reply = QMessageBox.question(
                None,
                "Update Python?",
                "The installed version of Python is older than the current version of RVE. Update?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,  # type: ignore
            )
            if reply == QMessageBox.Yes:  # type: ignore
                self.is_update_available = True
                print("Updating Python")
            else:
                is_update = False
        self.is_update_available = is_update
        return self.is_update_available

    def update_if_updates_available(self) -> None:

        if self.is_update_available:
            removeFolder(PYTHON_DIRECTORY)
            self.download()

class FFMpeg(Dependency):
    download_path = os.path.join(CWD, "ffmpeg")
    installed_path = FFMPEG_PATH

    def get_download_link(self) -> str:
        link = "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/"
        match PLATFORM:
            case "linux":
                link += "ffmpeg"
            case "win32":
                link += "ffmpeg.exe"
            case "darwin":
                link += "ffmpeg-macos-bin"
        return link

    def download(self):
        
        needs_network_else_exit()

        download_link = self.get_download_link()
        DownloadProgressPopup(link=download_link, downloadLocation=self.download_path, title="Downloading FFMpeg")
        FileHandler.createDirectory(os.path.dirname(self.installed_path))
        FileHandler.moveFile(self.download_path, self.installed_path)
        FileHandler.makeExecutable(self.installed_path)

class VCRedList(Dependency):
    updatable = False
    download_path = os.path.join(CWD, "bin", "VC_redist.x64.exe")
    installed_path = download_path

    def get_download_link(self) -> str:
        return "https://aka.ms/vs/17/release/vc_redist.x64.exe"
    
    def download(self):
        if PLATFORM == 'win32':
            needs_network_else_exit()

            download_link = self.get_download_link()
            DownloadProgressPopup(link = download_link, downloadLocation=self.download_path, title = "Downloading VCRedist")
            if not run_executable(
                [self.download_path, "/install", "/quiet", "/norestart"]
            ):  # keep trying until user says yes
                RegularQTPopup(
                    "Please click yes to allow VCRedlist to install!\nThe installer will now close."
                )




class DownloadDependencies:
    """
    Downloads platform specific dependencies python and ffmpeg to their respective locations and creates the directories

    """
    def __init__(self, use_torch_nightly:bool = False):
        self.use_torch_nightly = use_torch_nightly
    def download_all_deps(self):
        for dep in Dependency.__subclasses__():
            d = dep()
            d.download()

    def pip(
        self,
        deps: list,
        install: bool = True,
    ):  # going to have to make this into a qt module pop up
        createDirectory(TEMP_DOWNLOAD_PATH)
        command = [
            PYTHON_EXECUTABLE_PATH,
            "-m",
            "pip",
            "install" if install else "uninstall",
        ]
        origTemp = os.environ.get("TMPDIR")
        os.environ["TMPDIR"] = TEMP_DOWNLOAD_PATH
        if install:
            command += [
                "--no-warn-script-location",
                "--extra-index-url",
                "https://download.pytorch.org/whl/",  # switch to normal whl and test
                "--extra-index-url",
                "https://pypi.nvidia.com",
            ]
        else:
            command += ["-y"]
        command += deps
        # totalDeps = self.get_total_dependencies(deps)
        totalDeps = len(deps)
        log("Downloading Deps: " + str(command))
        log("Total Dependencies: " + str(totalDeps))

        DisplayCommandOutputPopup(
            command=command,
            title="Download Dependencies",
            progressBarLength=totalDeps,
        )
        command = [
            PYTHON_EXECUTABLE_PATH,
            "-m",
            "pip",
            "cache",
            "purge",
        ]
        DisplayCommandOutputPopup(
            command=command,
            title="Purging Cache",
            progressBarLength=1,
        )
        if origTemp:
            os.environ["TMPDIR"] = str(origTemp)
        removeFolder(TEMP_DOWNLOAD_PATH)

    def getPlatformIndependentDeps(self):
        platformIndependentdeps = [
            "testresources",
            "requests",
            "opencv-python-headless",
            "pypresence",
            "scenedetect",
            "numpy==2.2.2",
            "sympy==1.13.1",
            "tqdm",
            "typing_extensions",
            "packaging",
            "mpmath",
            "pillow",
        ]
        return platformIndependentdeps

    def getPyTorchCUDADeps(self):
        """
        Installs:
        Default deps
        Pytorch CUDA deps
        """
        torchCUDADeps = [
            "torch==2.6.0+cu126",  #
            "torchvision==0.21.0+cu126",
            "safetensors",
            "einops",
            "cupy-cuda12x==13.3.0",
        ]
        return torchCUDADeps

    def getTensorRTDeps(self):
        """
        Installs:
        Default deps
        Pytorch CUDA deps
        TensorRT deps
        """
        tensorRTDeps = [
            "tensorrt==10.8.0.43",
            "tensorrt_cu12==10.8.0.43",
            "tensorrt-cu12_libs==10.8.0.43",
            "tensorrt_cu12_bindings==10.8.0.43",
            "--no-deps",
            "torch_tensorrt==2.6.0+cu126",
        ]

        return tensorRTDeps

    def downloadPyTorchCUDADeps(self, install: bool = True):
        if install:
            self.pip(self.getPlatformIndependentDeps())
        self.pip(self.getPyTorchCUDADeps(), install)

    def downloadTensorRTDeps(self, install: bool = True):
        if install:
            self.pip(self.getPlatformIndependentDeps())
        self.pip(
            self.getPyTorchCUDADeps(),
            install,
        )
        self.pip(
            self.getTensorRTDeps(),  # Has to be in this order, because i skip dependency check for torchvision
            install,
        )

    def downloadDirectMLDeps(self, install: bool = True):
        directMLDeps = [
            "onnxruntime-directml",
            "onnx",
            "onnxconverter-common",
        ] + self.getPlatformIndependentDeps()
        self.pip(directMLDeps, install)

    def downloadNCNNDeps(self, install: bool = True):
        """
        Installs:
        Default deps
        NCNN deps
        """
        if install:
            self.pip(self.getPlatformIndependentDeps())
        ncnnDeps = [
            "rife-ncnn-vulkan-python-tntwise==1.4.5",
            "upscale_ncnn_py==1.2.0",
            "ncnn==1.0.20240820",
            "numpy==2.2.2",
            "opencv-python-headless",
            "mpmath",
            "sympy==1.13.1",
        ]
        self.pip(ncnnDeps, install)

    def downloadPyTorchROCmDeps(self, install: bool = True):
        if install:
            self.pip(self.getPlatformIndependentDeps())

        rocmLinuxDeps = [
            "torch==2.6.0+rocm6.2.4",
            "torchvision==0.21.0+rocm6.2.4",
            "einops",
            "safetensors",
        ]
        if PLATFORM == "linux":
            self.pip(rocmLinuxDeps, install)


if __name__ == "__main__":
    downloadDependencies = DownloadDependencies()
    downloadDependencies.downloadPython()