from ..constants import FFMPEG_PATH, CPU_ARCH, PLATFORM, CWD
import os
from .FileHandler import FileHandler
import requests
download_path = os.path.join(CWD, "ffmpeg")
installed_path = FFMPEG_PATH

def download_ffmpeg():
    if not os.path.exists(installed_path):
        link = "https://github.com/TNTwise/real-video-enhancer-models/releases/download/models/"
        match PLATFORM:
            case "linux":
                link += "ffmpeg" if CPU_ARCH == "x86_64" else "ffmpeg-linux-arm64"
            case "win32":
                link += "ffmpeg.exe" if CPU_ARCH == "x86_64" else "ffmpeg-windows-arm64.exe"
            case "darwin":
                link += "ffmpeg-macos-bin" if CPU_ARCH == "x86_64" else "ffmpeg-macos-arm"

        print("Downloading FFMpeg from " + link)
        response = requests.get(link, stream=True)
        total_size = int(response.headers.get("content-length", 0))
        block_size = 1024  
        with open(download_path, "wb") as file:
            for data in response.iter_content(block_size):
                file.write(data)
        print("Download completed.")

        FileHandler.createDirectory(os.path.dirname(installed_path))
        FileHandler.moveFile(download_path, installed_path)
        FileHandler.makeExecutable(installed_path)