import sys
import os
sys.path.append(os.getcwd())
from backend.src.utils.GetFFMpeg import download_ffmpeg

if __name__ == "__main__":
    download_ffmpeg()
