from abc import ABC, abstractmethod
from typing import List
import subprocess
import re
import cv2

if not __name__ == "__main__":
    from ..constants import FFMPEG_PATH
else:
    FFMPEG_PATH = "./bin/ffmpeg"

class VideoInfo(ABC):
    @abstractmethod
    def get_duration_seconds(self) -> float:
        pass
    @abstractmethod
    def get_total_frames(self) -> int:
        pass
    @abstractmethod
    def get_width_x_height(self) -> List[int]:
        pass
    @abstractmethod
    def get_fps(self) -> float:
        pass


class FFMpegInfoWrapper(VideoInfo):
    def __init__(self, input_file: str):
        self.input_file = input_file
        self._get_ffmpeg_info()
        
    def _get_ffmpeg_info(self):
        command = [
                FFMPEG_PATH,
                "-hide_banner",
                "-i",
                self.input_file,
                "-f",
                "null",
        ]
        
        self.ffmpeg_output = subprocess.Popen(command,  stderr=subprocess.PIPE, errors="replace").stderr.read()    
    
    def get_duration_seconds(self) -> float:
        total_duration:float = 0.0

        duration = re.search(r"Duration: (.*?),", self.ffmpeg_output).groups()[0]
        hours, minutes, seconds = duration.split(":")
        total_duration += int(int(hours) * 3600)
        total_duration += int(int(minutes) * 60)
        total_duration += float(seconds)
        return round(total_duration, 2)

    def get_total_frames(self) -> int:
        return int(self.get_duration_seconds() * self.get_fps())

    def get_width_x_height(self) -> List[int]:
        width, height = re.search(r"Video:.* (\d+)x(\d+)", self.ffmpeg_output).groups()[:2]
        return [int(width), int(height)]

    def get_fps(self) -> float:
        fps = re.search(r"(\d+\.?\d*) fps", self.ffmpeg_output).groups()[0]
        return float(fps)


class OpenCVInfo(VideoInfo):
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.cap = cv2.VideoCapture(input_file)
    
    def get_duration_seconds(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.get_fps()
    
    def get_total_frames(self) -> int:
        return int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def get_width_x_height(self) -> List[int]:
        return [int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
    
    def get_fps(self) -> float:
        return self.cap.get(cv2.CAP_PROP_FPS)
    
    def __del__(self):
        self.cap.release()


__all__ = ["FFMpegInfoWrapper", "OpenCVInfo"]

