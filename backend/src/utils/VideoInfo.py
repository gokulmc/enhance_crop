from abc import ABC, abstractmethod
from typing import List
import subprocess
import re
import cv2
from typing import Optional
import sys

if not __name__ == "__main__":
    from ..constants import FFMPEG_PATH
    from .Util import log, subprocess_popen_without_terminal, printAndLog

else:
    FFMPEG_PATH = "./bin/ffmpeg"
    from Util import log, subprocess_popen_without_terminal, printAndLog

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
    @abstractmethod
    def get_color_space(self) -> str:
        pass


class FFMpegInfoWrapper(VideoInfo):
    def __init__(self, input_file: str):
        self.input_file = input_file
        self._get_ffmpeg_info()

    def _get_ffmpeg_info(self):
        command = [
                FFMPEG_PATH,
                "-i",
                self.input_file,
                "-t",
                "00:00:00",
                "-f",
                "null",
                "/dev/null",
                "-hide_banner",
                
                
        ]

        self.ffmpeg_output_raw:str = subprocess_popen_without_terminal(command,  stderr=subprocess.PIPE, errors="replace").stderr.read()
        self.ffmpeg_output_stripped = self.ffmpeg_output_raw.lower().strip()
        try:
            for line in self.ffmpeg_output_raw.split("\n"):
                if "Stream #" in line and "Video" in line:
                    self.stream_line = line
                    break
            
            
            if self.stream_line is None:
                log("No video stream found in the input file.")
        except Exception:
            print(f"ERROR: Input file seems to have no video stream!", file=sys.stderr)
            exit(1)
            

    def get_duration_seconds(self) -> float:
        total_duration:float = 0.0

        duration = re.search(r"duration: (.*?),", self.ffmpeg_output_stripped).groups()[0]
        hours, minutes, seconds = duration.split(":")
        total_duration += int(int(hours) * 3600)
        total_duration += int(int(minutes) * 60)
        total_duration += float(seconds)
        return round(total_duration, 2)

    def get_total_frames(self) -> int:
        return int(self.get_duration_seconds() * self.get_fps())

    def get_width_x_height(self) -> List[int]:
        width, height = re.search(r"video:.* (\d+)x(\d+)",self.ffmpeg_output_stripped).groups()[:2]
        return [int(width), int(height)]

    def get_fps(self) -> float:
        fps = re.search(r"(\d+\.?\d*) fps", self.ffmpeg_output_stripped).groups()[0]
        return float(fps)
    
    def get_color_space(self) -> str:
        if self.stream_line:
            try:
                color_space = self.stream_line.split("),")[1].split(",")[1].split("/")[0].strip()
                if "progressive" in color_space.lower():
                    return None
                if len(color_space.strip()) > 1:
                    log(f"Color Space: {color_space}")
                    return color_space
            except Exception:
                printAndLog("No known color space detected in the input file.")
            return None
        return None
    
    def get_color_primaries(self) -> str:
        if self.stream_line:
            try:
                color_space = self.stream_line.split("),")[1].split("/")[1].strip()
                if len(color_space.strip()) > 1:
                    log(f"Color Primaries: {color_space}")
                    return color_space
            except Exception:
                log("No known color primaries detected in the input file.")
            return None
        return None
    
    def get_color_transfer(self) -> str:
        if self.stream_line:
            try:
                color_space = self.stream_line.split("),")[1].split("/")[2].replace(")","").split(",")[0].strip()
                if len(color_space.strip()) > 1:
                    log(f"Color Transfer: {color_space}")
                    return color_space
            except Exception:
                log("No known color transfer detected in the input file.")
            return None
        return None
    
    def get_pixel_format(self) -> str:
        try:
            pixel_format = self.stream_line.split(",")[1].split("(")[0].strip()
            log(f"Pixel Format: {pixel_format}")
        except Exception:
            log("ERROR: Cant detect pixel format.")
            pixel_format = None 
        return pixel_format



class OpenCVInfo(VideoInfo):
    def __init__(self, input_file: str, start_time: Optional[float] = None, end_time: Optional[float] = None):
        log("Getting Input Video Properties")
        self.input_file = input_file
        self.start_time = start_time
        self.end_time = end_time
        self.cap = cv2.VideoCapture(input_file)
        self.ffmpeg_info = FFMpegInfoWrapper(input_file)

    def is_valid_video(self):
        #frame_count = self.cap.get(cv2.CAP_PROP_FRAME_COUNT)
        #log(f"Frame count: {frame_count}")
        #if frame_count <= 1:
        #    log("Invalid video: Frame count is less than or equal to 1.")
        #    return False
        
        return self.cap.isOpened() and self.cap.get(cv2.CAP_PROP_FRAME_COUNT) 

    def get_duration_seconds(self) -> float:
        duration = self.cap.get(cv2.CAP_PROP_FRAME_COUNT) / self.get_fps()

        if self.start_time is not None and self.end_time is not None:
            duration = self.end_time - self.start_time
        elif self.start_time and not self.end_time:
            duration = duration - self.start_time
        elif self.end_time and not self.start_time:
            duration = self.end_time
        log(f"Duration: {duration}")
        return duration

    def get_total_frames(self) -> int:
        
        if self.start_time or self.end_time:
            fc = int(self.get_duration_seconds() * self.get_fps())
        else:
            fc =  int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        log(f"Frame count: {fc}")
        return fc

    def get_width_x_height(self) -> List[int]:
        res = [int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        log(f"Resoltion {res[0]}x{res[1]}")
        return res


    def get_fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        log(f"FPS: {fps}")
        return fps
    
    def get_color_space(self) -> str:
        return self.ffmpeg_info.get_color_space()
    
    def get_pixel_format(self) -> str:
        return self.ffmpeg_info.get_pixel_format()
    
    def get_color_transfer(self) -> str:
        return self.ffmpeg_info.get_color_transfer()

    def get_color_primaries(self) -> str:
        return self.ffmpeg_info.get_color_primaries()

    def __del__(self):
        self.cap.release()


__all__ = ["FFMpegInfoWrapper", "OpenCVInfo"]

if __name__ == "__main__":
    #video_path = "/home/pax/Downloads/juggle.mp4"
    #video_path = "/home/pax/Documents/test/LG New York HDR UHD 4K Demo.ts"
    video_path = "/home/pax/Documents/test/out.mkv"
    #video_path = "/home/pax/Videos/TVアニメ「WIND BREAKER Season 2」ノンクレジットオープニング映像「BOYZ」SixTONES [AWlUVr7Du04]_gmfss-pro_deh264-span_janai-v2_72.0fps_3840x2160.mkv"
    """print("Using FFMpeg:")
    video_info = FFMpegInfoWrapper(video_path)
    print(f"Duration: {video_info.get_duration_seconds()} seconds")
    print(f"Total Frames: {video_info.get_total_frames()}")
    print(f"Resolution: {video_info.get_width_x_height()}")
    print(f"FPS: {video_info.get_fps()}")
    print(f"Color Space: {video_info.get_color_space()}")
    print("\nUsing OpenCV:")"""
    video_info = OpenCVInfo(video_path)
    print(f"Duration: {video_info.get_duration_seconds()} seconds")
    print(f"Total Frames: {video_info.get_total_frames()}")
    print(f"Resolution: {video_info.get_width_x_height()}")
    print(f"FPS: {video_info.get_fps()}")
    print(f"Color Space: {video_info.get_color_space()}")
    print(f"Color Transfer: {video_info.get_color_transfer()}")
    print(f"Color Primaries: {video_info.get_color_primaries()}")
    print(f"Pixel Format: {video_info.get_pixel_format()}")