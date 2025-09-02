import subprocess
import re
from typing import List
from .Util import log, subprocess_popen_without_terminal
from .constants import FFMPEG_PATH
import sys

class FFMpegInfoWrapper:
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
            
            log(f"Stream line: {self.stream_line}")
            if self.stream_line is None:
                log("No video stream found in the input file.")
        except Exception:
            print(f"ERROR: Input file seems to have no video stream!", file=sys.stderr)
            self.stream_line = ""            

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

    def check_color_opt(self, color_opt:str) -> str | None:
        if self.stream_line:
            try:
                match color_opt:
                    case "Space":
                        color_opt_detected = self.stream_line.split("),")[1].split(",")[1].split("/")[0].strip()
                    case "Primaries":
                        color_opt_detected = self.stream_line.split("),")[1].split("/")[1].strip()
                    case "Transfer":
                        color_opt_detected = self.stream_line.split("),")[1].split("/")[2].replace(")","").split(",")[0].strip()
                        
                if "progressive" in color_opt_detected.lower():
                    return None
                if "unknown" in color_opt_detected.lower():
                    return None
                if len(color_opt_detected.strip()) > 1:
                    log(f"Color {color_opt}: {color_opt_detected}")
                    return color_opt_detected
                
            except Exception:
                log(f"No known color {color_opt} detected in the input file.")
                return None
        return None
    
    def get_color_space(self) -> str:
        return self.check_color_opt("Space")

    def get_color_primaries(self) -> str:
        return self.check_color_opt("Primaries")

    def get_color_transfer(self) -> str:
        return self.check_color_opt("Transfer")
    
    def get_pixel_format(self) -> str:
        try:
            pixel_format = self.stream_line.split(",")[1].split("(")[0].strip()
            log(f"Pixel Format: {pixel_format}")
        except Exception:
            log("ERROR: Cant detect pixel format.")
            pixel_format = None 
        return pixel_format
    
    def get_bitrate(self) -> int:
        bitrate = re.search(r"bitrate: (\d+)", self.ffmpeg_output_stripped)
        if bitrate:
            return int(bitrate.groups()[0])
        return 0
    
    def get_codec(self) -> str:
        codec = re.search(r"video: (\w+)", self.ffmpeg_output_stripped)
        if codec:
            return codec.groups()[0]
        return "unknown"
    
    def is_hdr(self) -> bool:
        hdr_indicators = ["bt2020", "pq", "hdr10", "dolby vision", "hlg"]
        for indicator in hdr_indicators:
            if indicator in self.ffmpeg_output_stripped:
                return True
        return False

class VideoLoader:
    def __init__(self, inputFile):
        self.inputFile = inputFile

    def loadVideo(self):
        log(f"Loading video file: {self.inputFile}")
        self.ffmpeg_info = FFMpegInfoWrapper(self.inputFile)

    def isValidVideo(self):
        try:
            log(f"Checking if video file is valid: {self.inputFile}")
            disabled_extensions = ["txt", "jpg", "jpeg", "png", "bmp", "webp"]
            file_extension = self.inputFile.split(".")[-1].lower()
            return self.ffmpeg_info.get_total_frames() > 1 and \
                    file_extension not in disabled_extensions
        except Exception as e:
            log(f"Error checking video validity: {e}")
            return False

    def getData(self):
        self.width, self.height = self.ffmpeg_info.get_width_x_height()
        
        self.bitrate = self.ffmpeg_info.get_bitrate()
        self.videoContainer = self.inputFile.split(".")[-1]
        self.codec_str = self.ffmpeg_info.get_codec()
       
        self.fps = self.ffmpeg_info.get_fps()
        self.total_frames = int(self.ffmpeg_info.get_total_frames())
        self.duration = self.total_frames / self.fps
        self.color_space = self.ffmpeg_info.get_color_space()
        self.pixel_format = self.ffmpeg_info.get_pixel_format()
        self.is_hdr = self.ffmpeg_info.is_hdr()
        log(f"Video Data: {self.width}x{self.height}, {self.fps} FPS, {self.total_frames} frames, {self.duration} seconds, "
            f"Bitrate: {self.bitrate}, Codec: {self.codec_str}, Color Space: {self.color_space}, Pixel Format: {self.pixel_format}")
