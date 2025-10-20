from abc import ABC, abstractmethod
from typing import List
import subprocess
import re
import cv2
from typing import Optional
import sys


FFMPEG_COLORSPACES = [
    "rgb",
    "bt709",
    "unknown",
    "reserved",
    "fcc",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "ycgco",
    "bt2020nc",
    "bt2020c",
    "smpte2085",
    "chroma-derived-nc",
    "chroma-derived-c",
    "ictcp"
]

FFMPEG_COLOR_PRIMARIES = [
    "reserved0",
    "bt709",
    "unknown",
    "reserved",
    "bt470m",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "film",
    "bt2020",
    "smpte428",
    "smpte431",
    "smpte432",
    "jedec-p22"
]
FFMPEG_COLOR_TRC = [
    "reserved0",
    "bt709",
    "unknown",
    "reserved",
    "bt470m",
    "bt470bg",
    "smpte170m",
    "smpte240m",
    "linear",
    "log100",
    "log316",
    "iec61966-2-4",
    "bt1361e",
    "iec61966-2-1",
    "bt2020-10",
    "bt2020-12",
    "smpte2084",
    "smpte428",
    "arib-std-b67"
]

if not __name__ == "__main__":
    from ..constants import FFMPEG_PATH
    from .Util import log, subprocess_popen_without_terminal

else:
    FFMPEG_PATH = "./bin/ffmpeg"
    from Util import log, subprocess_popen_without_terminal

class VideoInfo(ABC):
    @abstractmethod
    def get_duration_seconds(self) -> float: ...
    @abstractmethod
    def get_total_frames(self) -> int: ...
    @abstractmethod
    def get_width_x_height(self) -> List[int]: ...
    @abstractmethod
    def get_fps(self) -> float: ...
    @abstractmethod
    def get_color_space(self) -> str: ...
    @abstractmethod
    def get_pixel_format(self) -> str: ...
    @abstractmethod
    def get_color_transfer(self) -> str: ...
    @abstractmethod
    def get_color_primaries(self) -> str: ...
    @abstractmethod
    def get_bitrate(self) -> int: ...
    @abstractmethod
    def get_codec(self) -> str: ...
    @abstractmethod
    def is_hdr(self) -> bool: ...
    @abstractmethod
    def get_bit_depth(self) -> int: ...

class FFMpegInfoWrapper(VideoInfo):
    def __init__(self, input_file: str):
        self.input_file = input_file
        self.stream_line = None
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
        self.ffmpeg_output_raw = """
ffmpeg -i "D:\Youtube_ffv1_10bit_test.mov"-t 00:00:00 -f null NUL
ffmpeg version 2025-10-12-git-0bc54cddb1-full_build-www.gyan.dev Copyright (c) 2000-2025 the FFmpeg developers
  built with gcc 15.2.0 (Rev8, Built by MSYS2 project)
  configuration: --enable-gpl --enable-version3 --enable-static --disable-w32threads --disable-autodetect --enable-fontconfig --enable-iconv --enable-gnutls --enable-lcms2 --enable-libxml2 --enable-gmp --enable-bzlib --enable-lzma --enable-libsnappy --enable-zlib --enable-librist --enable-libsrt --enable-libssh --enable-libzmq --enable-avisynth --enable-libbluray --enable-libcaca --enable-libdvdnav --enable-libdvdread --enable-sdl2 --enable-libaribb24 --enable-libaribcaption --enable-libdav1d --enable-libdavs2 --enable-libopenjpeg --enable-libquirc --enable-libuavs3d --enable-libxevd --enable-libzvbi --enable-liboapv --enable-libqrencode --enable-librav1e --enable-libsvtav1 --enable-libvvenc --enable-libwebp --enable-libx264 --enable-libx265 --enable-libxavs2 --enable-libxeve --enable-libxvid --enable-libaom --enable-libjxl --enable-libvpx --enable-mediafoundation --enable-libass --enable-frei0r --enable-libfreetype --enable-libfribidi --enable-libharfbuzz --enable-liblensfun --enable-libvidstab --enable-libvmaf --enable-libzimg --enable-amf --enable-cuda-llvm --enable-cuvid --enable-dxva2 --enable-d3d11va --enable-d3d12va --enable-ffnvcodec --enable-libvpl --enable-nvdec --enable-nvenc --enable-vaapi --enable-libshaderc --enable-vulkan --enable-libplacebo --enable-opencl --enable-libcdio --enable-openal --enable-libgme --enable-libmodplug --enable-libopenmpt --enable-libopencore-amrwb --enable-libmp3lame --enable-libshine --enable-libtheora --enable-libtwolame --enable-libvo-amrwbenc --enable-libcodec2 --enable-libilbc --enable-libgsm --enable-liblc3 --enable-libopencore-amrnb --enable-libopus --enable-libspeex --enable-libvorbis --enable-ladspa --enable-libbs2b --enable-libflite --enable-libmysofa --enable-librubberband --enable-libsoxr --enable-chromaprint --enable-whisper
  libavutil      60. 13.100 / 60. 13.100
  libavcodec     62. 16.100 / 62. 16.100
  libavformat    62.  6.101 / 62.  6.101
  libavdevice    62.  2.100 / 62.  2.100
  libavfilter    11.  9.100 / 11.  9.100
  libswscale      9.  3.100 /  9.  3.100
  libswresample   6.  2.100 /  6.  2.100
Input #0, mov,mp4,m4a,3gp,3g2,mj2, from 'D:\Youtube_ffv1_10bit_test.mov':
  Metadata:
    major_brand     : qt
    minor_version   : 512
    compatible_brands: qt
    artist          : Just1n
    title           : Video Game Optimization Used To Be Borderline Black Magic
    date            : 2025
    encoder         : Lavf62.6.101
    comment         : https://www.youtube.com/watch?v=i-k9MGiiUR8
  Duration: 00:17:03.49, start: 0.000000, bitrate: 96128 kb/s
  Chapters:
    Chapter #0:0: start 0.000000, end 43.000000
      Metadata:
        title           : Context
    Chapter #0:1: start 43.000000, end 186.000000
      Metadata:
        title           : 3D Before 3D
    Chapter #0:2: start 186.000000, end 358.000000
      Metadata:
        title           : Perfect Dark
    Chapter #0:3: start 358.000000, end 813.000000
      Metadata:
        title           : PS2 Game Optimization
    Chapter #0:4: start 813.000000, end 884.000000
      Metadata:
        title           : The New Age
    Chapter #0:5: start 884.000000, end 962.000000
      Metadata:
        title           : Not Perfect. But Not The End Of The World
    Chapter #0:6: start 962.000000, end 1023.000000
      Metadata:
        title           : How Far We've Come
  Stream #0:0[0x1]: Video: ffv1 (FFV1 / 0x31564646), yuv420p10le(bt709, progressive), 1280x720, 95993 kb/s, SAR 1:1 DAR 16:9, 60 fps, 60 tbr, 15360 tbn (default)
    Metadata:
      handler_name    : ISO Media file produced by Google Inc.
      vendor_id       : FFMP
      encoder         : Lavc62.16.100 ffv1
  Stream #0:1[0x2](eng): Audio: aac (LC) (mp4a / 0x6134706D), 44100 Hz, stereo, fltp, 128 kb/s (default)
    Metadata:
      handler_name    : ISO Media file produced by Google Inc.
      vendor_id       : [0][0][0][0]
  Stream #0:2[0x3](eng): Data: bin_data (text / 0x74786574), 0 kb/s
    Metadata:
      handler_name    : SubtitleHandler
Stream mapping:
  Stream #0:0 -> #0:0 (ffv1 (native) -> wrapped_avframe (native))
  Stream #0:1 -> #0:1 (aac (native) -> pcm_s16le (native))
Press [q] to stop, [?] for help
Output #0, null, to 'NUL':
  Metadata:
    major_brand     : qt
    minor_version   : 512
    compatible_brands: qt
    artist          : Just1n
    title           : Video Game Optimization Used To Be Borderline Black Magic
    date            : 2025
    comment         : https://www.youtube.com/watch?v=i-k9MGiiUR8
    encoder         : Lavf62.6.101
  Chapters:
    Chapter #0:0: start 0.000000, end 0.000000
      Metadata:
        title           : Context
  Stream #0:0: Video: wrapped_avframe, yuv420p10le(bt709, progressive), 1280x720 [SAR 1:1 DAR 16:9], q=2-31, 200 kb/s, 60 fps, 60 tbn (default)
    Metadata:
      encoder         : Lavc62.16.100 wrapped_avframe
      handler_name    : ISO Media file produced by Google Inc.
      vendor_id       : FFMP
  Stream #0:1(eng): Audio: pcm_s16le, 44100 Hz, stereo, s16, 1411 kb/s (default)
    Metadata:
      encoder         : Lavc62.16.100 pcm_s16le
      handler_name    : ISO Media file produced by Google Inc.
      vendor_id       : [0][0][0][0]
[out#0/null @ 0000022552649800] video:0KiB audio:0KiB subtitle:0KiB other streams:0KiB global headers:0KiB muxing overhead: unknown
[out#0/null @ 0000022552649800] Output file is empty, nothing was encoded(check -ss / -t / -frames parameters if used)
frame=    0 fps=0.0 q=0.0 Lsize=N/A time=N/A bitrate=N/A speed=N/A elapsed=0:00:00.06"""
        self.ffmpeg_output_stripped = self.ffmpeg_output_raw.lower().strip()
        try:
            for line in self.ffmpeg_output_raw.split("\n"):
                if "Stream #" in line and "Video" in line:
                    self.stream_line = line
                    break
            
            
            if self.stream_line is None:
                log("No video stream found in the input file.")
        except Exception:
            log(f"ERROR: Input file seems to have no video stream!", file=sys.stderr)
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
    
    def check_color_opt(self, color_opt:str) -> str | None:
        if self.stream_line:
            if "ffv1" in self.get_codec():
                string_pattern = "1,"
            else:
                string_pattern = "),"
            try:
                match color_opt:
                    case "Space":
                        color_opt_detected = self.stream_line.split(string_pattern)[1].split(",")[1].split("/")[0].strip()
                        if color_opt_detected not in FFMPEG_COLORSPACES:
                            return None
                    case "Primaries":
                        color_opt_detected = self.stream_line.split(string_pattern)[1].split("/")[1].strip()
                        if color_opt_detected not in FFMPEG_COLOR_PRIMARIES:
                            return None
                    case "Transfer":
                        color_opt_detected = self.stream_line.split(string_pattern)[1].split("/")[2].replace(")","").split(",")[0].strip()
                        if color_opt_detected not in FFMPEG_COLOR_TRC:
                            return None

                if "progressive" in color_opt_detected.lower():
                    return None
                if "unknown" in color_opt_detected.lower():
                    return None
                
                if len(color_opt_detected.strip()) > 1:
                    return color_opt_detected
                
            except Exception:
                return None
        return None
    
    def get_color_space(self) -> str:
        return self.check_color_opt("Space")

    def get_color_primaries(self) -> str:
        return self.check_color_opt("Primaries")

    def get_color_transfer(self) -> str:
        return self.check_color_opt("Transfer")

    def get_pixel_format(self) -> str:
        if self.stream_line:
            try:
                pixel_format = self.stream_line.split(",")[1].split("(")[0].strip()
                return pixel_format
            except Exception:
                log("ERROR: Cant detect pixel format.")
        return None
    
    def is_hdr(self) -> bool:
        hdr_indicators = ["bt2020", "pq", "hdr10", "dolby vision", "hlg"]
        for indicator in hdr_indicators:
            if indicator in self.ffmpeg_output_stripped:
                return True
        return False
    
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
    
    def get_bit_depth(self) -> int:
        return 10 if "p10le" in self.ffmpeg_output_stripped else 8
    


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
        return duration

    def get_total_frames(self) -> int:
        
        if self.start_time or self.end_time:
            fc = int(self.get_duration_seconds() * self.get_fps())
        else:
            fc =  int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        return fc

    def get_width_x_height(self) -> List[int]:
        res = [int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))]
        return res


    def get_fps(self) -> float:
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        return fps
    
    def get_color_space(self) -> str:
        return self.ffmpeg_info.get_color_space()
    
    def get_pixel_format(self) -> str:
        return self.ffmpeg_info.get_pixel_format()
    
    def get_color_transfer(self) -> str:
        return self.ffmpeg_info.get_color_transfer()

    def get_color_primaries(self) -> str:
        return self.ffmpeg_info.get_color_primaries()

    def get_bitrate(self) -> int:
        return self.ffmpeg_info.get_bitrate()
    
    def get_codec(self) -> str:
        return self.ffmpeg_info.get_codec()
    
    def is_hdr(self) -> bool:
        return self.ffmpeg_info.is_hdr()
    
    def get_bit_depth(self) -> int:
        return self.ffmpeg_info.get_bit_depth()
    

    def __del__(self):
        self.cap.release()

def print_video_info(video_info: VideoInfo):
    print(f"Duration: {video_info.get_duration_seconds()} seconds")
    print(f"Total Frames: {video_info.get_total_frames()}")
    print(f"Resolution: {video_info.get_width_x_height()[0]}x{video_info.get_width_x_height()[1]}")
    print(f"FPS: {video_info.get_fps()}")
    print(f"Color Space: {video_info.get_color_space()}")
    print(f"Color Transfer: {video_info.get_color_transfer()}")
    print(f"Color Primaries: {video_info.get_color_primaries()}")
    print(f"Pixel Format: {video_info.get_pixel_format()}")
    print(f"Video Codec: {video_info.get_codec()}")
    print(f"Video Bitrate: {video_info.get_bitrate()} kbps")
    print(f"Is HDR: {video_info.is_hdr()}")
    print(f"Bit Depth: {video_info.get_bit_depth()}")

__all__ = ["FFMpegInfoWrapper", "OpenCVInfo", "print_video_info"]

if __name__ == "__main__":
    video_path = "/home/pax/Downloads/Life Untouched 4K Demo.mp4"
    #video_path = "/home/pax/Documents/test/LG New York HDR UHD 4K Demo.ts"
    #video_path = "/home/pax/Documents/test/out.mkv"
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
    print(f"Video Codec: {video_info.get_codec()}")
    print(f"Video Bitrate: {video_info.get_bitrate()} kbps")
    print(f"Is HDR: {video_info.is_hdr()}")
    print(f"Bit Depth: {video_info.get_bit_depth()}")