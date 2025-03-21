from abc import ABC
from dataclasses import dataclass
from typing import Optional
from .ui.SettingsTab import Settings

@dataclass
class Encoder(ABC):
    preset_tag: str
    preInputsettings: Optional[str]
    postInputSettings: Optional[str]


@dataclass
class VideoEncoder(Encoder):
    qualityControlMode: Optional[str] = "-crf"
    ...


@dataclass
class AudioEncoder(Encoder): ...


@dataclass
class SubtitleEncoder(Encoder): ...


# audio encoder options
class copyAudio(AudioEncoder):
    preset_tag = "copy_audio"
    preInputsettings = None
    postInputSettings = "-c:a copy"


class aac(AudioEncoder):
    preset_tag = "aac"
    preInputsettings = None
    postInputSettings = "-c:a aac"


class libmp3lame(AudioEncoder):
    preset_tag = "libmp3lame"
    preInputsettings = None
    postInputSettings = "-c:a libmp3lame"


# subtitle encoder options
class copySubtitles(SubtitleEncoder):
    preset_tag = "copy_subtitle"
    preInputsettings = None
    postInputSettings = "-c:s copy"


class srt(SubtitleEncoder):
    preset_tag = "srt"
    preInputsettings = None
    postInputSettings = "-c:s srt"


class ass(SubtitleEncoder):
    preset_tag = "ass"
    preInputsettings = None
    postInputSettings = "-c:s ass"


class webvtt(SubtitleEncoder):
    preset_tag = "webvtt"
    preInputsettings = None
    postInputSettings = "-c:s webvtt"


class libx264(VideoEncoder):
    preset_tag = "libx264"
    preInputsettings = None
    postInputSettings = "-c:v libx264"


class libx265(VideoEncoder):
    preset_tag = "libx265"
    preInputsettings = None
    postInputSettings = "-c:v libx265"


class vp9(VideoEncoder):
    preset_tag = "vp9"
    preInputsettings = None
    postInputSettings = "-c:v libvpx-vp9"
    qualityControlMode: str = "-cq:v"


class av1(VideoEncoder):
    preset_tag = "av1"
    preInputsettings = None
    postInputSettings = "-c:v libsvtav1"

class ffv1(VideoEncoder):
    preset_tag = "ffv1"
    preInputsettings = None
    postInputSettings = "-c:v ffv1"

class prores(VideoEncoder):
    preset_tag = "prores"
    preInputsettings = None
    postInputSettings = "-c:v prores_ks"


class x264_vulkan(VideoEncoder):
    preset_tag = "x264_vulkan"
    preInputsettings = "-init_hw_device vulkan=vkdev:0 -filter_hw_device vkdev"
    postInputSettings = "-filter:v format=nv12,hwupload -c:v h264_vulkan"
    # qualityControlMode: str = "-quality" # this is not implemented very well, quality ranges from 0-4 with little difference, so quality changing is disabled.


class x264_nvenc(VideoEncoder):
    preset_tag = "x264_nvenc"
    preInputsettings = "-hwaccel cuda -hwaccel_output_format cuda"
    postInputSettings = "-c:v h264_nvenc -preset slow"
    qualityControlMode: str = "-cq:v"


class x265_nvenc(VideoEncoder):
    preset_tag = "x265_nvenc"
    preInputsettings = "-hwaccel cuda -hwaccel_output_format cuda"
    postInputSettings = "-c:v hevc_nvenc -preset slow"
    qualityControlMode: str = "-cq:v"


class av1_nvenc(VideoEncoder):
    preset_tag = "av1_nvenc"
    preInputsettings = "-hwaccel cuda -hwaccel_output_format cuda"
    postInputSettings = "-c:v av1_nvenc -preset slow"
    qualityControlMode: str = "-cq:v"


class h264_vaapi(VideoEncoder):
    preset_tag = "x264_vaapi"
    preInputsettings = "-hwaccel vaapi -hwaccel_output_format vaapi"
    postInputSettings = "-rc_mode CQP -c:v h264_vaapi"
    qualityControlMode: str = "-qp"


class h265_vaapi(VideoEncoder):
    preset_tag = "x265_vaapi"
    preInputsettings = "-hwaccel vaapi -hwaccel_output_format vaapi"
    postInputSettings = "-rc_mode CQP -c:v hevc_vaapi"
    qualityControlMode: str = "-qp"


class av1_vaapi(VideoEncoder):
    preset_tag = "av1_vaapi"
    preInputsettings = "-hwaccel vaapi -hwaccel_output_format vaapi"
    postInputSettings = "-rc_mode CQP -c:v av1_vaapi"
    qualityControlMode: str = "-qp"


class FFMpegCommand:
    def __init__(self):
        self._settings = Settings()
        self._settings.readSettings()
        self._video_encoder = self._settings.settings['encoder']
        self._video_quality = self._settings.settings['video_quality']
        self.audio_encoder = self._settings.settings['audio_encoder']
        self.audio_bitrate = self._settings.settings['audio_bitrate']
        self.subtitle_encoder = self._settings.settings['subtitle_encoder']

    def build_command(self):
        self._command = []
        match self._video_encoder:
            case "libx264":
                self._command +=["-c:v","libx265"]
                match self._video_quality:
                    case "Very High":
                        self._command +=["-crf","15"]
                    case "High":
                        self._command +=["-crf","18"]
                    case "Medium":
                        self._command +=["-crf","23"]
                    case "Low":
                        self._command +=["-crf","28"]
            case "libx265":
                self._command +=["-c:v","libx265"]
                match self._video_quality:
                    case "Very High":
                        self._command +=["-crf","15"]
                    case "High":
                        self._command +=["-crf","18"]
                    case "Medium":
                        self._command +=["-crf","23"]
                    case "Low":
                        self._command +=["-crf","28"]
            case "vp9":
                self._command +=["-c:v","libvpx-vp9"]
                match self._video_quality:
                    case "Very High":
                        self._command +=["-cq:v","15"]
                    case "High":
                        self._command +=["-cq:v","18"]
                    case "Medium":
                        self._command +=["-cq:v","23"]
                    case "Low":
                        self._command +=["-cq:v","28"]
            case "av1":
                self._command +=["-c:v","libsvtav1"]
                match self._video_quality:
                    case "Very High":
                        self._command +=["-cq:v","15"]
                    case "High":
                        self._command +=["-cq:v","18"]
                    case "Medium":
                        self._command +=["-cq:v","23"]
                    case "Low":
                        self._command +=["-cq:v","28"]
            case "ffv1":
                self._command +=["-c:v","ffv1"]
                match self._video_quality:
                    case "Very High":
                        self._command +=["-level","3"]
                    case "High":
                        self._command +=["-level","4"]
                    case "Medium":
                        self._command +=["-level","5"]
                    case "Low":
                        self._command +=["-level","6"]
            case "prores":
                self._command +=["-c:v","prores_ks"]
                match self._video_quality:
                    case "Very High":
                        self._command +=["-profile:v","3"]
                    case "High":
                        self._command +=["-profile:v","2"]
                    case "Medium":
                        self._command +=["-profile:v","1"]
                    case "Low":
                        self._command +=["-profile:v","0"]
            case "x264_vulkan":
                self._command +=["-c:v","h264_vulkan"]
                self._command +=["-quality","0"]
            case "x264_nvenc":
                self._command +=["-c:v","h264_nvenc"]
                match self._video_quality:
                    case "Very High":
                        self._command +=["-cq:v","15"]
                    case "High":
                        self._command +=["-cq:v","18"]
                    case "Medium":
                        self._command +=["-cq:v","23"]
                    case "Low":
                        self._command +=["-cq:v","28"]
            case "x265_nvenc":
                self._command +=["-c:v","hevc_nvenc"]
                match self._video_quality:
                    case "Very High":
                        self._command +=["-cq:v","15"]
                    case "High":
                        self._command +=["-cq:v","18"]
                    case "Medium":
                        self._command +=["-cq:v","23"]
                    case "Low":
                        self._command +=["-cq:v","28"]
            case "av1_nvenc":
                self._command +=["-c:v","av1_nvenc"]
                match self._video_quality:
                    case "Very High":
                        self._command +=["-cq:v","15"]
                    case "High":
                        self._command +=["-cq:v","18"]
                    case "Medium":
                        self._command +=["-cq:v","23"]
                    case "Low":
                        self._command +=["-cq:v","28"]
            case "h264_vaapi":
                self._command +=["-c:v","h264_vaapi"]
                match self._video_quality:
                    case "Very High":
                        self._command +=["-crf","15"]
                    case "High":
                        self._command +=["-crf","18"]
                    case "Medium":
                        self._command +=["-crf","23"]
                    case "Low":
                        self._command +=["-crf","28"]
            case "h265_vaapi":
                self._command +=["-c:v","hevc_vaapi"]
                match self._video_quality:
                    case "Very High":
                        self._command +=["-crf","15"]
                    case "High":
                        self._command +=["-crf","18"]
                    case "Medium":
                        self._command +=["-crf","23"]
                    case "Low":
                        self._command +=["-crf","28"]
            case "av1_vaapi":
                self._command +=["-c:v","av1_vaapi"]
                match self._video_quality:
                    case "Very High":
                        self._command +=["-crf","15"]
                    case "High":
                        self._command +=["-crf","18"]
                    case "Medium":
                        self._command +=["-crf","23"]
                    case "Low":
                        self._command +=["-crf","28"]

            case _:
                self._command +=["-c:v","libx264"]
                match self._video_quality:
                    case "Very High":
                        self._command +=["-crf","15"]
                    case "High":
                        self._command +=["-crf","18"]
                    case "Medium":
                        self._command +=["-crf","23"]
                    case "Low":
                        self._command +=["-crf","28"]

        match self.audio_encoder:
            case "copy_audio":
                self._command +=["-c:a","copy"]
            case "aac":
                self._command +=["-c:a","aac"]
            case "libmp3lame":
                self._command +=["-c:a","libmp3lame"]
            case "opus":
                self._command +=["-c:a","libopus"]
            case _:
                self._command +=["-c:a","copy"]
        
        if self.audio_encoder != "copy_audio":
            self._command += ["-b:a",self.audio_bitrate]
        
        match self.subtitle_encoder:
            case "copy_subtitle":
                self._command +=["-c:s","copy"]
            case "srt":
                self._command +=["-c:s","srt"]
            case "ass":
                self._command +=["-c:s", "ass"]
            case "webvtt":
                self._command +=["-c:s", "webvtt"]

        return self._command

