from .ui.SettingsTab import Settings

class FFMpegCommand:
    def __init__(self, hdr_mode):
        self._settings = Settings()
        self._settings.readSettings()
        self._video_encoder = self._settings.settings['encoder']
        self._video_quality = self._settings.settings['video_quality']
        self._video_pixel_format = self._settings.settings['video_pixel_format']

        self.audio_encoder = self._settings.settings['audio_encoder']
        self.audio_bitrate = self._settings.settings['audio_bitrate']
        self.subtitle_encoder = self._settings.settings['subtitle_encoder']
        self.hdr_mode = hdr_mode

    def build_command(self):
        self._command = []
        match self._video_encoder:
            case "libx264":
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
                if self.hdr_mode:
                    self._command +=["-x264-params","colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"]
                
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
                if self.hdr_mode:
                    self._command +=["-x265-params","colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:master-display=G(13250,34500)B(7500,3000)R(34000,16000)WP(15635,16450)L(10000000,1)"]
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
        
        self._command += ["-pix_fmt",self._video_pixel_format]

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

