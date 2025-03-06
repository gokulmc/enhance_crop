import requests
try:
    from .constants import HAS_NETWORK_ON_STARTUP
except ImportError:
    from constants import HAS_NETWORK_ON_STARTUP
from dataclasses import dataclass
from .ui.SettingsTab import Settings
@dataclass
class TorchVersion:
    torch_version: str
    backend_version: str
    
    def __str__(self):
        return f"torch-{self.torch_version}+{self.backend_version}"
    
    @property
    def is_nightly(self):
        return ".dev" in self.torch_version


class TorchScraper:
    def __init__(self):
        settings = Settings()
        torch_version = settings.settings["pytorch_version"].split()[0] # has to be in the format "2.6.0" or "2.7.0.dev20220301"
        nightly = "dev" in torch_version
        if nightly:
            nightly_date = torch_version.split(".dev")[1]
        

        self.torchvision_version = "0.21.0"
        self.cuda_version = "cu126"
        self.rocm_version = "rocm6.2.4"
        self.xpu_version = "xpu"
        if nightly:
            self.torchvision_version = "0.22.0.dev" + nightly_date
            self.cuda_version = "cu128"
            self.rocm_version = "rocm6.3"
            self.xpu_version = "xpu"
        """
        torch cuda 2.6 -> cuda 12.6
        torch cuda 2.7 -> cuda 12.8
        torch rocm 2.6 -> rocm 6.2.4
        torch rocm 2.7 -> rocm 6.3
        torch xpu 2.6 -> xpu
        torch xpu 2.7 -> xpu
        """
        

        self.stable_url = 'https://download.pytorch.org/whl/'
        self.stable_torch = 'https://download.pytorch.org/whl/torch'
        self.nightly_url = 'https://download.pytorch.org/whl/nightly/'
        
    
    def get_torch_nightly_versions(depth=10):
        pass

    def get_latest_torch_version_nigthly(self):
        text = requests.get(self.nightly_url).text
    
    



if __name__ == "__main__":
    print(TorchScraper().get_updated_torch_versions())