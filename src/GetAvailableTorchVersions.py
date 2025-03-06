import requests
try:
    from .constants import HAS_NETWORK_ON_STARTUP
except ImportError:
    from constants import HAS_NETWORK_ON_STARTUP
from dataclasses import dataclass

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
        torch_stable = "2.6.0"
        nightly_date = "20250304"
        torch_nightly = "2.7.0.dev{nightly_date}".format(nightly_date=nightly_date)
        
        self.cuda_versions = ["cu126", "cu128"]
        self.rocm_versions = ["rocm6.2.4", "rocm6.3"]
        self.xpu_versions = ["xpu"]
        self.torch_versions = ["2.6.0", f"{torch_nightly}"]
        self.torch_to_torchvision_versions = {"2.6.0": "0.21.0", f"{torch_nightly}": "0.22.0.dev{nightly_date}".format(nightly_date=nightly_date)}
        """
        torch cuda 2.6 -> cuda 12.6
        torch cuda 2.7 -> cuda 12.8
        torch rocm 2.6 -> rocm 6.2.4
        torch rocm 2.7 -> rocm 6.3
        torch xpu 2.6 -> xpu
        torch xpu 2.7 -> xpu
        """
        self.torch_to_cuda_versions = {"2.6.0": "cu126", f"{torch_nightly}": "cu128"}
        self.torch_to_rocm_versions = {"2.6.0": "rocm6.2.4", f"{torch_nightly}": "rocm6.3"}
        self.torch_to_xpu_versions = {"2.6.0": "xpu", f"{torch_nightly}": "xpu"}

        self.stable_url = 'https://download.pytorch.org/whl/'
        self.stable_torch = 'https://download.pytorch.org/whl/torch'
        self.nightly_url = 'https://download.pytorch.org/whl/nightly/'
        
    
    def get_torch_nightly_versions(depth=10):
        pass

    def get_latest_torch_version_nigthly(self):
        text = requests.get(self.nightly_url).text
    
    



if __name__ == "__main__":
    print(TorchScraper().get_updated_torch_versions())