import requests
try:
    from .constants import HAS_NETWORK_ON_STARTUP
except ImportError:
    from constants import HAS_NETWORK_ON_STARTUP

class TorchScraper:
    def __init__(self):
        self.cuda_versions = ["cu126", "cu128"]
        self.rocm_versions = ["rocm6.2.4", "rocm6.3"]
        self.xpu_versions = ["xpu"]
        self.torch_versions = ["2.6.0", "2.7.0.dev"]

        self.stable_url = 'https://download.pytorch.org/whl/'
        self.stable_torch = 'https://download.pytorch.org/whl/torch'
        self.nightly_url = 'https://download.pytorch.org/whl/nightly/'

    
    def get_latest_torch_version_nigthly(self):
        text = requests.get(self.nightly_url).text
    
    def get_latest_torch_stable(self):
        ...
    
    def get_latest_torch_cuda_stable(self):
        ...

    def get_latest_torch_cuda_nightly(self):
        ...
    
    def get_latest_torch_rocm_stable(self):
        ...
    
    def get_latest_torch_rocm_nightly(self):
        ...
    
    



if __name__ == "__main__":
    print(TorchScraper().get_updated_torch_versions())