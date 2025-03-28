import os
from PySide6.QtWidgets import QMainWindow
from .QTcustom import RegularQTPopup, NetworkCheckPopup, addNotificationToButton, disable_combobox_item_by_text
from ..DownloadDeps import DownloadDependencies
from ..DownloadModels import DownloadModel
from ..ModelHandler import (
    ncnnInterpolateModels,
    pytorchInterpolateModels,
    ncnnUpscaleModels,
    pytorchUpscaleModels,
)
from PySide6.QtWidgets import QMessageBox
from .Updater import ApplicationUpdater
from ..constants import IS_FLATPAK, MODELS_PATH, PLATFORM, CWD, USE_LOCAL_BACKEND
from ..GetAvailableTorchVersions import TorchScraper
from ..Util import FileHandler

def downloadModelsBasedOnInstalledBackend(installed_backends: list):
    if NetworkCheckPopup():
        for backend in installed_backends:
            match backend:
                case "ncnn":
                    for model in ncnnInterpolateModels:
                        DownloadModel(
                            model, ncnnInterpolateModels[model][1], MODELS_PATH
                        )
                    for model in ncnnUpscaleModels:
                        DownloadModel(model, ncnnUpscaleModels[model][1], MODELS_PATH)
                case "pytorch":  # no need for tensorrt as it uses pytorch models
                    for model in pytorchInterpolateModels:
                        DownloadModel(
                            model, pytorchInterpolateModels[model][1], MODELS_PATH
                        )
                    for model in pytorchUpscaleModels:
                        DownloadModel(
                            model, pytorchUpscaleModels[model][1], MODELS_PATH
                        )


class DownloadTab:
    def __init__(
        self,
        parent: QMainWindow,
        backends: list,
    ):
        self.parent = parent
        self.downloadDeps = DownloadDependencies()
        self.backends = backends
        self.applicationUpdater = ApplicationUpdater()

        # set this all to not visible, as scrapping the idea for now.
        if PLATFORM != "linux":
            disable_combobox_item_by_text(self.parent.pytorch_backend, "ROCm (Linux Only)")
        if IS_FLATPAK or USE_LOCAL_BACKEND:
            self.parent.uninstallAppBtn.setDisabled(True)
        else:
            self.parent.uninstallAppBtn.clicked.connect(self.uninstallApp)

        self.parent.ApplicationUpdateContainer.setVisible(False)
        self.QButtonConnect()

    def QButtonConnect(self):
        self.parent.downloadNCNNBtn.clicked.connect(lambda: self.download("ncnn", True))
        self.parent.downloadTorchBtn.clicked.connect(
            lambda: self.download("torch", True)
        )
        self.parent.downloadTensorRTBtn.clicked.connect(
            lambda: self.download("tensorrt", True)
        )
        self.parent.downloadDirectMLBtn.clicked.connect(
            lambda: self.download("directml", True)
        )
        self.parent.downloadAllModelsBtn.clicked.connect(
            lambda: downloadModelsBasedOnInstalledBackend(
                ["ncnn", "pytorch", "tensorrt", "directml"]
            )
        )
        self.parent.downloadSomeModelsBasedOnInstalledBackendbtn.clicked.connect(
            lambda: downloadModelsBasedOnInstalledBackend(self.backends)
        )
        self.parent.uninstallNCNNBtn.clicked.connect(
            lambda: self.download("ncnn", False)
        )
        self.parent.uninstallTorchBtn.clicked.connect(
            lambda: self.download("torch", False)
        )
        self.parent.uninstallTensorRTBtn.clicked.connect(
            lambda: self.download("tensorrt", False)
        )
        self.parent.uninstallDirectMLBtn.clicked.connect(
            lambda: self.download("directml", False)
        )
        self.parent.selectPytorchCustomModel.clicked.connect(
            lambda: self.parent.importCustomModel("pytorch")
        )
        self.parent.selectNCNNCustomModel.clicked.connect(
            lambda: self.parent.importCustomModel("ncnn")
        )

    def uninstallApp(self):
        reply = QMessageBox.question(
            self,
            "",
            "Are you sure you want to uninstall?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,  # type: ignore
        )
        if reply == QMessageBox.Yes:  # type: ignore
            FileHandler().removeFolder(CWD)
            os._exit(0)
        

    def download(self, dep, install: bool = True):
        """
        Downloads the specified dependency.
        Parameters:
        - dep (str): The name of the dependency to download.
        Returns:
        - None
        """
        pytorch_ver = self.parent.pytorch_version.currentText().split()[0]
        pytorch_backend = self.parent.pytorch_backend.currentText().split()[0]
        torchvision_ver = TorchScraper().torchvision_version

        if dep.lower() == "pytorch" or dep.lower() == "tensorrt":
            
            pytorch_backend = TorchScraper().cuda_version
        elif pytorch_backend.lower() == "rocm":
            pytorch_backend = TorchScraper().rocm_version
        else:
            pytorch_backend = TorchScraper().xpu_version
        
        if NetworkCheckPopup(
            "https://pypi.org/"
        ):  # check for network before installing
            return_code = self.downloadDeps.downloadPythonDeps(dep, pytorch_ver, torchvision_ver, pytorch_backend, install)
            if return_code == 0:
                RegularQTPopup(
                    "Download Complete\nPlease restart the application to apply changes."
                )
            else:
                RegularQTPopup("Download Failed!\nPlease check logs for more info.")
