from abc import  abstractmethod
import os
import subprocess
import sys
import os
import subprocess
import sys
import shutil

import urllib.request

PLATFORM = sys.platform


def zero_mainwindow_size():
    import xml.etree.ElementTree as ET

    def set_mainwindow_size_zero(path="testRVEInterface.ui"):
        tree = ET.parse(path)
        root = tree.getroot()

        geometry = root.find('.//property[@name="geometry"]/rect')
        if geometry is not None:
            width = geometry.find("width")
            height = geometry.find("height")
            if width is not None:
                width.text = "0"
            if height is not None:
                height.text = "0"
            tree.write(path)

    set_mainwindow_size_zero()

class PythonManager:

    PYTHON_VENV_PATH = "venv\\Scripts\\python.exe" if PLATFORM == "win32" else "venv/bin/python3"
    PYTHON_SYSTEM_EXECUTABLE = "python3"

    def setup_python(self):
        self.__create_venv()
        self.__install_pip_in_venv()
        self.__install_requirements_in_venv()

    def __create_venv(self):
        print("Creating virtual environment")
        command = [self.PYTHON_SYSTEM_EXECUTABLE, "-m", "venv", "venv"]
        subprocess.run(command)


    def __install_pip_in_venv(self):
        command = [
            self.PYTHON_VENV_PATH,
            "-m",
            "ensurepip",
        ]
        subprocess.run(command)

    def __install_requirements_in_venv(self):
        print("Installing requirements in virtual environment")
        if not os.path.isfile("requirements.txt"):
            raise FileNotFoundError("No requirements.txt in current directory!")
        command = [
            self.PYTHON_VENV_PATH,
            "-m",
            "pip",
            "install",
            "-r",
            "requirements.txt",
        ]

        subprocess.run(command)
        

    def get_venv_site_packages(self):
        command = [
            self.PYTHON_VENV_PATH,
            "-c",
            'import site; print("\\n".join(site.getsitepackages()))',
        ]
        result = subprocess.run(command, stdout=subprocess.PIPE, text=True)
        site_packages = result.stdout.strip().split('\n')[0]
        if os.path.exists(site_packages):
            return site_packages
        site_packages = site_packages.replace('dist','site')
        if os.path.exists(site_packages):
            return site_packages
        print(site_packages)
        raise FileNotFoundError("Unable to locate site packages for python venv!")
    


class BuildManager:

    @abstractmethod
    def build(self):
        ...
    
    def download_file(self, url, destination):
        print(f"Downloading file from {url}")
        urllib.request.urlretrieve(url, destination)
        print("File downloaded successfully")

    

    def build_gui(self):
        print("Building GUI")
        zero_mainwindow_size()
        if PLATFORM == "darwin" or PLATFORM == "linux":
            os.system(
                f"{self.get_site_packages()}/PySide6/Qt/libexec/uic -g python testRVEInterface.ui > mainwindow.py"
            )
        if PLATFORM == "win32":
            os.system(
                r".\venv\Lib\site-packages\PySide6\uic.exe -g python testRVEInterface.ui > mainwindow.py"
            )
    
    def build_resources(self):
        print("Building resources.rc")
        if PLATFORM == "darwin" or PLATFORM == "linux":
            os.system(
                f"{self.get_site_packages()}/PySide6/Qt/libexec/rcc -g python resources.qrc > resources_rc.py"
            )
        if PLATFORM == "win32":
            os.system(
                r".\venv\Lib\site-packages\PySide6\rcc.exe -g python resources.qrc > resources_rc.py"
            )


class PyInstaller(BuildManager):
    def build(self):

        ...













def build_executable(dist_dir=None):
    print("Building executable")
    if PLATFORM == "win32" or PLATFORM == "darwin":
        if dist_dir is None:
            dist_dir = "dist"
        command = [
            python_path(),
            "-m",
            "PyInstaller",
            "REAL-Video-Enhancer.py",
            "--icon=icons/logo-v2.ico",
            "--noconfirm",
            "--noupx",
            "--distpath",
            dist_dir,
            # "--noconsole", this caused issues, maybe I can fix it later
        ]
    else:
        if dist_dir is None:
            dist_dir = "bin"
        command = [
            python_path(),
            "-m",
            "cx_Freeze",
            "REAL-Video-Enhancer.py",
            "--target-dir",
            dist_dir,
        ]
    subprocess.run(command)


def copy_backend(build_dir=None):
    print("Copying backend")
    if PLATFORM == "win32":
        if build_dir is None:
            build_dir = "dist"
        try:
            os.system(f"cp -r backend {build_dir}/REAL-Video-Enhancer/backend")
        except Exception:
            pass
        if not os.path.exists(rf"{build_dir}\\REAL-Video-Enhancer\\backend"):
            os.system(
                f'xcopy "./backend" "./{build_dir}/REAL-Video-Enhancer/backend" /E /I'
            )
    if PLATFORM == "linux":
        if build_dir is None:
            build_dir = "bin"
        os.system(f"cp -r backend {build_dir}/")



def build_venv():
    create_venv()
    install_pip_in_venv()
    install_requirements_in_venv()


if len(sys.argv) > 1:
    if sys.argv[1] == "--create_venv" or sys.argv[1] == "--build_exe":
        build_venv()

if not os.path.exists("venv"):
    build_venv()

build_gui()
build_resources()

if "--build_dir_override" in sys.argv:
    build_dir = sys.argv[sys.argv.index("--build_dir_override") + 1]
    build_executable(build_dir)
    # copy_backend(build_dir=build_dir)
if "--build_exe" in sys.argv and "--build_dir_override" not in sys.argv:
    build_executable()
    # copy_backend()
