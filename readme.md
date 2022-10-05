# CNNWorkshopCNN

This application has no documentation, yet. The goal is to create some kind of editor for CNNs. 

## Set up Python
The software requires a local Python installation in Python/PythonInstallation

Recommended procedure to install Python:

First install conda. Windows users can install conda [here](https://docs.conda.io/en/latest/miniconda.html)

Open Miniconda (Windows) or a new terminal (Linux). 
Move to the Python/PythonInstallation folder using the cd command. Then type
    conda create -p . python=3.8
Follow the instructions and proceed until Python is installed.
Important: The python executable has to be located directly in Python\PythonInstallation
Upgrade pip.
    .\python.exe -m pip install --upgrade pip
Then, install the required packages. If you desire GPU support, visit https://pytorch.org/ to get the correct extra-index-url for your system and install cuda https://developer.nvidia.com/cuda-downloads.
    .\python.exe -m pip install torch torchvision torchgeometry opencv-python websockets --extra-index-url https://download.pytorch.org/whl/cu116 
