# CNNWorkshopCNN

This application has no documentation, yet. The goal is to create some kind of editor for CNNs. 

## Set up Python
The software requires Python because the deep learing part is implemented in Pytorch. While you can use a virtual environment for Python I found it easier to have a local Python installation in the Project that can be easily deployed. The local Python installation should be located at Python/PythonInstallation

Recommended procedure to install Python:

First install Miniconda [here](https://docs.conda.io/en/latest/miniconda.html)

Open Miniconda (Windows) or a new terminal (Linux). 
Move to the Python/PythonInstallation folder using the cd command. Then type
    conda create -p . python=3.9.16
Activate the environment using
    conda activate ./
Follow the instructions and proceed until Python is installed.
Important: The python executable has to be located directly in Python/PythonInstallation
Upgrade pip.
    pip install --upgrade pip
Add Script directory to path:
    conda install conda-build
    conda develop PathToYourProjectFolder/CNNWorkshopVR/Python/Scripts/Scripts
Then, install the required packages. If you desire GPU support, visit https://pytorch.org/ to get the correct extra-index-url for your system and install cuda https://developer.nvidia.com/cuda-downloads.
    conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia
    pip install torchgeometry
    conda install -c conda-forge cudnn
    conda install matplotlib 
    conda install -c conda-forge websockets scikit-learn
    pip install opencv-python-headless

Check if pytorch has GPU access:
    python
    import torch
    torch.cuda.is_available()

You should get True.
    
The new Python installation has to know where the Python scripts for this application are located. In the Lib/site-packages directory of the local Python installation add a new text file "mypath.pth". Write a single line into the file that contains the path to the Scripts folder. For example:
PathToYourProjectFolder/CNNWorkshopVR/Python/Scripts

## Start application

To start the application, first start the server
    conda activate PathToYourProjectFolder/CNNWorkshopVR/Python/PythonInstallation
    python PathToYourProjectFolder/CNNWorkshopVR/Python/Scripts/Scripts/DLWebServer.py
Then you can start the Godot application.