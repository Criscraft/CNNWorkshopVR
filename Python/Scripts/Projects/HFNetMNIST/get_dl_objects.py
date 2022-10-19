import torch
import numpy as np
import os
from Projects.HFNetMNIST.THFNet import THFNet
from Projects.HFNetMNIST.TransformTestGS import TransformTestGS
from Projects.HFNetMNIST.TransformToTensor import TransformToTensor
from Projects.HFNetMNIST.MNIST import MNIST
from Scripts.DLData import DLData
from Scripts.DLNetwork import DLNetwork
from Scripts.NoiseGenerator import NoiseGenerator

N_CLASSES = 10
SEED = 42
IMAGE_SHAPE = (1, 28, 28)
NORM_MEAN = [0.13]
NORM_STD = [0.31]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")


def get_network():

    model = THFNet(
        n_classes=10,
        start_config={
            'k' : 3, 
            'filter_mode' : 'UnevenPosOnly',
            'n_angles' : 4,
            'n_channels_in' : 1,
            'n_channels_out' : 20, # muss teilbar durch shuffle_conv_groups sein sowie durch die Anzahl an Klassen 
            'f' : 4,
            'handcrafted_filters_require_grad' : False,
            'shuffle_conv_groups' : 1,
        },
        blockconfig_list=[
            {'k' : 3, 
            'filter_mode_1' : 'UnevenPosOnly',
            'filter_mode_2' : 'UnevenPosOnly',
            'n_angles' : 4,
            'n_blocks' : 2,
            'n_channels_in' : 20,
            'n_channels_out' : 20,
            'avgpool' : True if i>0 else False,
            'f' : 1,
            'handcrafted_filters_require_grad' : False,
            'shuffle_conv_groups' : 20 // 4,
            } for i in range(4)],
        avgpool_after_firstlayer=False,
        #statedict='hfnet_blocks_2_2_2_2_shallow.pt',
    )
        
    dl_network = DLNetwork(model, device, True, IMAGE_SHAPE, softmax=False)
    for param in model.embedded_model.parameters():
        param.requires_grad = False
    
    return dl_network


def get_dataset():

    transform_test = TransformTestGS(NORM_MEAN, NORM_STD)
    transform_to_tensor = TransformToTensor()

    dataset = MNIST(b_train=False, transform='transform_test', root=os.path.dirname(os.path.realpath(__file__)), download=False)
    dataset.prepare({'transform_test' : transform_test})

    dataset_to_tensor = MNIST(b_train=False, transform='transform_to_tensor', root=os.path.dirname(os.path.realpath(__file__)), download=False)
    dataset_to_tensor.prepare({'transform_to_tensor' : transform_to_tensor})

    np.random.seed(SEED)
    data_indices = None

    dldata = DLData(dataset, dataset_to_tensor, data_indices, dataset.class_names, N_CLASSES)
    
    return dldata


def get_noise_generator():
    noise_generator = NoiseGenerator(device, IMAGE_SHAPE, grayscale=True)
    return [noise_generator]