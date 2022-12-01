import torch
import numpy as np
import os
from Projects.HFNetMNIST.PFNetSimple import PFNetSimple
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

    model = PFNetSimple(
        n_classes=10,
        start_config={
            'n_channels_in' : 1,
            'filter_mode' : 'Uneven',
            'n_angles' : 2,
            'handcrafted_filters_require_grad' : False,
            'f' : 4,
            'k' : 3, 
            'stride' : 1,
        },
        blockconfig_list=[
            {'n_channels_in' : 16,
            'n_channels_out' : 16, # n_channels_out % shuffle_conv_groups == 0 and n_channels_out % n_classes == 0 
            'filter_mode' : 'Uneven',
            'n_angles' : 2,
            'f' : 1,
            'k' : 3, 
            'handcrafted_filters_require_grad' : False,
            'shuffle_conv_groups' : 16 // 4,
            'avgpool' : True if i>0 else False,
            } for i in range(3)],
        avgpool_after_firstlayer=False,
        #init_mode = 'zero',
        activation='relu',
        statedict=os.path.join('..', 'Projects', 'HFNetMNIST', 'model_mnist_pfnet_simple.pt'),
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
    
    return dldata, transform_test


def get_noise_generator():
    noise_generator = NoiseGenerator(device, IMAGE_SHAPE, grayscale=True)
    return noise_generator