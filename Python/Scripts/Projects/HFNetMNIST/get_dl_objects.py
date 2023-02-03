import torch
import numpy as np
import os
from Projects.HFNetMNIST.TranslationNet import TranslationNet
from Projects.HFNetMNIST.TransformTestGS import TransformTestGS
from Projects.HFNetMNIST.TransformToTensor import TransformToTensor
from Projects.HFNetMNIST.MNIST import MNIST
from Scripts.DLData import DLData
from Scripts.DLNetwork import DLNetwork
from Scripts.NoiseGenerator import NoiseGenerator

N_CLASSES = 10
SEED = 42
IMAGE_SHAPE = (1, 28, 28)
NORM_MEAN = [0.]
NORM_STD = [1.]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")


def get_network():

    model = TranslationNet(
        n_classes=10,
        blockconfig_list=[
            {'n_channels_in' : 1 if i==0 else 6*8,
            'n_channels_out' : 6*8, # n_channels_out % shuffle_conv_groups == 0
            'conv_groups' : 6*8 // 8,
            'pool_mode' : "avgpool" if i in [3, 6] else "",
            'spatial_mode' : "predefined_filters", # one of predefined_filters and parameterized_translation
            'spatial_requires_grad' : False,
            'filter_mode' : "TranslationSharp8", # one of Even, Uneven, All, Random, Smooth, EvenPosOnly, UnevenPosOnly, TranslationSmooth, TranslationSharp4, TranslationSharp8
            'n_angles' : 4,
            'translation_k' : 5,
            'randomroll' : -1,
            'normalization_mode' : 'layernorm', # one of batchnorm, layernorm
            'permutation' : 'identity', # one of shifted, identity, disabled
            } for i in range(9)],
        init_mode='zero', # one of uniform, uniform_translation_as_pfm, zero, identity
        conv_expressions = ["digit_parts_1_2_5", "digit_parts_6_8_9", "x_longstr_st1", "x_longstr_st2", "longstr_edges_st1", "longstr_edges_diag_st1"],
        #statedict=os.path.join('..', 'Projects', 'HFNetMNIST', 'model_mnist_translationnet_predefined_filters_translation_groupedconv_scalereg.pt'),
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