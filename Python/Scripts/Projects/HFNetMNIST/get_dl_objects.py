import torch
from torchvision import transforms
import numpy as np
import os
from Projects.HFNetMNIST.TranslationNetMNIST import TranslationNetMNIST
from Projects.HFNetMNIST.MNIST import MNIST
from Scripts.DLData import DLData
from Scripts.DLNetwork import DLNetwork
from Scripts.GrayGenerator import GrayGenerator
from Scripts.NoiseGenerator import NoiseGenerator

N_CLASSES = 10
SEED = 42
IMAGE_SHAPE = (1, 28, 28)
NORM_MEAN = [0.]
NORM_STD = [1.]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")


def get_network(report_feature_visualization_results):
    n_channels_list = [1*8, 2*8, 2*8, 2*8, 3*8, 5*8]
    
    # model = TranslationNetMNIST(
    #     n_classes=10,
    #     blockconfig_list=[
    #         {'n_channels_in' : 1 if i==0 else n_channels_list[i-1],
    #         'n_channels_out' : n_channels_list[i], # n_channels_out % shuffle_conv_groups == 0
    #         'conv_groups' : n_channels_list[i] // 8,
    #         'pool_mode' : "avgpool" if i in [3, 5] else "",
    #         'spatial_mode' : "predefined_filters", # one of predefined_filters and parameterized_translation
    #         'spatial_requires_grad' : False,
    #         'filter_mode' : "TranslationSharp8", # one of Even, Uneven, All, Random, Smooth, EvenPosOnly, UnevenPosOnly, TranslationSmooth, TranslationSharp4, TranslationSharp8
    #         'n_angles' : 4,
    #         'translation_k' : 5,
    #         'randomroll' : -1,
    #         'normalization_mode' : 'layernorm', # one of batchnorm, layernorm
    #         'permutation' : 'shifted', # one of shifted, identity, disabled
    #         } for i in range(6)],
    #     init_mode='uniform', # one of uniform, uniform_translation_as_pfm, zero, identity, kaiming
    #     pool_mode="avgpool",
    #     #conv_expressions=["digits_A", "digits_B", "big_curves", "curves", "big_corners"],
    #     conv_expressions_path = "conv_expressions_8_filters.txt",
    #     statedict=os.path.join('..', 'Projects', 'HFNetMNIST', 'avgpool_all_layers_trained.pt'),
    # )

    model = TranslationNetMNIST(
        n_classes=10,
        blockconfig_list=[
            {'n_channels_in' : 1 if i==0 else n_channels_list[i-1],
            'n_channels_out' : n_channels_list[i], # n_channels_out % shuffle_conv_groups == 0
            'conv_groups' : n_channels_list[i] // 8,
            'pool_mode' : "avgpool" if i in [3, 5] else "",
            'spatial_mode' : "predefined_filters", # one of predefined_filters and parameterized_translation
            'spatial_requires_grad' : False,
            'filter_mode' : "TranslationSharp8", # one of Even, Uneven, All, Random, Smooth, EvenPosOnly, UnevenPosOnly, TranslationSmooth, TranslationSharp4, TranslationSharp8
            'n_angles' : 4,
            'translation_k' : 5,
            'randomroll' : -1,
            'normalization_mode' : 'layernorm', # one of batchnorm, layernorm
            'permutation' : 'shifted', # one of shifted, identity, disabled
            } for i in range(6)],
        init_mode='uniform', # one of uniform, uniform_translation_as_pfm, zero, identity, kaiming
        pool_mode="avgpool",
        conv_expressions=["digits_A", "digits_B", "big_curves", "curves", "big_corners"],
        conv_expressions_path = "conv_expressions_8_filters.txt",
        #statedict=os.path.join('..', 'Projects', 'HFNetMNIST', 'lppool_block_6_linear_layer_trained.pt'),
    )
    
    dl_network = DLNetwork(model, device, True, IMAGE_SHAPE, report_feature_visualization_results, softmax=False, norm_mean=NORM_MEAN, norm_std=NORM_STD)
    
    return dl_network


def get_dataset():

    transform_norm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])
    transform_test = transforms.Compose([
        transforms.Resize(IMAGE_SHAPE[1:]),
        transforms.ToTensor()
    ])
    transform_test_norm = transforms.Compose([
            transforms.Resize(IMAGE_SHAPE[1:]),
            transforms.ToTensor(),
            transforms.Normalize(mean=NORM_MEAN, std=NORM_STD),
    ])

    dataset = MNIST(
        b_train=False, 
        transform='transform_test', 
        root=os.path.dirname(os.path.realpath(__file__)), 
        download=False,
    )
    dataset.prepare({'transform_test' : transform_test})

    dataset_test_norm = MNIST(
        b_train=False, 
        transform='transform_test_norm', 
        root=os.path.dirname(os.path.realpath(__file__)), 
        download=False,
    )
    dataset_test_norm.prepare({'transform_test_norm' : transform_test_norm})

    np.random.seed(SEED)
    data_indices = []

    dldata = DLData(dataset, data_indices, dataset.class_names, N_CLASSES)

    loader_test_norm = torch.utils.data.DataLoader(dataset_test_norm, batch_size=64)
    
    return dldata, transform_norm, loader_test_norm


def get_noise_generator():
    noise_generator = NoiseGenerator(device, IMAGE_SHAPE)
    return noise_generator