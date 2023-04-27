import os
import torch
from torchvision import transforms
import numpy as np
from Projects.Flowers102.TranslationResNet import TranslationResNet
from Projects.Flowers102.Flowers102 import Flowers102
from Scripts.DLData import DLData
from Scripts.DLNetwork import DLNetwork
from Scripts.NoiseGenerator import NoiseGenerator

N_CLASSES = 102
SEED = 42
IMAGE_SHAPE = (3, 224, 224)
NORM_MEAN = [0.43553507, 0.3777357, 0.28795698]
NORM_STD = [0.26542637, 0.21253887, 0.21966127]

#NORM_MEAN = [0., 0., 0.]
#NORM_STD = [1., 1., 1.]
use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")


def get_network():
    
    n_channels_list = [
        64, 64, # pool stage 1
        128, 128, # pool stage 2
        256, 256, # pool stage 3
        512, 512, # pool stage 4
    ]

    model = TranslationResNet(
        n_classes=102,
        first_block_config={
            'spatial_mode' : "predefined_filters", # one of predefined_filters and dense_convolution
            'n_channels_in' : 3,
            'n_channels_out' : 64,
            'k' : 3,
            'stride' : 2,
            'padding': 1,
            'filter_mode' : "EvenAndUneven", # for predefined_filters only
            'n_angles' : 4, # for predefined_filters only
            'filters_require_grad' : False, # for predefined_filters only
            'f' : 16, # for predefined_filters only
        },
        blockconfig_list=[
            {'n_channels_in' : 64 if i==0 else n_channels_list[i-1],
            'n_channels_out' : n_channels_list[i],
            'conv_groups' : 1,
            'pool_mode' : "",
            'spatial_mode' : "predefined_filters", # one of predefined_filters and dense_convolution
            'parameterized_translation' : False,
            'filters_require_grad' : False, # for predefined_filters only
            'filter_mode' : "EvenAndUneven", # for predefined_filters only; one of Even, Uneven, EvenAndUneven, Random, Smooth, EvenPosOnly, UnevenPosOnly, TranslationSmooth, TranslationSharp4, TranslationSharp8
            'n_angles' : 4, # for predefined_filters only
            'k' : 3,
            'translation_k' : 5, # for parameterized_translation only
            'randomroll' : 0,
            'stride' : 2 if i in [2, 4, 6] else 1,
            } for i in range(8)],
        init_mode='kaiming', # one of uniform, zero, identity, kaiming
        first_pool_mode="maxpool", # one of maxpool, avgpool, lppool
        global_pool_mode="avgpool", # one of maxpool, avgpool, lppool
        permutation_mode='disabled', # one of shifted, identity, disabled
        statedict=os.path.join('..', 'Projects', 'Flowers102', 'pfnet18.pt'),
    )

    # model = TranslationResNet(
    #     n_classes=102,
    #     first_block_config={
    #         'spatial_mode' : "dense_convolution", # one of predefined_filters and dense_convolution
    #         'n_channels_in' : 3,
    #         'n_channels_out' : 64,
    #         'k' : 7,
    #         'stride' : 2,
    #         'padding': 3,
    #         #'filter_mode' : "EvenAndUneven", # for predefined_filters only
    #         #'n_angles' : 4, # for predefined_filters only
    #         #'filters_require_grad' : False, # for predefined_filters only
    #         #'f' : 16, # for predefined_filters only
    #     },
    #     blockconfig_list=[
    #         {'n_channels_in' : 64 if i==0 else n_channels_list[i-1],
    #         'n_channels_out' : n_channels_list[i],
    #         'conv_groups' : 1,
    #         'pool_mode' : "",
    #         'spatial_mode' : "dense_convolution", # one of predefined_filters and dense_convolution
    #         'parameterized_translation' : False,
    #         'filters_require_grad' : False, # for predefined_filters only
    #         'filter_mode' : "EvenAndUneven", # for predefined_filters only; one of Even, Uneven, EvenAndUneven, Random, Smooth, EvenPosOnly, UnevenPosOnly, TranslationSmooth, TranslationSharp4, TranslationSharp8
    #         'n_angles' : 4, # for predefined_filters only
    #         'k' : 3,
    #         'translation_k' : 5, # for parameterized_translation only
    #         'randomroll' : 0,
    #         'stride' : 2 if i in [2, 4, 6] else 1,
    #         } for i in range(8)],
    #     init_mode='kaiming', # one of uniform, zero, identity, kaiming
    #     first_pool_mode="maxpool", # one of maxpool, avgpool, lppool
    #     global_pool_mode="avgpool", # one of maxpool, avgpool, lppool
    #     permutation_mode='disabled', # one of shifted, identity, disabled
    #     statedict=os.path.join('..', 'Projects', 'Flowers102', 'resnet18.pt'),
    # )
        
    dl_network = DLNetwork(model, device, True, IMAGE_SHAPE, softmax=False)
    for param in model.embedded_model.parameters():
        param.requires_grad = False
    
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

    dataset = Flowers102(
        datapath=os.path.join("..", "..", "Datasets", "Flowers102", "flowers102"),
        path_to_splits=os.path.join("..", "..", "Datasets", "Flowers102", "setid.mat"),
        path_to_labels=os.path.join("..", "..", "Datasets", "Flowers102", "imagelabels.mat"),
        mode='test',
        transform='transform_test',
    )
    dataset.prepare({'transform_test' : transform_test})

    dataset_test_norm = Flowers102(
        datapath=os.path.join("..", "..", "Datasets", "Flowers102", "flowers102"),
        path_to_splits=os.path.join("..", "..", "Datasets", "Flowers102", "setid.mat"),
        path_to_labels=os.path.join("..", "..", "Datasets", "Flowers102", "imagelabels.mat"),
        mode='test',
        transform='transform_test_norm',
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