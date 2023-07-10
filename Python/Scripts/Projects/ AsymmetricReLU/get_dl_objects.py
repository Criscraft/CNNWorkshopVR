import torch
from torchvision import transforms
import numpy as np
import os
from Projects.AssymetricReLU.ReLUNet import ReLUNet
from Projects.AssymetricReLU.custom_images import CustomImages
from Scripts.DLData import DLData
from Scripts.DLNetwork import DLNetwork
from Scripts.NoiseGeneratorLight import NoiseGenerator

SEED = 42
IMAGE_SHAPE = (1, 15, 15)
NORM_MEAN = [0.]
NORM_STD = [1.]
N_CLASSES = 1
use_cuda = torch.cuda.is_available()
device = torch.device("cuda") if use_cuda else torch.device("cpu")


def get_network(report_feature_visualization_results):
    model = ReLUNet(
        activation="relu"
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

    dataset = CustomImages()
    dataset.prepare({'transform_test' : transform_test})

    dataset_test_norm = CustomImages()
    dataset_test_norm.prepare({'transform_test_norm' : transform_test_norm})

    np.random.seed(SEED)
    data_indices = []

    dldata = DLData(dataset, data_indices, dataset.class_names, N_CLASSES)

    loader_test_norm = torch.utils.data.DataLoader(dataset_test_norm, batch_size=64)
    
    return dldata, transform_norm, loader_test_norm


def get_noise_generator():
    noise_generator = NoiseGenerator(device, IMAGE_SHAPE)
    return noise_generator