import os
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms
import torchgeometry as tgm
import numpy as np
import random
from enum import Enum


BATCHSIZE = 32
imagecount = 0


class FeatureVisualizationParams(object):
    
    class Mode(str, Enum):
        AVERAGE = "AVERAGE"
        CENTERPIXEL = "CENTERPIXEL"
        PERCENTILE = "PERCENTILE"
        
    def __init__(self,
        target_size=(28, 28),
        mode=Mode.CENTERPIXEL,
        epochs=200,
        epochs_without_robustness_transforms=10,
        lr=20.,
        degrees=10,
        blur_sigma=0.5,
        roll=4,
        fraction_to_maximize=0.25,
        ):

        self.target_size = target_size
        self.mode = mode
        self.epochs = epochs
        self.epochs_without_robustness_transforms = epochs_without_robustness_transforms
        self.lr = lr
        self.degrees = degrees
        self.blur_sigma = blur_sigma
        self.roll = roll
        self.fraction_to_maximize = fraction_to_maximize


class FeatureVisualizer(object):
    
    def __init__(self,
        export = False,
        export_path = '',
        export_interval=50,
        fv_settings=FeatureVisualizationParams()):
        
        super().__init__()
        self.export = export
        self.export_interval = export_interval
        self.export_path = export_path
        self.fv_settings = fv_settings

        self.regularizer = self.create_regularizer()
        self.export_transformation = ExportTransform()


    def visualize(self, model, module, device, init_image, n_channels, channels=None):
        global imagecount
        export_meta = []
        if channels is None:
            channels = np.arange(n_channels)
        
        init_image = init_image.to(device)
        if init_image.ndim==3:
            init_image = init_image.unsqueeze(0)
        
        n_batches = int( np.ceil( n_channels / float(BATCHSIZE) ) )
        
        print("start gradient ascent on images")

        created_image_aggregate = []
        for batchid in range(n_batches):
            channels_batch = channels[batchid * BATCHSIZE : (batchid + 1) * BATCHSIZE]
            n_batch_items = len(channels_batch)
            created_image = init_image.detach().clone().repeat(n_batch_items, 1, 1, 1)

            for epoch in range(self.fv_settings.epochs):
                if epoch < self.fv_settings.epochs - self.fv_settings.epochs_without_robustness_transforms:
                    with torch.no_grad():
                        created_image = self.regularizer(created_image)
                
                created_image = created_image.detach()#.clone()
                created_image.requires_grad = True

                # Set gradients zero
                if hasattr(created_image, 'grad') and created_image.grad is not None:
                    created_image.grad.data.zero_()
                # model.zero_grad could be unnecessary, but I am not sure
                model.zero_grad()

                out_dict = model.forward_features({'data' : created_image}, module)
                output = out_dict['module_dicts'][0]['activation']
                if output.shape[2] == 1:
                    output = output.flatten(2)
                    loss_list = [output[i, j] for i, j in enumerate(channels_batch)]
                else:
                    loss_list = []
                    if self.fv_settings.mode == FeatureVisualizationParams.Mode.AVERAGE:
                        for i, j in enumerate(channels_batch):
                            activation = output[i, j]
                            loss_list.append(activation.mean())
                    elif self.fv_settings.mode == FeatureVisualizationParams.Mode.CENTERPIXEL:
                        for i, j in enumerate(channels_batch):
                            activation = output[i, j]
                            loss_list.append(activation[activation.shape[0]//2, activation.shape[1]//2])
                    elif self.fv_settings.mode == FeatureVisualizationParams.Mode.PERCENTILE:
                        for i, j in enumerate(channels_batch):
                            activation = output[i, j]
                            activation_percentile = torch.quantile(activation, 1. - self.fv_settings.fraction_to_maximize, interpolation='linear')
                            mean_new = activation[activation>activation_percentile].mean()
                            if torch.isnan(mean_new):
                                mean_new = activation.mean()
                            loss_list.append(mean_new)

                loss = -torch.stack(loss_list).sum()

                loss.backward()

                gradients = created_image.grad / (torch.sqrt((created_image.grad**2).sum((1,2,3), keepdims=True)) + 1e-6)
                created_image = created_image - gradients * self.fv_settings.lr

                if epoch % 20 == 0:
                    print(epoch, loss.item())

                if self.export and (epoch % self.export_interval == 0 or epoch == self.epochs - 1):
                    with torch.no_grad():
                        export_images = self.export_transformation(created_image.detach().cpu())
                        if export_images.shape[1] != 3:
                            export_images = export_images.expand(-1, 3, -1, -1)
                        for i, channel in enumerate(channels_batch):
                            path = os.path.join(self.export_path, "_".join([str(channel), str(epoch), str(imagecount) + ".jpg"]))
                            export_meta.append({'path' : path, 'channel' : int(channel), 'epoch' : epoch})
                            cv2.imwrite(path, export_images[i].transpose((1,2,0)))
                            imagecount += 1

            created_image_aggregate.append(created_image.detach().cpu())

        created_image = torch.cat(created_image_aggregate, 0)
        
        return created_image, export_meta


    def create_regularizer(self):
        return Regularizer(
            self.fv_settings.target_size, 
            self.fv_settings.degrees, 
            self.fv_settings.blur_sigma, 
            self.fv_settings.roll
        )


class ExportTransform(object):

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            x = x.cpu().numpy()

        minimum = x.min((1,2,3), keepdims=True)
        maximum = x.max((1,2,3), keepdims=True)
        x = x - minimum
        x = x / (maximum - minimum + 1e-6)
        x = x * 255
        x = x.astype(np.uint8)

        return x


class Regularizer(object):

    def __init__(self, target_size, degrees, blur_sigma, roll):
        
        transform_list = []
        if degrees > 0:
            padding = int((degrees / 45.) * target_size[1] / (2. * np.sqrt(2.))) # approximately the size of the blank spots created by image rotation.
            rotation = transforms.RandomApply(torch.nn.ModuleList([
                transforms.Pad(padding, padding_mode='edge'),
                transforms.RandomRotation(degrees=degrees),
                transforms.CenterCrop(target_size[1:]),
                ]), p=0.3)
            transform_list.append(rotation)

        if blur_sigma > 0.:
            blurring = tgm.image.GaussianBlur((5, 5), (blur_sigma, blur_sigma))
            # transforms.GaussianBlur(5, sigma=(0.1, blur_sigma))
            transform_list.append(blurring)

        if roll > 0:
            rolling = transforms.RandomApply(torch.nn.ModuleList([
                RandomRoll(roll),
                ]), p=0.3)
            transform_list.append(rolling)
        
        transform_list.append(DistributionRegularizer())

        self.transformation = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transformation(x)


class DistributionRegularizer(nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        # mean = x.mean((1,2,3), keepdims=True)
        # std = x.std((1,2,3), keepdims=True)
        # x_reg = (x - mean) / (std + 1e-6)
        # x_new = ((1. - self.blend) * x) +  self.blend * x_reg
        x = x.clamp(-2, 2)
        return x


class RandomRoll(nn.Module):

    def __init__(self, roll):
        super().__init__()
        self.roll = roll

    def forward(self, x):
        x = torch.roll(x, (random.randint(-self.roll, self.roll), random.randint(-self.roll, self.roll)), dims=(2, 3))
        return x