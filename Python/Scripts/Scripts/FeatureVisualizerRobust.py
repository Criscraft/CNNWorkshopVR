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


BATCHSIZE = 16
imagecount = 0


class FeatureVisualizationParams(object):
    
    class Mode(str, Enum):
        AVERAGE = "AVERAGE"
        CENTERPIXEL = "CENTERPIXEL"
        PERCENTILE = "PERCENTILE"
        
    def __init__(self,
        mode=Mode.CENTERPIXEL,
        epochs=200,
        epochs_without_robustness_transforms=10,
        lr=20.,
        degrees=0,
        blur_sigma=0.5,
        roll=0,
        fraction_to_maximize=0.25,
        slope_leaky_relu_scheduling=True,
        final_slope_leaky_relu = 0.01,
        pool_mode="avgpool", # has no effect but is listed here for completeness
        mimic_poolstage_filter_size=False, # has no effect but is listed here for completeness
        norm_mean = (0.,),
        norm_std=(1.,),
        ):

        self.mode = mode
        self.epochs = epochs
        self.epochs_without_robustness_transforms = epochs_without_robustness_transforms
        self.lr = lr
        self.degrees = degrees
        self.blur_sigma = blur_sigma
        self.roll = roll
        self.fraction_to_maximize = fraction_to_maximize
        self.slope_leaky_relu_scheduling = slope_leaky_relu_scheduling
        self.final_slope_leaky_relu = final_slope_leaky_relu
        self.norm_mean = norm_mean
        self.norm_std = norm_std


class FeatureVisualizer(object):
    
    def __init__(self,
        export = False,
        export_path = '../export',
        export_interval=1e6,
        fv_settings=FeatureVisualizationParams()):
        
        super().__init__()
        self.export = export
        self.export_interval = export_interval
        self.export_path = export_path
        self.set_fv_settings(fv_settings)

    
    def set_fv_settings(self, feature_visualization_params):
        self.fv_settings = feature_visualization_params


    def visualize(self, model, module, device, init_image, n_channels, channels=None):
        global imagecount
        export_meta = []
        if channels is None:
            channels = np.arange(n_channels)
        n_batches = int( np.ceil( n_channels / float(BATCHSIZE) ) )
        
        norm_mean = torch.tensor(self.fv_settings.norm_mean)
        norm_mean = norm_mean.reshape((1, -1, 1, 1))
        norm_std = torch.tensor(self.fv_settings.norm_std)
        norm_std = norm_std.reshape((1, -1, 1, 1))

        if init_image.ndim==3:
            init_image = init_image.unsqueeze(0)
        #init_image = (init_image - norm_mean) / norm_std # transform init_image from image space to network input space
        init_image = init_image.to(device)
        
        export_transformation = ExportTransform(norm_mean, norm_std)
        distribution_regularizer = DistributionRegularizer(norm_mean.to(init_image.device), norm_std.to(init_image.device))
        regularizer = self.create_regularizer(init_image.shape[2:])
        
        
        print("Start gradient ascent on images")
        print("Feature visualization parameters:")
        print(vars(self.fv_settings))

        created_image_aggregate = []
        for batchid in range(n_batches):
            channels_batch = channels[batchid * BATCHSIZE : (batchid + 1) * BATCHSIZE]
            n_batch_items = len(channels_batch)
            created_image = init_image.detach().clone().repeat(n_batch_items, 1, 1, 1)
            # LeakyReLU slope scheduling
            if self.fv_settings.slope_leaky_relu_scheduling:
                model.embedded_model.set_neg_slope_of_leaky_relus(1.)
                start_epoch = 0.2 * self.fv_settings.epochs
                start_slope = 1.0
                end_epoch = 0.3 * self.fv_settings.epochs
                end_slope = self.fv_settings.final_slope_leaky_relu

            for epoch in range(self.fv_settings.epochs):

                # LeakyReLU slope scheduling
                if self.fv_settings.slope_leaky_relu_scheduling:    
                    if epoch < start_epoch:
                        slope = 1.0
                    elif epoch > end_epoch:
                        slope = self.fv_settings.final_slope_leaky_relu
                    else:
                        p = (epoch - start_epoch) / (end_epoch - start_epoch)
                        slope = p * end_slope + (1 - p) * start_slope
                        model.embedded_model.set_neg_slope_of_leaky_relus(slope)

                with torch.no_grad():
                    if epoch < self.fv_settings.epochs - self.fv_settings.epochs_without_robustness_transforms:
                        created_image = regularizer(created_image)
                    created_image = distribution_regularizer(created_image)
                
                created_image = created_image.detach()
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
                
                gradients = created_image.grad
                fft = torch.fft.fft2(gradients)
                fft_mag = torch.abs(fft)
                fft = fft / (fft_mag + 1e-6)
                fft_gradients = torch.fft.ifft2(fft)
                fft_gradients = torch.real(fft_gradients)
                #gradients = 0.5 * gradients + 0.5 * fft_gradients
                gradients = fft_gradients / (torch.sqrt((fft_gradients**2).sum((1,2,3), keepdim=True)) + 1e-6)
                created_image = created_image - gradients * self.fv_settings.lr

                if epoch % 20 == 0:
                    print(epoch, loss.item())

                if self.export and (epoch % self.export_interval == 0 or epoch == self.fv_settings.epochs - 1):
                    with torch.no_grad():
                        export_images = export_transformation(created_image.detach().cpu())
                        if export_images.shape[1] != 3:
                            export_images = export_images.expand(-1, 3, -1, -1)
                        for i, channel in enumerate(channels_batch):
                            path = os.path.join(self.export_path, "_".join([str(channel), str(epoch), str(imagecount) + ".png"]))
                            export_meta.append({'path' : path, 'channel' : int(channel), 'epoch' : epoch})
                            image = export_images[i]
                            if image.shape[0] == 3:
                                image = image[[2, 1, 0]] # sort color channels to BGR
                            image = image.transpose((1,2,0)) # sort dims to H, W, C
                            cv2.imwrite(path, image)
                            imagecount += 1

            created_image_aggregate.append(created_image.detach().cpu())

        created_image_aggregate = torch.cat(created_image_aggregate, 0)
        created_image_aggregate = export_transformation(created_image_aggregate)
        
        return created_image_aggregate, export_meta


    def create_regularizer(self, target_size):
        return Regularizer(
            self.fv_settings.degrees, 
            self.fv_settings.blur_sigma, 
            self.fv_settings.roll,
            target_size,
        )


class ExportTransform(object):

    def __init__(self, norm_mean, norm_std):
        self.norm_mean = norm_mean
        self.norm_std = norm_std

    def __call__(self, x):
        # x is on cpu
        with torch.no_grad():
            #mean = x.mean((1,2,3), keepdim=True)
            #std = x.std((1,2,3), keepdim=True)
            #std = 1.
            #x = (x - mean) / std
            x = x * self.norm_std + self.norm_mean
            x = x.clamp(0., 1.)
        x = x.numpy()
        x = x * 255
        x = x.astype(np.uint8)
        return x


# TODO Remove the target size. the size of the images should be determined by the init image.
class Regularizer(object):

    def __init__(self, degrees, blur_sigma, roll, target_size, color_jitter=True):
        transform_list = []
        if degrees > 0:
            padding = int((degrees / 45.) * target_size[1] / (2. * np.sqrt(2.))) # approximately the size of the blank spots created by image rotation.
            rotation = transforms.RandomApply(torch.nn.ModuleList([
                transforms.Pad(padding, padding_mode='edge'),
                #transforms.Pad(padding, fill=0, padding_mode='constant'),
                transforms.RandomRotation(degrees=degrees),
                transforms.CenterCrop(target_size[1:]),
                ]), p=0.2)
            transform_list.append(rotation)

        if blur_sigma > 0.:
            blurring = tgm.image.GaussianBlur((5, 5), (blur_sigma, blur_sigma))
            # transforms.GaussianBlur(5, sigma=(0.1, blur_sigma))
            transform_list.append(blurring)

        if roll > 0:
            rolling = transforms.RandomApply(torch.nn.ModuleList([
                RandomRoll(roll),
                ]), p=0.2)
            transform_list.append(rolling)

        if color_jitter:
            cjitter = transforms.RandomApply(torch.nn.ModuleList([
                transforms.ColorJitter(brightness=0.1, contrast=0.2, saturation=0.1, hue=0.02),
                ]), p=0.2)
            transform_list.append(cjitter)
            

        self.transformation = transforms.Compose(transform_list)

    def __call__(self, x):
        return self.transformation(x)


class DistributionRegularizer(nn.Module):

    def __init__(self, norm_mean, norm_std):
        super().__init__()

        self.norm_mean = norm_mean
        self.norm_std = norm_std


    def forward(self, x):
        # Transform to image space
        # x = x * self.norm_std + self.norm_mean
        # # x should in the range [0, 1]
        # minimum = x.min(1, keepdim=True)[0].min(2, keepdim=True)[0].min(3, keepdim=True)[0]
        # maximum = x.max(1, keepdim=True)[0].max(2, keepdim=True)[0].max(3, keepdim=True)[0]
        # x = x - minimum
        # x = x / (maximum - minimum + 1e-6)
        # # Transform back to network input space
        # x = (x - self.norm_mean) / self.norm_std
        x = x * self.norm_std + self.norm_mean
        x = x * 0.999
        x = x.clamp(0., 1.)
        x = (x - self.norm_mean) / self.norm_std
        #mean = x.mean((1,2,3), keepdim=True)
        #std = x.std((1,2,3), keepdim=True)
        #std = 1.
        #x = (x - mean) / std
        return x


class RandomRoll(nn.Module):

    def __init__(self, roll):
        super().__init__()
        self.roll = roll

    def forward(self, x):
        x = torch.roll(x, (random.randint(-self.roll, self.roll), random.randint(-self.roll, self.roll)), dims=(2, 3))
        return x