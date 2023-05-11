import torch

class GrayGenerator:

    def __init__(self, device, shape):
        self.device = device
        self.active_noise_image = None
        self.shape = shape


    def generate_noise_image(self):
        self.active_noise_image = torch.ones(*self.shape, device=self.device) * 0.5


    def get_noise_image(self):
        if self.active_noise_image is None:
            self.generate_noise_image()
        return self.active_noise_image.clone()