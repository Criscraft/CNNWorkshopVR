import torch

class NoiseGenerator:

    def __init__(self, device, shape):
        self.device = device
        self.active_noise_image = None
        self.shape = shape


    def generate_noise_image(self):
        # Generate image with values between 0 and 1
        self.active_noise_image = torch.rand(*self.shape, device=self.device)

        # self.active_noise_image = torch.zeros(*self.shape, device=self.device)
        # self.active_noise_image[0, 1::4, 1::4] = 1.
        # self.active_noise_image[1, 2::4, 2::4] = 1.
        # self.active_noise_image[2, 3::4, 3::4] = 1.


    def get_noise_image(self):
        if self.active_noise_image is None:
            self.generate_noise_image()
        return self.active_noise_image.clone()