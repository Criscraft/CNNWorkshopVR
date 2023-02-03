import torch

class NoiseGenerator:

    def __init__(self, device, shape, grayscale=False):
        self.device = device
        self.active_noise_image = None
        self.shape = shape


    def generate_noise_image(self):
        # Generate image with values between 0 and 1
        # self.active_noise_image = torch.randn(*self.shape, device=self.device)
        # minimum = self.active_noise_image.min()
        # maximum = self.active_noise_image.max()
        # self.active_noise_image = (self.active_noise_image - minimum) / (maximum - minimum)
        self.active_noise_image = torch.zeros(*self.shape, device=self.device)
        self.active_noise_image[:, self.shape[1]//2, self.shape[2]//2] = 1.


    def get_noise_image(self):
        if self.active_noise_image is None:
            self.generate_noise_image()
        return self.active_noise_image