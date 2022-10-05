from torchvision import transforms


class TransformTestGS(object):

    def __init__(self,
        norm_mean=[1.],
        norm_std=[1.],
        ):
        super().__init__()

        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=norm_mean[0], std=norm_std[0]),
        ])


    def __call__(self, img):
        return self.transform(img)