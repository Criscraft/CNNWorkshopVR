import torch.utils.data as data
from torchvision.datasets import MNIST as MNIST_ORIG


class MNIST(data.Dataset):

    # images of shape [1, 28, 28]

    def __init__(self,
        b_train=True,
        transform='',
        root='',
        download=False,
        tags={}):
        super().__init__()

        self.b_train = b_train
        self.transform_name = transform
        self.root = root
        self.download = download
        self.tags = tags


    def prepare(self, shared_modules):
        transform_fn = None
        if self.transform_name:
            transform_fn = shared_modules[self.transform_name]
        self.embedded_dataset = MNIST_ORIG(root=self.root, train=self.b_train, download=self.download, transform=transform_fn)

        self.class_names = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
        
        
    def __len__(self):
        return len(self.embedded_dataset)


    def __getitem__(self, idx):
        item = self.embedded_dataset[idx]
        sample = {'data' : item[0], 'label' : item[1], 'id' : idx, 'tags' :  dict(self.tags)}
        return sample