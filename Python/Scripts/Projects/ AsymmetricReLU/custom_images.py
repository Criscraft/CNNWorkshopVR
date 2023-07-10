import torch.utils.data as data
import numpy as np


class CustomImages(data.Dataset):

    # images of shape [1, 11, 11]

    def __init__(self):
        super().__init__()


    def prepare(self, shared_modules):
        self.transform_fn = None
        if self.transform_name:
            self.transform_fn = shared_modules[self.transform_name]


        self.images = []

        # all gray
        x = np.zeros((11, 11))
        x = (x, 0) # add dummy label
        self.images.append(x)

        # center pos
        x = np.zeros((11, 11))
        x[4:6, 4:6] = 1.0
        x = (x, 0)
        self.images.append(x)

        # center neg
        x = np.zeros((11, 11))
        x[4:6, 4:6] = -1.0
        x = (x, 0)
        self.images.append(x)

        self.class_names = ['dummy_label']
        
        
    def __len__(self):
        return len(self.embedded_dataset)


    def __getitem__(self, idx):
        item = self.transform_fn(self.images[idx])
        sample = {'data' : item[0], 'label' : item[1], 'id' : idx, 'tags' :  dict(self.tags)}
        return sample