import torch.utils.data as data
import os
from shutil import copytree
import numpy as np
import scipy.io as sio
from PIL import Image
from multiprocessing import Manager
from typing import Tuple
from collections import OrderedDict


class Flowers102(data.Dataset):

    def __init__(self,
        datapath: str,
        path_to_splits: str,
        path_to_labels: str,
        mode: str,
        transform: str = '',
        copy_data_to: str = '',
        convert_to_rbg_image: bool = True,
        tags: Tuple = {},
        nsamples_per_class: int = -1,
        custom_seed: int = 42):
        super().__init__()

        self.datapath = datapath
        self.path_to_splits = path_to_splits
        self.path_to_labels = path_to_labels
        self.mode = mode
        self.transform_name = transform
        self.copy_data_to = copy_data_to
        self.convert_to_rbg_image = convert_to_rbg_image
        self.tags = tags
        self.nsamples_per_class = nsamples_per_class
        self.custom_seed = custom_seed
        self.class_names = [
            'pink primrose',
            'hard-leaved pocket orchid',
            'canterbury bells',
            'sweet pea',
            'english marigold',
            'tiger lily',
            'moon orchid',
            'bird of paradise',
            'monkshood',
            'globe thistle',
            'snapdragon',
            "colt's foot",
            'king protea',
            'spear thistle',
            'yellow iris',
            'globe-flower',
            'purple coneflower',
            'peruvian lily',
            'balloon flower',
            'giant white arum lily',
            'fire lily',
            'pincushion flower',
            'fritillary',
            'red ginger',
            'grape hyacinth',
            'corn poppy',
            'prince of wales feathers',
            'stemless gentian',
            'artichoke',
            'sweet william',
            'carnation',
            'garden phlox',
            'love in the mist',
            'mexican aster',
            'alpine sea holly',
            'ruby-lipped cattleya',
            'cape flower',
            'great masterwort',
            'siam tulip',
            'lenten rose',
            'barbeton daisy',
            'daffodil',
            'sword lily',
            'poinsettia',
            'bolero deep blue',
            'wallflower',
            'marigold',
            'buttercup',
            'oxeye daisy',
            'common dandelion',
            'petunia',
            'wild pansy',
            'primula',
            'sunflower',
            'pelargonium',
            'bishop of llandaff',
            'gaura',
            'geranium',
            'orange dahlia',
            'pink-yellow dahlia',
            'cautleya spicata',
            'japanese anemone',
            'black-eyed susan',
            'silverbush',
            'californian poppy',
            'osteospermum',
            'spring crocus',
            'bearded iris',
            'windflower',
            'tree poppy',
            'gazania',
            'azalea',
            'water lily',
            'rose',
            'thorn apple',
            'morning glory',
            'passion flower',
            'lotus',
            'toad lily',
            'anthurium',
            'frangipani',
            'clematis',
            'hibiscus',
            'columbine',
            'desert-rose',
            'tree mallow',
            'magnolia',
            'cyclamen',
            'watercress',
            'canna lily',
            'hippeastrum',
            'bee balm',
            'ball moss',
            'foxglove',
            'bougainvillea',
            'camellia',
            'mallow',
            'mexican petunia',
            'bromelia',
            'blanket flower',
            'trumpet creeper',
            'blackberry lily',
            ]
        
    def prepare(self, shared_modules):
        #load data to /tmp if not already there
        if self.copy_data_to:
            datapath_local = os.path.join(self.copy_data_to, self.datapath.split('/')[-1])
            if not os.path.isdir(datapath_local):
                copytree(self.datapath, datapath_local)
            self.datapath = datapath_local

        self.transform = None
        if self.transform_name:
            self.transform = shared_modules[self.transform_name]
        
        split = sio.loadmat(self.path_to_splits)
        if self.mode=='train':
            idx = np.concatenate((split['trnid'][0], split['valid'][0]))
        elif self.mode=='test':
            idx = split['tstid'][0]
        elif self.mode=='all':
            idx = np.concatenate((split['trnid'][0], split['valid'][0], split['tstid'][0]))
        else:
            raise ValueError("Unknown mode " + self.mode + " given.")

        image_paths = [f"image_{i:05.0f}.jpg" for i in idx]
        image_paths = np.array(image_paths)
        labels = sio.loadmat(self.path_to_labels)
        labels = labels['labels'][0].astype(np.int64) - 1 #matlab compatibility, apparantly pytorch really needs the datatype to be np.int64 and not np.int32, since otherwise there is the error Expected object of scalar type Long but got scalar type Int for argument #2 'target' in call to _thnn_nll_loss_forward
        labels = labels[idx - 1] #matlab compatibility
        self.n_classes = len(np.unique(labels))
        self.label_to_indices = {c : np.where(labels==c)[0] for c in range(self.n_classes)}

        #pick nsamples_per_class many images per class and omit the other images
        if self.nsamples_per_class > 0:
            idx = []
            randomstate = np.random.RandomState(self.custom_seed)
            for c in range(self.n_classes):
                idx_class = self.label_to_indices[c]
                permutation = randomstate.permutation(len(idx_class))
                idx.extend(idx_class[permutation[:self.nsamples_per_class]])
            idx = np.array(idx)

            labels = labels[idx]
            image_paths = image_paths[idx]
            self.label_to_indices = {c : np.where(labels==c)[0] for c in range(self.n_classes)}

        # use manager to improve the shared memory between workers which load data. Avoids the effect of ever increasing memory usage. See https://github.com/pytorch/pytorch/issues/13246#issuecomment-445446603
        manager = Manager() 
        self.tags = manager.dict(self.tags)
        self.labels = manager.list(labels)
        self.image_paths = manager.list(image_paths)
        

    def __len__(self):
        return len(self.labels)


    def __getitem__(self, idx):
        path = self.image_paths[idx]
        label = self.labels[idx]

        filename = os.path.join(self.datapath, path)
        out_image = Image.open(filename)
        if self.convert_to_rbg_image:
            out_image = out_image.convert('RGB')
        if self.transform is not None:
            out_image = self.transform(out_image)
        sample = {'data' : out_image, 'label' : label, 'id' : idx, 'tags' : dict(self.tags), 'path' : filename}
        return sample