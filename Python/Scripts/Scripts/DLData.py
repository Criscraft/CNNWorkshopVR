class DLData(object):
    
    def __init__(self, dataset, dataset_to_tensor, data_indices, class_names, n_classes):
        super().__init__()
        self.dataset = dataset
        self.dataset_to_tensor = dataset_to_tensor
        self.data_indices = data_indices
        self.class_names = class_names
        self.n_classes = n_classes
    
    
    def __len__(self):
        
        if self.data_indices:
            return len(self.data_indices)
        else:
            return len(self.dataset)
    
    def get_data_item(self, idx: int, transformed=True):
        if transformed:
            dataset = self.dataset
        else:
            dataset = self.dataset_to_tensor
        if self.data_indices:
            item = dataset[self.data_indices[idx]]
        else:
            item = dataset[idx]
        return item['data'], item['label']

    