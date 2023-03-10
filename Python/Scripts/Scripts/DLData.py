class DLData(object):
    
    def __init__(self, dataset, data_indices, class_names, n_classes):
        super().__init__()
        self.dataset = dataset
        self.data_indices = data_indices
        self.class_names = class_names
        self.n_classes = n_classes
    
    
    def __len__(self):
        
        if self.data_indices:
            return len(self.data_indices)
        else:
            return len(self.dataset)
    
    def get_data_item(self, idx: int):
        if self.data_indices:
            item = self.dataset[self.data_indices[idx]]
        else:
            item = self.dataset[idx]
        return item['data'], item['label']

    