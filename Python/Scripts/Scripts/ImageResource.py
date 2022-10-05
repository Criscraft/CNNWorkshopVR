from enum import Enum

class ImageResource(object):
    
    class Mode(str, Enum):
        DATASET = "DATASET"
        ACTIVATION = "ACTIVATION"
        FEATURE_VISUALIZATION = "FEATURE_VISUALIZATION"
        NOISE = "NOISE"
        
    def __init__(self, 
        id : int = -1,
        module_id : int = -1,
        channel_id : int = -1,
        mode : Mode = None,
        label : str = "",
        data = None,
        ):
        
        self.id = id
        self.module_id = module_id
        self.channel_id = channel_id
        self.mode = mode
        self.data = data
        self.label = label
    