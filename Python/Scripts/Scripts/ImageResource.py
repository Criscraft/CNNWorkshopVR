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
        value_zero_decoded : float = -1.,
        value_255_decoded : float = -1.,
        ):
        
        self.id = id
        self.module_id = module_id
        self.channel_id = channel_id
        self.mode = mode
        self.data = data
        self.label = label
        self.value_zero_decoded = value_zero_decoded
        self.value_255_decoded = value_255_decoded
    