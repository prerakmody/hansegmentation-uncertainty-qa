import tensorflow as tf

class ZipDataset:

    def __init__(self, datasets):
        self.datasets = datasets
        self.datasets_generators = []
        self._init_constants()
        
    
    def _init_constants(self):
        self.HU_MIN = self.datasets[0].HU_MIN
        self.HU_MAX = self.datasets[0].HU_MAX
        self.dataset_dir_processed = self.datasets[0].dataset_dir_processed
        self.grid        = self.datasets[0].grid
        self.pregridnorm = self.datasets[0].pregridnorm
        
    def __len__(self):
        length = 0
        for dataset in self.datasets:
            length += len(dataset)
        
        return length

    def generator(self):
        for dataset in self.datasets:
            self.datasets_generators.append(dataset.generator())
        return tf.data.experimental.sample_from_datasets(self.datasets_generators) #<_DirectedInterleaveDataset shapes: (<unknown>, <unknown>, <unknown>, <unknown>), types: (tf.float32, tf.float32, tf.int16, tf.string)>

    def get_subdataset(self, param_name):
        if type(param_name) == str:
            for dataset in self.datasets:
                if dataset.name == param_name:
                    return dataset
        else:
            print (' - [ERROR][ZipDataset] param_name needs to a str')
        
        return None
    
    def get_label_map(self, label_map_full=False):
        if label_map_full:
            return self.datasets[0].LABEL_MAP_FULL
        else:
            return self.datasets[0].LABEL_MAP
    
    def get_label_colors(self):
        return self.datasets[0].LABEL_COLORS
    
    def get_label_weights(self):
        return self.datasets[0].LABEL_WEIGHTS
    
    def get_mask_type(self, idx=0):
        return self.datasets[idx].mask_type