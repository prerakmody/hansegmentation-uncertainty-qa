# Import private libraries
import src.config as config
import src.dataloader.utils as utils

# Import public libraries
import pdb
import time
import json
import itertools
import traceback
import numpy as np
from pathlib import Path

import tensorflow as tf
 

class HaNDeepMindTCIADataset:
    """
    Contains data from https://github.com/deepmind/tcia-ct-scan-dataset
    - Ref: Deep learning to achieve clinically applicable segmentation of head and neck anatomy for radiotherapy
    """

    def __init__(self, data_dir, dir_type, annotator_type
                    , grid=True, crop_init=False, resampled=False, mask_type=config.MASK_TYPE_ONEHOT
                    , transforms=[], filter_grid=False, random_grid=False, pregridnorm=False
                    , patient_shuffle=False
                    , centred_dataloader_prob = 0.0
                    , parallel_calls=None, deterministic=False
                    , debug=False):
        
        self.name = '{}_DeepMindTCIA'.format(config.HEAD_AND_NECK)
        self.class_name = 'HaNDeepMindTCIADataset'

        # Params - Source 
        self.data_dir       = data_dir
        self.dir_type       = dir_type
        self.annotator_type = annotator_type

        # Params - Spatial (x,y)
        self.grid      = grid
        self.crop_init = crop_init
        self.resampled = resampled
        self.mask_type = mask_type

        # Params - Transforms/Filters
        self.transforms  = transforms
        self.filter_grid = filter_grid
        self.random_grid = random_grid
        self.pregridnorm = pregridnorm

        # Params - Memory related
        self.patient_shuffle = patient_shuffle
        
        # Params - Dataset related
        self.centred_dataloader_prob = centred_dataloader_prob

        # Params - TFlow Dataloader related
        self.parallel_calls = parallel_calls # [1, tf.data.experimental.AUTOTUNE]
        self.deterministic  = deterministic

        # Params - Debug
        self.debug         = debug

        # Data items
        self.data       = {}
        self.paths_img  = []
        self.paths_mask = []
        self.cache      = {}
        self.filter     = None

        # Config
        self.dataset_config = getattr(config, self.name)

        # Function calls
        self._download()
        self._init_data()

    def __len__(self):
        
        if self.grid:

            if self.crop_init:

                if self.voxel_shape_cropped == [240,240,80]:

                    if self.grid_size == [240,240,80]:
                        return len(self.paths_img) * 1
                    elif self.grid_size == [240,240,40]:
                        return len(self.paths_img) * 2
                    elif self.grid_size == [140,140,40]:
                        return len(self.paths_img) * 8
            else:
                return None

        else:
            return len(self.paths_img)
    
    def _download(self):

        self.dataset_dir           = Path(self.data_dir).joinpath(self.name)
        self.dataset_dir_raw       = Path(self.dataset_dir).joinpath(config.DIRNAME_RAW)
        
        self.VOXEL_RESO            = self.dataset_config[config.KEY_VOXELRESO]
        if self.VOXEL_RESO == (0.8,0.8,2.5):
            self.dataset_dir_processed = Path(self.dataset_dir).joinpath(config.DIRNAME_PROCESSED)
        self.dataset_dir_processed_3D  = Path(self.dataset_dir_processed).joinpath(self.dir_type, self.annotator_type, config.DIRNAME_SAVE_3D)
        self.dataset_dir_datatypes     = [
                                        Path(config.DATALOADER_DEEPMINDTCIA_TEST, config.DATALOADER_DEEPMINDTCIA_ONC)
                                        , Path(config.DATALOADER_DEEPMINDTCIA_TEST, config.DATALOADER_DEEPMINDTCIA_RAD)
                                        , Path(config.DATALOADER_DEEPMINDTCIA_VAL, config.DATALOADER_DEEPMINDTCIA_ONC)
                                        , Path(config.DATALOADER_DEEPMINDTCIA_VAL, config.DATALOADER_DEEPMINDTCIA_RAD)
                                        ]

        if not Path(self.dataset_dir_raw).exists() or not Path(self.dataset_dir_processed).exists():
            print ('')
            print (' ------------------ {} Dataset ------------------'.format(self.name))
        
        if not Path(self.dataset_dir_raw).exists():
            print ('')
            print (' ------------------ Download Data ------------------')
            from src.dataloader.extractors.han_deepmindtcia import HaNDeepMindTCIADownloader
            downloader = HaNDeepMindTCIADownloader(self.dataset_dir_raw, self.dataset_dir_processed)
            downloader.download()
            downloader.sort()
        
        if not Path(self.dataset_dir_processed_3D).exists():
            print ('')
            print (' ------------------ Process Data (3D) ------------------')
            from src.dataloader.extractors.han_deepmindtcia import HaNDeepMindTCIAExtractor
            extractor = HaNDeepMindTCIAExtractor(self.name, self.dataset_dir_raw, self.dataset_dir_processed, self.dataset_dir_datatypes)
            extractor.extract3D()
            print ('')
            print (' ------------------------- * -------------------------')
            print ('')
    
    def _init_data(self):
        
        # Step 0 - Init vars
        self.patient_meta_info = {}
        self.path_img_csv      = ''
        self.path_mask_csv     = ''
        if self.VOXEL_RESO == (0.8,0.8,2.5):
            self.patients_z_prob   = []
        else:
            self.patients_z_prob   = []

        # Step 1 - Define global paths
        self.data_dir_processed = Path(self.dataset_dir_processed).joinpath(self.dir_type, self.annotator_type)
        if not Path(self.data_dir_processed).exists():
            print (' - [ERROR][{}][_init_data()] Processed Dir Path issue: {}'.format(self.class_name, self.data_dir_processed))
        self.data_dir_processed_3D = Path(self.data_dir_processed).joinpath(config.DIRNAME_SAVE_3D)

        # Step 2.1 - Get paths for 2D/3D
        if self.resampled is False:
            self.path_img_csv  = Path(self.data_dir_processed_3D).joinpath(config.FILENAME_CSV_IMG)
            self.path_mask_csv = Path(self.data_dir_processed_3D).joinpath(config.FILENAME_CSV_MASK)
        else:
            self.path_img_csv  = Path(self.data_dir_processed_3D).joinpath(config.FILENAME_CSV_IMG_RESAMPLED)
            self.path_mask_csv = Path(self.data_dir_processed_3D).joinpath(config.FILENAME_CSV_MASK_RESAMPLED)
        
        # Step 2.2 - Get file paths
        if Path(self.path_img_csv).exists() and Path(self.path_mask_csv).exists():
            self.paths_img  = utils.read_csv(self.path_img_csv)
            self.paths_mask = utils.read_csv(self.path_mask_csv)

            exit_condition = False
            for path_img in self.paths_img:
                if not Path(path_img).exists():
                    print (' - [ERROR][{}][_init_data()] path issue: path_img: {}/{}, {}'.format(self.class_name, self.dir_type, self.annotator_type, path_img))
                    exit_condition = True
            for path_mask in self.paths_mask:
                if not Path(path_mask).exists():
                    print (' - [ERROR][{}][_init_data()] path issue: path_mask: {}/{}, {}'.format(self.class_name, self.dir_type, self.annotator_type, path_mask))
                    exit_condition = True
            
            if exit_condition:
                print ('\n - [ERROR][{}][_init_data()] Exiting due to path issues'.format(self.class_name))
                import sys; sys.exit(1)

        else:
            print (' - [ERROR][{}] Issue with path'.format(self.class_name))
            print (' -- [ERROR][{}] self.path_img_csv : ({}) {}'.format(self.class_name, Path(self.path_img_csv).exists(), self.path_img_csv ))
            print (' -- [ERROR][{}] self.path_mask_csv: ({}) {}'.format(self.class_name, Path(self.path_mask_csv).exists(), self.path_mask_csv ))

        # Step 3.1 - Meta for labels
        self.LABEL_MAP          = self.dataset_config[config.KEY_LABEL_MAP]
        self.LABEL_COLORS       = self.dataset_config[config.KEY_LABEL_COLORS]
        if len(self.dataset_config[config.KEY_LABEL_WEIGHTS]):
            self.LABEL_WEIGHTS      = np.array(self.dataset_config[config.KEY_LABEL_WEIGHTS])
            self.LABEL_WEIGHTS      = (self.LABEL_WEIGHTS / np.sum(self.LABEL_WEIGHTS)).tolist()
        else:
            self.LABEL_WEIGHTS = []

        # Step 3.2 - Meta for voxel HU
        self.HU_MIN             = self.dataset_config['HU_MIN']
        self.HU_MAX             = self.dataset_config['HU_MAX']

        # Step 4 - Get patient meta info
        if self.resampled is False:
            print (' - [Warning][{}]: This dataloader is not extracting 3D Volumes which have been resampled to the same 3D voxel spacing: {}/{}'.format(self.class_name, self.dir_type, self.annotator_type))
        for path_img in self.paths_img:
            path_img_json = Path(path_img).parent.absolute().joinpath(config.FILENAME_VOXEL_INFO)
            with open(str(path_img_json), 'r') as fp:
                patient_config_file = json.load(fp)
                
                patient_id = Path(path_img).parts[-2]
                midpoint_idxs_mean    = []
                missing_label_names   = []
                voxel_shape_resampled = []
                voxel_shape_orig      = []
                if self.resampled is False:
                    midpoint_idxs_mean  = patient_config_file[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_MEAN_MIDPOINT]
                    missing_label_names = patient_config_file[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_LABEL_MISSING]
                    voxel_shape_orig    = patient_config_file[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_SHAPE]
                else:
                    if config.TYPE_VOXEL_RESAMPLED in patient_config_file:
                        midpoint_idxs_mean    = patient_config_file[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_MEAN_MIDPOINT]
                        missing_label_names   = patient_config_file[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_LABEL_MISSING]
                        voxel_shape_orig      = patient_config_file[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_SHAPE]
                        voxel_shape_resampled = patient_config_file[config.TYPE_VOXEL_RESAMPLED][config.KEYNAME_SHAPE]
                        
                    else:
                        print (' - [ERROR][{}][_init_data()] There is no resampled data and you have set resample=True'.format(self.class_name))
                        print ('  -- Delete _data/{}/processed directory and set VOXEL_RESO to a tuple of pixel spacing values for x,y,z axes'.format(self.name))
                        print ('  -- Exiting now! ')
                        import sys; sys.exit(1)
                
                if len(midpoint_idxs_mean):
                    midpoint_idxs_mean = np.array(midpoint_idxs_mean).astype(config.DATATYPE_VOXEL_IMG).tolist()
                    self.patient_meta_info[patient_id] = {
                        config.KEYNAME_MEAN_MIDPOINT     : midpoint_idxs_mean
                        , config.KEYNAME_LABEL_MISSING   : missing_label_names
                        , config.KEYNAME_SHAPE_ORIG      : voxel_shape_orig
                        , config.KEYNAME_SHAPE_RESAMPLED : voxel_shape_resampled
                    }    
                else:
                    if self.crop_init:
                        print ('')
                        print (' - [ERROR][{}][_init_data()] Crop is set to true and there is no midpoint mean idx'.format(self.class_name))
                        print ('  -- Exiting now! ')
                        import sys; sys.exit(1)
        
        # Step 6 - Meta for grid sampling
        if self.resampled:

            if self.crop_init:
                self.crop_info        = self.dataset_config[config.KEY_PREPROCESS][config.TYPE_VOXEL_RESAMPLED][config.KEY_CROP][str(self.VOXEL_RESO)]
                self.w_lateral_crop   = self.crop_info[config.KEY_MIDPOINT_EXTENSION_W_LEFT]
                self.w_medial_crop    = self.crop_info[config.KEY_MIDPOINT_EXTENSION_W_RIGHT]
                self.h_posterior_crop = self.crop_info[config.KEY_MIDPOINT_EXTENSION_H_BACK]
                self.h_anterior_crop  = self.crop_info[config.KEY_MIDPOINT_EXTENSION_H_FRONT]
                self.d_cranial_crop   = self.crop_info[config.KEY_MIDPOINT_EXTENSION_D_TOP]
                self.d_caudal_crop    = self.crop_info[config.KEY_MIDPOINT_EXTENSION_D_BOTTOM]
                self.voxel_shape_cropped = [self.w_lateral_crop+self.w_medial_crop
                                            , self.h_posterior_crop+self.h_anterior_crop
                                            , self.d_cranial_crop+self.d_caudal_crop]

            if self.grid:
                grid_3D_params         = self.dataset_config[config.KEY_GRID_3D][config.TYPE_VOXEL_RESAMPLED][str(self.VOXEL_RESO)]
                self.grid_size         = grid_3D_params[config.KEY_GRID_SIZE]
                self.grid_overlap      = grid_3D_params[config.KEY_GRID_OVERLAP]
                self.SAMPLER_PERC      = grid_3D_params[config.KEY_GRID_SAMPLER_PERC]
                self.RANDOM_SHIFT_MAX  = grid_3D_params[config.KEY_GRID_RANDOM_SHIFT_MAX]
                self.RANDOM_SHIFT_PERC = grid_3D_params[config.KEY_GRID_RANDOM_SHIFT_PERC]

                self.w_grid, self.h_grid, self.d_grid          = self.grid_size
                self.w_overlap, self.h_overlap, self.d_overlap = self.grid_overlap
            
            else:

                if self.crop_init:
                    self.w_grid, self.h_grid, self.d_grid = self.voxel_shape_cropped
                else:
                    print (' - [ERROR][HaNMICCAI2015Dataset] No info present for non-grid cropping')
            
        else:
            if self.crop_init:
                self.crop_info        = self.dataset_config[config.KEY_PREPROCESS][config.TYPE_VOXEL_ORIGSHAPE][config.KEY_CROP][str(self.VOXEL_RESO)]
                self.w_lateral_crop   = self.crop_info[config.KEY_MIDPOINT_EXTENSION_W_LEFT]
                self.w_medial_crop    = self.crop_info[config.KEY_MIDPOINT_EXTENSION_W_RIGHT]
                self.h_posterior_crop = self.crop_info[config.KEY_MIDPOINT_EXTENSION_H_BACK]
                self.h_anterior_crop  = self.crop_info[config.KEY_MIDPOINT_EXTENSION_H_FRONT]
                self.d_cranial_crop   = self.crop_info[config.KEY_MIDPOINT_EXTENSION_D_TOP]
                self.d_caudal_crop    = self.crop_info[config.KEY_MIDPOINT_EXTENSION_D_BOTTOM]
                self.voxel_shape_cropped = [self.w_lateral_crop+self.w_medial_crop
                                            , self.h_posterior_crop+self.h_anterior_crop
                                            , self.d_cranial_crop+self.d_caudal_crop]
                
            if self.grid:
                pass
            else:
                if self.crop_init:
                    self.w_grid, self.h_grid, self.d_grid = self.voxel_shape_cropped
                else:
                    print (' - [ERROR][HaNMICCAI2015Dataset] No info present for non-crop size')

    def generator(self):
        """
         - Note: 
            - In general, even when running your model on an accelerator like a GPU or TPU, the tf.data pipelines are run on the CPU
                - Ref: https://www.tensorflow.org/guide/data_performance_analysis#analysis_workflow
        """

        try:
            
            if len(self.paths_img) and len(self.paths_mask):
                
                # Step 1 - Create basic generator
                dataset = None
                dataset = tf.data.Dataset.from_generator(self._generator3D
                    , output_types=(config.DATATYPE_TF_FLOAT32, config.DATATYPE_TF_UINT8, config.DATATYPE_TF_INT32, tf.string)
                    ,args=())
                
                # Step 2 - Get 3D data
                dataset = dataset.map(self._get_data_3D, num_parallel_calls=self.parallel_calls, deterministic=self.deterministic)
                        
                # Step 3 - Filter function
                if self.filter_grid:
                    dataset = dataset.filter(self.filter.execute)

                # Step 4 - Data augmentations
                if len(self.transforms):
                    for transform in self.transforms:
                        try:
                            dataset = dataset.map(transform.execute, num_parallel_calls=self.parallel_calls, deterministic=self.deterministic)
                        except:
                            traceback.print_exc()
                            print (' - [ERROR][{}][generator()] Issue with transform: {}'.format(self.class_name, transform.name))
                else:
                    print ('')
                    print (' - [INFO][{}][generator()] No transformations available! - {}, {}'.format(self.class_name, self.dir_type, self.annotator_type))
                    print ('')

                # Step 6 - Return
                return dataset
            
            else:
                return None

        except:
            traceback.print_exc()
            pdb.set_trace()
            return None
    
    def _get_paths(self, idx):
        patient_id = ''
        study_id   = ''
        path_img, path_mask = '', ''

        if self.debug:
            path_img  = Path(self.paths_img[0]).absolute()
            path_mask = Path(self.paths_mask[0]).absolute()
            path_img, path_mask = self.path_debug_3D(path_img, path_mask)
        else:
            path_img  = Path(self.paths_img[idx]).absolute()
            path_mask = Path(self.paths_mask[idx]).absolute()

        if path_img.exists() and path_mask.exists():
            patient_id = Path(path_img).parts[-2]
            study_id   = Path(path_img).parts[-5] + '-' + Path(path_img).parts[-4]
        else:
            print (' - [ERROR] Issue with path')
            print (' -- [ERROR][{}] path_img : {}'.format(self.class_name, path_img))
            print (' -- [ERROR][{}] path_mask: {}'.format(self.class_name, path_mask))
        
        return path_img, path_mask, patient_id, study_id
    
    def _generator3D(self):

        # Step 0 - Init
        res = []

        # Step 1 - Get patient idxs
        idxs = np.arange(len(self.paths_img)).tolist()  #[:3]
        if self.single_sample: idxs = idxs[0:1] # [2:4]
        if self.patient_shuffle: np.random.shuffle(idxs)

        # Step 2 - Proceed on the basis of grid sampling or full-volume (self.grid=False) sampling
        if self.grid:
            
            # Step 2.1 - Get grid sampler info for each patient-idx
            sampler_info = {}
            for idx in idxs:
                path_img           = Path(self.paths_img[idx]).absolute() 
                patient_id         = path_img.parts[-2]
                
                if config.TYPE_VOXEL_RESAMPLED in str(path_img):
                    voxel_shape = self.patient_meta_info[patient_id][config.KEYNAME_SHAPE_RESAMPLED]    
                else:
                    voxel_shape = self.patient_meta_info[patient_id][config.KEYNAME_SHAPE_ORIG]

                if self.crop_init:
                    if patient_id in self.patients_z_prob:
                        voxel_shape[0] = self.voxel_shape_cropped[0]
                        voxel_shape[1] = self.voxel_shape_cropped[1]
                    else:
                        voxel_shape = self.voxel_shape_cropped

                grid_idxs_width   = utils.split_into_overlapping_grids(voxel_shape[0], len_grid=self.grid_size[0], len_overlap=self.grid_overlap[0])
                grid_idxs_height  = utils.split_into_overlapping_grids(voxel_shape[1], len_grid=self.grid_size[1], len_overlap=self.grid_overlap[1])   
                grid_idxs_depth   = utils.split_into_overlapping_grids(voxel_shape[2], len_grid=self.grid_size[2], len_overlap=self.grid_overlap[2])
                sampler_info[idx] = list(itertools.product(grid_idxs_width,grid_idxs_height,grid_idxs_depth))

            # Step 2.2 - Loop over all patients and their grids
            # Note - Grids of a patient are extracted in order
            for i, idx in enumerate(idxs):
                path_img, path_mask, patient_id, study_id = self._get_paths(idx)
                missing_labels                            = self.patient_meta_info[patient_id][config.KEYNAME_LABEL_MISSING]
                bgd_mask                                  = 1 # by default
                if len(missing_labels): 
                    bgd_mask = 0
                if path_img.exists() and path_mask.exists():
                    for sample_info in sampler_info[idx]:
                        grid_idxs = sample_info
                        meta1     = [idx] + [grid_idxs[0][0], grid_idxs[1][0], grid_idxs[2][0]]  # only include w_start, h_start, d_start
                        meta2     = '-'.join([self.name, study_id, patient_id + '_resample_' + str(self.resampled)])
                        path_img  = str(path_img)
                        path_mask = str(path_mask)
                        res.append((path_img, path_mask, meta1, meta2, bgd_mask))
        
        else:
            label_names = list(self.LABEL_MAP.keys())
            for i, idx in enumerate(idxs):
                path_img, path_mask, patient_id, study_id = self._get_paths(idx)
                missing_label_names = self.patient_meta_info[patient_id][config.KEYNAME_LABEL_MISSING]
                bgd_mask            = 1
                
                # if len(missing_labels): bgd_mask = 0
                if len(missing_label_names):
                    if len(set(label_names) - set(missing_label_names)): 
                        bgd_mask = 0
                    
                if path_img.exists() and path_mask.exists():
                    meta1 = [idx] + [0,0,0] # dummy for w_start, h_start, d_start
                    meta2 ='-'.join([self.name, study_id, patient_id + '_resample_' + str(self.resampled)])
                    path_img = str(path_img)
                    path_mask = str(path_mask)
                    res.append((path_img, path_mask, meta1, meta2, bgd_mask))

        # Step 3 - Yield
        for each in res:
            path_img, path_mask, meta1, meta2, bgd_mask = each
            
            vol_img_npy, vol_mask_npy, spacing = self._get_cache_item_old(path_img, path_mask)
            if vol_img_npy is None and vol_mask_npy is None:
                vol_img_npy, vol_mask_npy, spacing = self._get_volume_from_path(path_img, path_mask)    
                self._set_cache_item_old(path_img, path_mask, vol_img_npy, vol_mask_npy, spacing)
            
            spacing           = tf.constant(spacing, dtype=tf.int32)
            vol_img_npy_shape = tf.constant(vol_img_npy.shape, dtype=tf.int32)
            meta1             = tf.concat([meta1, spacing, vol_img_npy_shape, [bgd_mask]], axis=0) # [idx,[grid_idxs],[spacing],[shape]]

            yield (vol_img_npy, vol_mask_npy, meta1, meta2)
    
    def _get_cache_item_old(self, path_img, path_mask):
        if 'img' in self.cache and 'mask' in self.cache:
            if path_img in self.cache['img'] and path_mask in self.cache['mask']:
                # print (' - [_get_cache_item()] ')
                return self.cache['img'][path_img], self.cache['mask'][path_mask], self.cache['spacing']
            else:
                return None, None, None
        else:
            return None, None, None
    
    def _set_cache_item_old(self, path_img, path_mask, vol_img, vol_mask, spacing):
        
        self.cache = {
            'img': {path_img: vol_img}
            , 'mask': {path_mask: vol_mask}
            , 'spacing': spacing
        }
    
    def _set_cache_item(self, path_img, path_mask, vol_img, vol_mask, spacing):
        if len(self.cache) == 0:
            self.cache = {path_img: [vol_img, vol_mask, spacing]}
            self.cache_id = {path_img:0}
        elif len(self.cache) == 1:
            self.cache[path_img] = [vol_img, vol_mask, spacing]
            self.cache_id[path_img] = 1
        elif len(self.cache) == 2:
            max_order_id = max(self.cache_id.values())
            for path_img_ in self.cache_id:
                if self.cache_id[path_img_] == max_order_id - 1:
                    self.cache.pop(path_img_)
            self.cache[path_img] = [vol_img, vol_mask, spacing]
            self.cache_id[path_img] = max_order_id+1
    
    def _get_cache_item(self, path_img, path_mask):
        
        if path_img in self.cache:
            return self.cache[path_img]
        else:
            return None, None, None
    
    def _get_volume_from_path(self, path_img, path_mask, verbose=False):

        # Step 1 - Get volumes
        if verbose: t0 = time.time()
        vol_img_sitk   = utils.read_mha(path_img)
        vol_img_npy    = utils.sitk_to_array(vol_img_sitk)
        vol_mask_sitk  = utils.read_mha(path_mask)
        vol_mask_npy   = utils.sitk_to_array(vol_mask_sitk)     
        spacing        = np.array(vol_img_sitk.GetSpacing())

        # Step 2 - Perform init crop on volumes
        if self.crop_init:
            patient_id = str(Path(path_img).parts[-2])
            mean_point = np.array(self.patient_meta_info[patient_id][config.KEYNAME_MEAN_MIDPOINT]).astype(np.uint16).tolist()
            vol_img_npy_shape_prev = vol_img_npy.shape

            # Step 2.1 - Perform crops in H,W region
            vol_img_npy = vol_img_npy[
                    mean_point[0] - self.w_lateral_crop    : mean_point[0] + self.w_medial_crop
                    , mean_point[1] - self.h_anterior_crop : mean_point[1] + self.h_posterior_crop
                    , :
                ]
            vol_mask_npy = vol_mask_npy[
                    mean_point[0] - self.w_lateral_crop    : mean_point[0] + self.w_medial_crop
                    , mean_point[1] - self.h_anterior_crop : mean_point[1] + self.h_posterior_crop
                    , :
                ]
            
            # Step 2.2 - Perform crops in D region
            if self.grid:
                if self.VOXEL_RESO == (0.8,0.8,2.5):
                    vol_img_npy  = vol_img_npy[:,  :, mean_point[2] - self.d_caudal_crop : mean_point[2] + self.d_cranial_crop]
                    vol_mask_npy = vol_mask_npy[:, :, mean_point[2] - self.d_caudal_crop : mean_point[2] + self.d_cranial_crop]
                else:
                    vol_img_npy  = vol_img_npy[:,  :, mean_point[2] - self.d_caudal_crop : mean_point[2] + self.d_cranial_crop]
                    vol_mask_npy = vol_mask_npy[:, :, mean_point[2] - self.d_caudal_crop : mean_point[2] + self.d_cranial_crop]
            else:
                if self.resampled:
                    if self.VOXEL_RESO == (0.8,0.8,2.5):
                        vol_img_npy  = vol_img_npy[:,  :, mean_point[2] - self.d_caudal_crop : mean_point[2] + self.d_cranial_crop]
                        vol_mask_npy = vol_mask_npy[:, :, mean_point[2] - self.d_caudal_crop : mean_point[2] + self.d_cranial_crop]
                    else:
                        vol_img_npy  = vol_img_npy[:,  :, mean_point[2] - self.d_caudal_crop : mean_point[2] + self.d_cranial_crop]
                        vol_mask_npy = vol_mask_npy[:, :, mean_point[2] - self.d_caudal_crop : mean_point[2] + self.d_cranial_crop]
                else:
                    vol_img_npy  = vol_img_npy[:,  :, mean_point[2] - self.d_caudal_crop : mean_point[2] + self.d_cranial_crop]
                    vol_mask_npy = vol_mask_npy[:, :, mean_point[2] - self.d_caudal_crop : mean_point[2] + self.d_cranial_crop]
            
            # Step 3 - Pad (with=0) if volume is less in z-dimension
            (vol_img_npy_x, vol_img_npy_y, vol_img_npy_z) = vol_img_npy.shape
            if vol_img_npy_z < self.d_caudal_crop + self.d_cranial_crop:
                del_z = self.d_caudal_crop + self.d_cranial_crop - vol_img_npy_z
                vol_img_npy = np.concatenate((vol_img_npy, np.zeros((vol_img_npy_x, vol_img_npy_y, del_z))), axis=2)
                vol_mask_npy = np.concatenate((vol_mask_npy, np.zeros((vol_img_npy_x, vol_img_npy_y, del_z))), axis=2)
        
        if verbose: print (' - [HaNMICCAI2015Dataset._get_volume_from_path()] Time: ({}):{}s'.format(Path(path_img).parts[-2],  round(time.time() - t0,2)))        
        if self.pregridnorm:
            vol_img_npy[vol_img_npy <= self.HU_MIN] = self.HU_MIN
            vol_img_npy[vol_img_npy >= self.HU_MAX] = self.HU_MAX
            vol_img_npy                             = (vol_img_npy -np.mean(vol_img_npy))/np.std(vol_img_npy) #Standardize (z-scoring)
        
        return tf.cast(vol_img_npy, dtype=config.DATATYPE_TF_FLOAT32), tf.cast(vol_mask_npy, dtype=config.DATATYPE_TF_UINT8), tf.constant(spacing*100, dtype=config.DATATYPE_TF_INT32)
        
    @tf.function
    def _get_new_grid_idx(self, start, end, max):
        
        start_prev = start
        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.RANDOM_SHIFT_PERC:
            
            delta_left   = start
            delta_right  = max - end
            shift_voxels = tf.random.uniform([], minval=0, maxval=self.RANDOM_SHIFT_MAX, dtype=tf.dtypes.int32)

            if delta_left > self.RANDOM_SHIFT_MAX and delta_right > self.RANDOM_SHIFT_MAX:
                if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.RANDOM_SHIFT_PERC:
                    start = start - shift_voxels
                    end   = end - shift_voxels
                else:
                    start = start + shift_voxels
                    end   = end + shift_voxels

            elif delta_left > self.RANDOM_SHIFT_MAX and delta_right <= self.RANDOM_SHIFT_MAX:
                start = start - shift_voxels
                end   = end - shift_voxels

            elif delta_left <= self.RANDOM_SHIFT_MAX and delta_right > self.RANDOM_SHIFT_MAX:
                start = start + shift_voxels
                end   = end + shift_voxels

        return start_prev, start, end

    @tf.function
    def _get_new_grid_idx_centred(self, grid_size_half, max_pt, mid_pt):
        
        # Step 1 - Return vars
        start, end = 0,0

        # Step 2 - Define margin on either side of mid point
        margin_left = mid_pt
        margin_right = max_pt - mid_pt

        # Ste p2 - Calculate vars
        if margin_left >= grid_size_half and margin_right >= grid_size_half:
            start = mid_pt - grid_size_half
            end = mid_pt + grid_size_half
        elif margin_right < grid_size_half:
            if margin_left >= grid_size_half + (grid_size_half - margin_right): 
                end = mid_pt + margin_right
                start = mid_pt - grid_size_half - (grid_size_half - margin_right)
            else:
                tf.print(' - [ERROR][_get_new_grid_idx_centred()] Cond 2 problem')
        elif margin_left < grid_size_half:
            if margin_right >= grid_size_half + (grid_size_half - margin_left):
                start = mid_pt - margin_left
                end = mid_pt + grid_size_half + (grid_size_half-margin_left)
            else:
                tf.print(' - [ERROR][_get_new_grid_idx_centred()] Cond 3 problem')
        
        return start, end

    @tf.function
    def _get_data_3D(self, vol_img, vol_mask, meta1, meta2):
        """
        Params
        ------
        meta1: [idx, [w_start, h_start, d_start], [spacing_x, spacing_y, spacing_z], [shape_x, shape_y, shape_z], [bgd_mask]]
        """

        vol_img_npy = None
        vol_mask_npy = None

        # Step 1 - Proceed on the basis of grid sampling or full-volume (self.grid=False) sampling
        if self.grid:
            
            if 0: #tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.centred_dataloader_prob:

                # Step 1.1 - Get a label_id and its mean
                label_id           = tf.cast(tf.random.categorical(tf.math.log([self.LABEL_WEIGHTS]), 1), dtype=config.DATATYPE_TF_UINT8)[0][0]  # the LABEL_WEGIHTS sum to 1; the log is added as tf.random.categorical expects logits
                label_id_idxs      = tf.where(tf.math.equal(label_id, vol_mask))
                label_id_idxs_mean = tf.math.reduce_mean(label_id_idxs, axis=0)
                label_id_idxs_mean = tf.cast(label_id_idxs_mean, dtype=config.DATATYPE_TF_INT32)
                
                # Step 1.2 - Create a grid around that mid-point
                w_start_prev = meta1[1]
                h_start_prev = meta1[2]
                d_start_prev = meta1[3]
                w_max        = meta1[7]
                h_max        = meta1[8]
                d_max        = meta1[9]
                w_grid       = self.grid_size[0] 
                h_grid       = self.grid_size[1] 
                d_grid       = self.grid_size[2]
                w_mid        = label_id_idxs_mean[0]
                h_mid        = label_id_idxs_mean[1] 
                d_mid        = label_id_idxs_mean[2]

                w_start, w_end = self._get_new_grid_idx_centred(w_grid//2, w_max, w_mid)
                h_start, h_end = self._get_new_grid_idx_centred(h_grid//2, h_max, h_mid)
                d_start, d_end = self._get_new_grid_idx_centred(d_grid//2, d_max, d_mid)

                meta1_diff = tf.convert_to_tensor([0,w_start - w_start_prev, h_start - h_start_prev, d_start - d_start_prev,0,0,0,0,0,0,0])
                meta1      = meta1 + meta1_diff

            else:
                
                # tf.print(' - [INFO] regular dataloader: ', self.dir_type)
                # Step 1.1 - Get raw images/masks and extract grid
                w_start = meta1[1]
                w_end   = w_start + self.grid_size[0]
                h_start = meta1[2]
                h_end   = h_start + self.grid_size[1]
                d_start = meta1[3]
                d_end   = d_start + self.grid_size[2]

                # Step 1.2 - Randomization of grid 
                if self.random_grid:
                    w_max = meta1[7]
                    h_max = meta1[8]
                    d_max = meta1[9]

                    w_start_prev = w_start
                    d_start_prev = d_start
                    w_start_prev, w_start, w_end = self._get_new_grid_idx(w_start, w_end, w_max)
                    h_start_prev, h_start, h_end = self._get_new_grid_idx(h_start, h_end, h_max)
                    d_start_prev, d_start, d_end = self._get_new_grid_idx(d_start, d_end, d_max)

                    meta1_diff = tf.convert_to_tensor([0,w_start - w_start_prev, h_start - h_start_prev, d_start - d_start_prev,0,0,0,0,0,0,0])
                    meta1 = meta1 + meta1_diff
                
            # Step 1.3 - Extracting grid
            vol_img_npy  = tf.identity(vol_img[w_start:w_end, h_start:h_end, d_start:d_end])
            vol_mask_npy = tf.identity(vol_mask[w_start:w_end, h_start:h_end, d_start:d_end])
            
            
        else:
            vol_img_npy = vol_img
            vol_mask_npy = vol_mask

        # Step 2 - One-hot or not
        vol_mask_classes = []
        label_ids_mask = []
        label_ids = sorted(list(self.LABEL_MAP.values()))
        if self.mask_type == config.MASK_TYPE_ONEHOT:
            vol_mask_classes = tf.concat([tf.expand_dims(tf.math.equal(vol_mask_npy, label), axis=-1) for label in label_ids], axis=-1) # [H,W,D,L]
            for label_id in label_ids:
                label_ids_mask.append(tf.cast(tf.math.reduce_any(vol_mask_classes[:,:,:,label_id]), dtype=config.DATATYPE_TF_INT32))
            
        elif self.mask_type == config.MASK_TYPE_COMBINED:
            vol_mask_classes = vol_mask_npy
            unique_classes, _ = tf.unique(tf.reshape(vol_mask_npy,[-1]))
            unique_classes = tf.cast(unique_classes, config.DATATYPE_TF_INT32)
            for label_id in label_ids:
                label_ids_mask.append(tf.cast(tf.math.reduce_any(tf.math.equal(unique_classes, label_id)), dtype=config.DATATYPE_TF_INT32))    
        
        # Step 2.2 - Handling background mask explicitly if there is a missing label
        bgd_mask = meta1[-1]
        label_ids_mask[0] = bgd_mask
        meta1 = meta1[:-1] # removes the bgd mask index

        # Step 3 - Dtype conversion and expading dimensions            
        if self.mask_type == config.MASK_TYPE_ONEHOT:
            x = tf.cast(tf.expand_dims(vol_img_npy, axis=3), dtype=config.DATATYPE_TF_FLOAT32) # [H,W,D,1]
        else:
            x = tf.cast(vol_img_npy, dtype=config.DATATYPE_TF_FLOAT32) # [H,W,D]
        y = tf.cast(vol_mask_classes, dtype=config.DATATYPE_TF_FLOAT32) # [H,W,D,L]

        # Step 4 - Append info to meta1
        meta1 = tf.concat([meta1, label_ids_mask], axis=0)

        # Step 5 - return
        return (x, y, meta1, meta2)
        