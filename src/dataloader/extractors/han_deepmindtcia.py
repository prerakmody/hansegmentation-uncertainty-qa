# Import private libraries
import medloader.dataloader.config as config
import medloader.dataloader.utils as utils

# Import public libraries
import pdb
import tqdm
import nrrd
import copy
import traceback
import numpy as np
from pathlib import Path

class HaNDeepMindTCIADownloader:

    def __init__(self, dataset_dir_raw, dataset_dir_processed):
        
        self.class_name            = 'HaNDeepMindTCIADownloader' 
        self.dataset_dir_raw       = dataset_dir_raw 
        self.dataset_dir_processed = dataset_dir_processed

        self.url_zip               = 'https://github.com/deepmind/tcia-ct-scan-dataset/archive/refs/heads/master.zip'
        self.unzipped_folder_name  = self.url_zip.split('/')[-5] + '-' + self.url_zip.split('/')[-1].split('.')[0]

    def download(self):
        
        # Step 1 - Make raw directory
        self.dataset_dir_raw.mkdir(parents=True, exist_ok=True)

        # Step 2 - Download .zip and then unzip it
        path_zip_folder = Path(self.dataset_dir_raw, self.unzipped_folder_name + '.zip')

        if not Path(path_zip_folder).exists():
            utils.download_zip(self.url_zip, path_zip_folder, self.dataset_dir_raw)
        else:
            utils.read_zip(path_zip_folder, self.dataset_dir_raw)
    
    def sort(self):

        path_download_nrrds = Path(self.dataset_dir_raw).joinpath(self.unzipped_folder_name, 'nrrds')
        if Path(path_download_nrrds).exists():
            path_download_nrrds_test_src = Path(path_download_nrrds).joinpath('test')
            path_download_nrrds_val_src  = Path(path_download_nrrds).joinpath('validation')

            path_nrrds_test_dest = Path(self.dataset_dir_raw).joinpath('test')
            path_nrrds_val_dest  = Path(self.dataset_dir_raw).joinpath('validation')

            utils.move_folder(path_download_nrrds_test_src, path_nrrds_test_dest)
            utils.move_folder(path_download_nrrds_val_src, path_nrrds_val_dest)
        else:
            print (' - [ERROR][{}] Could not find nrrds folder: {}'.format(self.class_name, path_download_nrrds))
    

class HaNDeepMindTCIAExtractor:

    def __init__(self, name, dataset_dir_raw, dataset_dir_processed, dataset_dir_datatypes):
        
        self.name                  = name
        self.class_name            = 'HaNDeepMindTCIAExtractor'
        self.dataset_dir_raw       = dataset_dir_raw 
        self.dataset_dir_processed = dataset_dir_processed 
        self.dataset_dir_datatypes = dataset_dir_datatypes
    
        self._preprint()
        self._init_constants()
    
    def _preprint(self):
        self.VOXEL_RESO = getattr(config, self.name)[config.KEY_VOXELRESO]
        print ('')
        print (' - [{}] VOXEL_RESO: {}'.format(self.class_name, self.VOXEL_RESO))
        print ('')
    
    def _init_constants(self):
        
        # File names
        self.DATATYPE_ORIG       = '.nrrd'
        self.IMG_VOXEL_FILENAME  = 'CT_IMAGE.nrrd'
        self.MASK_VOXEL_FILENAME = 'mask.nrrd'
        self.MASK_ORGANS_FOLDERNAME = 'segmentations'

        # Label information
        self.dataset_config = getattr(config, self.name)
        self.LABEL_MAP_MICCAI2015_DEEPMINDTCIA = self.dataset_config[config.KEY_LABELMAP_MICCAI_DEEPMINDTCIA]
        self.LABEL_MAP                         = self.dataset_config[config.KEY_LABEL_MAP]
        self.IGNORE_LABELS                     = self.dataset_config[config.KEY_IGNORE_LABELS]
        self.LABELID_MIDPOINT                  = self.dataset_config[config.KEY_LABELID_MIDPOINT]
    
    def extract3D(self):
        
        if 1:
            import concurrent
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as executor:
                for dir_type in self.dataset_dir_datatypes:
                    path_extract3D = Path(self.dataset_dir_raw).joinpath(dir_type)
                    if Path(path_extract3D).exists():
                        executor.submit(self._extract3D_patients, path_extract3D)
                    else:
                        print (' - [ERROR][{}][extract3D()] {} does not exist: '.format(self.class_name, path_extract3D))
                    
        else:
            for dir_type in [Path(config.DATALOADER_DEEPMINDTCIA_VAL, config.DATALOADER_DEEPMINDTCIA_ONC)]:
                path_extract3D = Path(self.dataset_dir_raw).joinpath(dir_type)
                if Path(path_extract3D).exists():
                    self._extract3D_patients(path_extract3D)
                else:
                    print (' - [ERROR][{}][extract3D()] {} does not exist: '.format(self.class_name, path_extract3D))
        
        print ('')
        print (' - Note: You can view the 3D data in visualizers like MeVisLab or 3DSlicer')
        print ('')
    
    def _extract3D_patients(self, dir_dataset):

        dir_type                = Path(Path(dir_dataset).parts[-2], Path(dir_dataset).parts[-1])
        paths_global_voxel_img  = []
        paths_global_voxel_mask = []

        # Step 1 - Loop over patients of dir_type and get their img and mask paths
        with tqdm.tqdm(total=len(list(dir_dataset.glob('*'))), desc='[3D][{}] Patients: '.format(str(dir_type)), disable=False) as pbar:
            for _, patient_dir_path in enumerate(dir_dataset.iterdir()):
                try:
                    if Path(patient_dir_path).is_dir():
                        voxel_img_filepath, voxel_mask_filepath, _ = self._extract3D_patient(patient_dir_path)
                        paths_global_voxel_img.append(voxel_img_filepath)
                        paths_global_voxel_mask.append(voxel_mask_filepath)
                        pbar.update(1)
                    
                except:
                    print ('')
                    print (' - [ERROR][{}][_extract3D_patients()] Error with patient_id: {}'.format(self.class_name, Path(patient_dir_path).parts[-3:]))
                    traceback.print_exc()
                    pdb.set_trace()
        
        # Step 2 - Save paths in .csvs 
        if len(paths_global_voxel_img) and len(paths_global_voxel_mask):
            paths_global_voxel_img = list(map(lambda x: str(x), paths_global_voxel_img))
            paths_global_voxel_mask = list(map(lambda x: str(x), paths_global_voxel_mask))
            utils.save_csv(Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_3D, config.FILENAME_CSV_IMG), paths_global_voxel_img)
            utils.save_csv(Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_3D, config.FILENAME_CSV_MASK), paths_global_voxel_mask)
            
            if len(self.VOXEL_RESO):
                paths_global_voxel_img_resampled = []
                for path_global_voxel_img in paths_global_voxel_img:
                    path_global_voxel_img_parts     = list(Path(path_global_voxel_img).parts)
                    patient_id                      = path_global_voxel_img_parts[-1].split('_')[-1].split('.')[0]
                    path_global_voxel_img_parts[-1] = config.FILENAME_IMG_RESAMPLED_3D.format(patient_id)
                    path_global_voxel_img_resampled = Path(*path_global_voxel_img_parts)
                    paths_global_voxel_img_resampled.append(path_global_voxel_img_resampled)
                
                paths_global_voxel_mask_resampled = []
                for path_global_voxel_mask in paths_global_voxel_mask:
                    path_global_voxel_mask_parts     = list(Path(path_global_voxel_mask).parts)
                    patient_id                       = path_global_voxel_mask_parts[-1].split('_')[-1].split('.')[0]
                    path_global_voxel_mask_parts[-1] = config.FILENAME_MASK_RESAMPLED_3D.format(patient_id)
                    path_global_voxel_mask_resampled = Path(*path_global_voxel_mask_parts)
                    paths_global_voxel_mask_resampled.append(path_global_voxel_mask_resampled)

                utils.save_csv(Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_3D, config.FILENAME_CSV_IMG_RESAMPLED), paths_global_voxel_img_resampled)
                utils.save_csv(Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_3D, config.FILENAME_CSV_MASK_RESAMPLED), paths_global_voxel_mask_resampled)
                
        else:
            print (' - [ERROR][{}][_extract3D_patients()] Unable to save .csv'.format(self.class_name))
            pdb.set_trace()
            print (' - Exiting!')
            import sys; sys.exit(1)
    
    def _extract3D_patient(self, patient_dir):

        try:
            voxel_img, voxel_img_headers, voxel_mask, voxel_mask_headers = self._get_data3D(patient_dir) 

            dir_type   = Path(Path(patient_dir).parts[-3], Path(patient_dir).parts[-2])
            patient_id = Path(patient_dir).parts[-1]
            return self._save_data3D(dir_type, patient_id, voxel_img, voxel_img_headers, voxel_mask, voxel_mask_headers)

        except:
            print (' - [ERROR][{}][_extract_patient()] path_folder: {}'.format(self.class_name, patient_dir.parts[-3:]))
            traceback.print_exc()
            pdb.set_trace()

    def _get_data3D(self, patient_dir):
        
        try:
            voxel_img, voxel_mask                 = [], []
            voxel_img_headers, voxel_mask_headers = {}, {}

            if Path(patient_dir).exists():
                
                if Path(patient_dir).is_dir():

                    # Step 1 - Get Voxel Data
                    path_voxel_img = Path(patient_dir).joinpath(self.IMG_VOXEL_FILENAME)
                    voxel_img, voxel_img_headers = self._get_voxel_img(path_voxel_img)

                    # Step 2 - Get Mask Data
                    path_voxel_mask = Path(patient_dir).joinpath(self.MASK_VOXEL_FILENAME)
                    voxel_mask, voxel_mask_headers = self._get_voxel_mask(path_voxel_mask)
                
            else:
                print (' - [ERROR][{}][get_data()]: Path does not exist: patient_dir: {}'.format(self.class_name, patient_dir))

            return voxel_img, voxel_img_headers, voxel_mask, voxel_mask_headers

        except:
            print (' - [ERROR][{}][get_data()] patient_dir: '.format(self.class_name, patient_dir.parts[-3:]))
            traceback.print_exc()
            pdb.set_trace()
    
    def _get_voxel_img(self, path_voxel, histogram=False):

        try:
            if Path(path_voxel).exists():
                voxel_img_data, voxel_img_header = nrrd.read(str(path_voxel))  # shape=[H,W,D]

                if histogram:
                    import matplotlib.pyplot as plt
                    plt.hist(voxel_img_data.flatten())
                    plt.show()

                return voxel_img_data, voxel_img_header
            else:
                print (' - [ERROR][{}][_get_voxel_img()]: Path does not exist: {}'.format(self.class_name, path_voxel))
        except:
            traceback.print_exc()
            pdb.set_trace()
    
    def _get_voxel_mask(self, path_voxel_mask):
        
        try:

            # Step 1 - Get mask data and headers
            if 0: #Path(path_voxel_mask).exists():
                voxel_mask_data, voxel_mask_headers = nrrd.read(str(path_voxel_mask)) 
                
            else:
                path_mask_folder = Path(*Path(path_voxel_mask).parts[:-1]).joinpath(self.MASK_ORGANS_FOLDERNAME)
                voxel_mask_data, voxel_mask_headers = self._merge_masks(path_mask_folder)

            # Step 2 - Make a list of available headers
            voxel_mask_headers = dict(voxel_mask_headers)
            voxel_mask_headers[config.KEYNAME_LABEL_OARS] = []
            voxel_mask_headers[config.KEYNAME_LABEL_MISSING] = []
            label_map_inverse    = {label_id: label_name for label_name, label_id in self.LABEL_MAP.items()}
            label_ids_all        = self.LABEL_MAP.values()
            label_ids_voxel_mask = np.unique(voxel_mask_data)
            for label_id in label_ids_all:
                label_name = label_map_inverse[label_id]
                if label_id not in label_ids_voxel_mask:
                    voxel_mask_headers[config.KEYNAME_LABEL_MISSING].append(label_name)
                else:
                    voxel_mask_headers[config.KEYNAME_LABEL_OARS].append(label_name)

            return voxel_mask_data, voxel_mask_headers

        except:
            traceback.print_exc()
            pdb.set_trace()
    
    def _merge_masks(self, path_mask_folder):
        
        try:
            voxel_mask_full    = []
            voxel_mask_headers = {}
            labels_oars        = []
            labels_missing     = []
            patient_id         = Path(path_mask_folder).parts[-2]

            if Path(path_mask_folder).exists():
                with tqdm.tqdm(total=len(list(Path(path_mask_folder).glob('*{}'.format(self.DATATYPE_ORIG)))), leave=False, disable=True) as pbar_mask:
                    for filepath_mask in Path(path_mask_folder).iterdir():
                        class_name            = Path(filepath_mask).parts[-1].split(self.DATATYPE_ORIG)[0]
                        class_name_miccai2015 = self.LABEL_MAP_MICCAI2015_DEEPMINDTCIA.get(class_name, None)    
                        class_id = -1
                        
                        if class_name_miccai2015 in self.LABEL_MAP:
                            class_id = self.LABEL_MAP[class_name_miccai2015]
                            labels_oars.append(class_name_miccai2015)
                            voxel_mask, voxel_mask_headers = nrrd.read(str(filepath_mask))

                        if class_id not in self.IGNORE_LABELS and class_id > 0:
                            if len(voxel_mask_full) == 0: 
                                voxel_mask_full = np.array(voxel_mask, copy=True)    
                            idxs = np.argwhere(voxel_mask > 0)
                            voxel_mask_full[idxs[:,0], idxs[:,1], idxs[:,2]] = class_id
                            if 0:
                                print (' - [merge_masks()] class_id:', class_id, ' || name: ', class_name_miccai2015, ' || idxs: ', len(idxs))
                                print (' --- [merge_masks()] label_ids: ', np.unique(voxel_mask_full))

                        pbar_mask.update(1)
                        
                path_mask = Path(*Path(path_mask_folder).parts[:-1]).joinpath(self.MASK_VOXEL_FILENAME)
                nrrd.write(str(path_mask), voxel_mask_full, voxel_mask_headers)
                        
            else:
                print (' - [ERROR][{}][_merge_masks()] path_mask_folder: {} does not exist '.format(self.class_name, path_mask_folder))
        
            return voxel_mask_full, voxel_mask_headers
        
        except:
            traceback.print_exc()
            pdb.set_trace()
    
    def _save_data3D(self, dir_type, patient_id, voxel_img, voxel_img_headers, voxel_mask, voxel_mask_headers):
        
        try:
            
            # Step 1 - Create directory
            voxel_savedir = Path(self.dataset_dir_processed).joinpath(dir_type, config.DIRNAME_SAVE_3D)
            
            # Step 2.1 - Save voxel
            voxel_img_headers_new = {config.TYPE_VOXEL_ORIGSHAPE:{}}
            voxel_img_headers_new[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_PIXEL_SPACING] = voxel_img_headers[config.KEY_NRRD_PIXEL_SPACING][voxel_img_headers[config.KEY_NRRD_PIXEL_SPACING] > 0].tolist()
            voxel_img_headers_new[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_ORIGIN]        = voxel_img_headers[config.KEY_NRRD_ORIGIN].tolist()
            voxel_img_headers_new[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_SHAPE]         = voxel_img_headers[config.KEY_NRRD_SHAPE].tolist()
            voxel_img_headers_new[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_OTHERS]        = voxel_img_headers
            voxel_img_headers_new[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_LABEL_MISSING] = voxel_mask_headers[config.KEYNAME_LABEL_MISSING]
            voxel_img_headers_new[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_LABEL_OARS]    = voxel_mask_headers[config.KEYNAME_LABEL_OARS]
            
            if len(voxel_img) and len(voxel_mask):
                resample_save = False
                if len(self.VOXEL_RESO):
                    resample_save = True
                
                # Find midpoint 3D coord of self.LABELID_MIDPOINT
                meanpoint_idxs      = np.argwhere(voxel_mask == self.LABELID_MIDPOINT)
                meanpoint_idxs_mean = np.mean(meanpoint_idxs, axis=0)
                voxel_img_headers_new[config.TYPE_VOXEL_ORIGSHAPE][config.KEYNAME_MEAN_MIDPOINT] = meanpoint_idxs_mean.tolist()

                return utils.save_as_mha(voxel_savedir, patient_id, voxel_img, voxel_img_headers_new, voxel_mask
                        , labelid_midpoint=self.LABELID_MIDPOINT
                        , resample_spacing=self.VOXEL_RESO)
            
            else:
                print (' - [ERROR][HaNMICCAI2015Extractor] Error with patient_id: ', patient_id)
                
        except:
            traceback.print_exc()
            pdb.set_trace()