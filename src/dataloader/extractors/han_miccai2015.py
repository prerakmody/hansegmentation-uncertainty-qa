# Import private libraries
import src.config as config
import src.dataloader.utils as utils

# Import public libraries
import pdb
import tqdm
import nrrd
import copy
import traceback
import numpy as np
from pathlib import Path


class HaNMICCAI2015Downloader:

    def __init__(self, dataset_dir_raw, dataset_dir_processed):
        self.dataset_dir_raw = dataset_dir_raw 
        self.dataset_dir_processed = dataset_dir_processed

    def download(self):
        self.dataset_dir_raw.mkdir(parents=True, exist_ok=True)
        # Step 1 - Download .zips and unzip them
        urls_zip = ['http://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part1.zip'
                    , 'http://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part2.zip'
                    , 'http://www.imagenglab.com/data/pddca/PDDCA-1.4.1_part3.zip']
        
        import concurrent
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for url_zip in urls_zip:
                filepath_zip = Path(self.dataset_dir_raw, url_zip.split('/')[-1])

                # Step 1.1 - Download .zip and then unzip it
                if not Path(filepath_zip).exists():
                    executor.submit(utils.download_zip, url_zip, filepath_zip, self.dataset_dir_raw)
                else:
                    executor.submit(utils.read_zip, filepath_zip, self.dataset_dir_raw)
                
                # Step 1.2 - Unzip .zip
                # executor.submit(utils.read_zip(filepath_zip, self.dataset_dir_raw)

    def sort(self, dataset_dir_datatypes, dataset_dir_datatypes_ranges):
        
        print ('')
        import tqdm
        import shutil
        import numpy as np

        # Step 1 - Make necessay directories
        self.dataset_dir_raw.mkdir(parents=True, exist_ok=True)
        for each in dataset_dir_datatypes:
            path_tmp = Path(self.dataset_dir_raw).joinpath(each)
            path_tmp.mkdir(parents=True, exist_ok=True)

        # Step 2 - Sort
        with tqdm.tqdm(total=len(list(Path(self.dataset_dir_raw).glob('0522*'))), desc='Sorting', leave=False) as pbar:
            for path_patient in self.dataset_dir_raw.iterdir():
                if '.zip' not in path_patient.parts[-1]: #and path_patient.parts[-1] not in dataset_dir_datatypes:
                    try:
                        patient_number = Path(path_patient).parts[-1][-3:]
                        if patient_number.isdigit():
                            folder_id = np.digitize(patient_number, dataset_dir_datatypes_ranges)
                            shutil.move(src=str(path_patient), dst=str(Path(self.dataset_dir_raw).joinpath(dataset_dir_datatypes[folder_id])))
                            pbar.update(1)
                    except:
                        traceback.print_exc()
                        pdb.set_trace()

class HaNMICCAI2015Extractor:
    """
    More information on the .nrrd format can be found here: http://teem.sourceforge.net/nrrd/format.html#space
    """

    def __init__(self, name, dataset_dir_raw, dataset_dir_processed, dataset_dir_datatypes):
        
        self.name                  = name
        self.dataset_dir_raw       = dataset_dir_raw 
        self.dataset_dir_processed = dataset_dir_processed 
        self.dataset_dir_datatypes = dataset_dir_datatypes
        self.folder_prefix         = '0522'
        
        self._preprint()
        self._init_constants()
    
    def _preprint(self):
        self.VOXEL_RESO = getattr(config, self.name)[config.KEY_VOXELRESO]
        print ('')
        print (' - [HaNMICCAI2015Extractor] VOXEL_RESO: ', self.VOXEL_RESO)
        print ('')
        
    def _init_constants(self):
        
        # File names
        self.DATATYPE_ORIG       = '.nrrd'
        self.IMG_VOXEL_FILENAME  = 'img.nrrd'
        self.MASK_VOXEL_FILENAME = 'mask.nrrd'

        # Label information
        self.LABEL_MAP        = getattr(config, self.name)[config.KEY_LABEL_MAP]
        self.IGNORE_LABELS    = getattr(config,self.name)[config.KEY_IGNORE_LABELS]
        self.LABELID_MIDPOINT = getattr(config, self.name)[config.KEY_LABELID_MIDPOINT]
    
    def extract3D(self):
        
        import concurrent
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            for dir_type in self.dataset_dir_datatypes:
                executor.submit(self._extract3D_patients, Path(self.dataset_dir_raw).joinpath(dir_type))
        # for dir_type in ['train']:
        #     self._extract3D_patients(Path(self.dataset_dir_raw).joinpath(dir_type))
        
        print ('')
        print (' - Note: You can view the 3D data in visualizers like MeVisLab or 3DSlicer')
        print ('')
    
    def _extract3D_patients(self, dir_dataset):

        dir_type                = Path(dir_dataset).parts[-1]
        paths_global_voxel_img  = []
        paths_global_voxel_mask = []

        # Step 1 - Loop over patients of dir_type and get their img and mask paths
        dir_type_idx = self.dataset_dir_datatypes.index(dir_type)
        with tqdm.tqdm(total=len(list(dir_dataset.glob('*'))), desc='[3D][{}] Patients: '.format(dir_type), disable=False, position=dir_type_idx) as pbar:
            for _, patient_dir_path in enumerate(dir_dataset.iterdir()):
                try:
                    if Path(patient_dir_path).is_dir():
                        voxel_img_filepath, voxel_mask_filepath, _ = self._extract3D_patient(patient_dir_path)
                        paths_global_voxel_img.append(voxel_img_filepath)
                        paths_global_voxel_mask.append(voxel_mask_filepath)
                        pbar.update(1)
                    
                except:
                    print ('')
                    print (' - [ERROR][HaNMICCAI2015Extractor] Error with patient_id: ', Path(patient_dir_path).parts[-2:])
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
            print (' - [ERROR][HaNMICCAI2015Extractor] Unable to save .csv')
            pdb.set_trace()
            print (' - Exiting!')
            import sys; sys.exit(1)
    
    def _extract3D_patient(self, patient_dir):

        try:
            voxel_img, voxel_img_headers, voxel_mask, voxel_mask_headers = self._get_data3D(patient_dir) 

            dir_type = Path(patient_dir).parts[-2]
            patient_id = Path(patient_dir).parts[-1]
            return self._save_data3D(dir_type, patient_id, voxel_img, voxel_img_headers, voxel_mask, voxel_mask_headers)

        except:
            print (' - [ERROR][_extract_patient()] path_folder: ', patient_dir.parts[-1])
            traceback.print_exc()
            pdb.set_trace()

    def _get_data3D(self, patient_dir):
        try:
            voxel_img, voxel_mask = [], []
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
                print (' - Error: Path does not exist: patient_dir', patient_dir)

            return voxel_img, voxel_img_headers, voxel_mask, voxel_mask_headers

        except:
            print (' - [ERROR][get_data()] patient_dir: ', patient_dir.parts[-1])
            traceback.print_exc()
            pdb.set_trace()
    
    def _get_voxel_img(self, path_voxel, histogram=False):
        try:
            if path_voxel.exists():
                voxel_img_data, voxel_img_header = nrrd.read(str(path_voxel))  # shape=[H,W,D]

                if histogram:
                    import matplotlib.pyplot as plt
                    plt.hist(voxel_img_data.flatten())
                    plt.show()

                return voxel_img_data, voxel_img_header
            else:
                print (' - Error: Path does not exist: ', path_voxel)
        except:
            traceback.print_exc()
            pdb.set_trace()
    
    def _get_voxel_mask(self, path_voxel_mask):
        try:

            # Step 1 - Get mask data and headers
            if Path(path_voxel_mask).exists():
                voxel_mask_data, voxel_mask_headers = nrrd.read(str(path_voxel_mask)) 
                
            else:
                path_mask_folder = Path(*Path(path_voxel_mask).parts[:-1]).joinpath('structures')
                voxel_mask_data, voxel_mask_headers = self._merge_masks(path_mask_folder)

            # Step 2 - Make a list of available headers
            voxel_mask_headers = dict(voxel_mask_headers)
            voxel_mask_headers[config.KEYNAME_LABEL_OARS] = []
            voxel_mask_headers[config.KEYNAME_LABEL_MISSING] = []
            label_map_inverse = {label_id: label_name for label_name, label_id in self.LABEL_MAP.items()}
            label_ids_all = self.LABEL_MAP.values()
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
            voxel_mask_full = []
            voxel_mask_headers = {}
            labels_oars = []
            labels_missing = []
            if Path(path_mask_folder).exists():
                with tqdm.tqdm(total=len(list(Path(path_mask_folder).glob('*{}'.format(self.DATATYPE_ORIG)))), leave=False, disable=True) as pbar_mask:
                    for filepath_mask in Path(path_mask_folder).iterdir():
                        class_name = Path(filepath_mask).parts[-1].split(self.DATATYPE_ORIG)[0]
                        class_id = -1
                        
                        if class_name in self.LABEL_MAP:
                            class_id = self.LABEL_MAP[class_name]
                            labels_oars.append(class_name)
                            voxel_mask, voxel_mask_headers = nrrd.read(str(filepath_mask))
                        # else:
                        #     print (' - [ERROR][HaNMICCAI2015Extractor][_merge_masks] Unknown class name: ', class_name)

                        if class_id not in self.IGNORE_LABELS and class_id > 0:
                            if len(voxel_mask_full) == 0: 
                                voxel_mask_full = copy.deepcopy(voxel_mask)    
                            idxs = np.argwhere(voxel_mask > 0)
                            voxel_mask_full[idxs[:,0], idxs[:,1], idxs[:,2]] = class_id
                            if 0:
                                print (' - [merge_masks()] class_id:', class_id, ' || name: ', class_name, ' || idxs: ', len(idxs))
                                print (' --- [merge_masks()] label_ids: ', np.unique(voxel_mask_full))

                        pbar_mask.update(1)
                        
                path_mask = Path(*Path(path_mask_folder).parts[:-1]).joinpath(self.MASK_VOXEL_FILENAME)
                nrrd.write(str(path_mask), voxel_mask_full, voxel_mask_headers)
                        
            else:
                print (' - Error with path_mask_folder: ', path_mask_folder)
        
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
                
                # Find average HU value in the brainstem
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
