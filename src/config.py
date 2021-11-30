############################################################
#                            INIT                          #
############################################################

from pathlib import Path
PROJECT_DIR = Path(__file__).parent.absolute().parent.absolute()
MAIN_DIR = Path(PROJECT_DIR).parent.absolute()

import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3" 
os.environ["TF_GPU_THREAD_MODE"] = "gpu_private" # to avoid large "Kernel Launch Time"

import tensorflow as tf
try:
    if len(tf.config.list_physical_devices('GPU')):
        tf.config.experimental.set_memory_growth(tf.config.list_physical_devices('GPU')[0], True)
        sys_details = tf.sysconfig.get_build_info()
        print (' - [TFlow Build Info] ver: ', tf.__version__, 'CUDA(major.minor):',  sys_details["cuda_version"], ' || cuDNN(major): ', sys_details["cudnn_version"])

    else:
        print (' - No GPU present!! Exiting ...')
        import sys; sys.exit(1)
except:
    pass

############################################################
#                    MODEL RELATED                         #
############################################################
MODEL_CHKPOINT_MAINFOLDER  = '_models'
MODEL_CHKPOINT_NAME_FMT    = 'ckpt_epoch{:03d}'
MODEL_LOGS_FOLDERNAME      = 'logs' 
MODEL_IMGS_FOLDERNAME      = 'images'
MODEL_PATCHES_FOLDERNAME   = 'patches'

EXT_NRRD = '.nrrd'

MODE_TRAIN     = 'Train'
MODE_TRAIN_VAL = 'Train_val'
MODE_VAL       = 'Val'
MODE_VAL_NEW   = 'Val_New'
MODE_TEST      = 'Test'
MODE_DEEPMINDTCIA_TEST_ONC = 'DeepMindTCIATestOnc'
MODE_DEEPMINDTCIA_TEST_RAD = 'DeepMindTCIATestRad'
MODE_DEEPMINDTCIA_VAL_ONC  = 'DeepMindTCIAValOnc'
MODE_DEEPMINDTCIA_VAL_RAD  = 'DeepMindTCIAValRad'

ACT_SIGMOID = 'sigmoid'
ACT_SOFTMAX = 'softmax'

MODEL_FOCUSNET_DROPOUT       = 'ModelFocusNetDropOut'
MODEL_FOCUSNET_FLIPOUT       = 'ModelFocusNetFlipOut'

OPTIMIZER_ADAM = 'Adam'

THRESHOLD_SIGMA_IGNORE = 0.3
MIN_SIZE_COMPONENT = 10

KL_DIV_FIXED     = 'fixed'
KL_DIV_ANNEALING = 'annealing'

############################################################
#                      EVAL RELATED                        #
############################################################
KEY_DICE_AVG    = 'dice_avg'
KEY_DICE_LABELS = 'dice_labels'
KEY_HD_AVG      = 'hd_avg'
KEY_HD_LABELS   = 'hd_labels'
KEY_HD95_AVG    = 'hd95_avg'
KEY_HD95_LABELS = 'hd95_labels'
KEY_MSD_AVG     = 'msd_avg'
KEY_MSD_LABELS  = 'msd_labels'
KEY_ECE_AVG     = 'ece_avg'
KEY_ECE_LABELS  = 'ece_labels'
KEY_AVU_ENT       = 'avu_ent'
KEY_AVU_PAC_ENT   = 'avu_pac_ent'
KEY_AVU_PUI_ENT   = 'avu_pui_ent'
KEY_THRESH_ENT    = 'avu_thresh_ent'
KEY_AVU_MIF       = 'avu_mif'
KEY_AVU_PAC_MIF   = 'avu_pac_mif'
KEY_AVU_PUI_MIF   = 'avu_pui_mif'
KEY_THRESH_MIF    = 'avu_thresh_mif'
KEY_AVU_UNC       = 'avu_unc'
KEY_AVU_PAC_UNC   = 'avu_pac_unc'
KEY_AVU_PUI_UNC   = 'avu_pui_unc'
KEY_THRESH_UNC    = 'avu_thresh_unc'

PAVPU_UNC_THRESHOLD = 'adaptive-median' # [0.3, 'adaptive', 'adaptive-median']
PAVPU_ENT_THRESHOLD  = 0.5
PAVPU_MIF_THRESHOLD  = 0.1
PAVPU_GRID_SIZE     = (4,4,2) 
PAVPU_RATIO_NEG     = 0.9

KEY_ENT  = 'ent'
KEY_MIF  = 'mif'
KEY_STD  = 'std'
KEY_PERC = 'perc'

KEY_SUM  = 'sum'
KEY_AVG  = 'avg'
 
CMAP_MAGMA = 'magma'
CMAP_GRAY  = 'gray'

FILENAME_EVAL3D_JSON     = 'res.json'

FOLDERNAME_TMP = '_tmp'
FOLDERNAME_TMP_BOKEH = 'bokeh-plots'
FOLDERNAME_TMP_ENTMIF = 'entmif'

VAL_ECE_NAN = -0.1
VAL_DICE_NAN = -1.0
VAL_MC_RUNS_DEFAULT = 20

KEY_PATIENT_GLOBAL = 'global'

SUFFIX_DET = '-Det'
SUFFIX_MC  = '-MC{}'

KEY_MC_RUNS = 'MC_RUNS'
KEY_TRAINING_BOOL = 'training_bool'

############################################################
#                      LOSSES RELATED                      #
############################################################
LOSS_DICE     = 'Dice'
LOSS_CE       = 'CE'
LOSS_CE_BASIC = 'CE-Basic'

############################################################
#                   DATALOADER RELATED                     #
############################################################

HEAD_AND_NECK = 'HaN'
PROSTATE = 'Prostrate'
THORACIC = 'Thoracic'

DIRNAME_PROCESSED         = 'processed'
DIRNAME_PROCESSED_SPACING = 'processed_{}'
DIRNAME_RAW               = 'raw'
DIRNAME_SAVE_3D           = 'data_3D'

FILENAME_JSON_IMG = 'img.json'
FILENAME_JSON_MASK = 'mask.json'
FILENAME_JSON_IMG_RESAMPLED = 'img_resampled.json'
FILENAME_JSON_MASK_RESAMPLED = 'mask_resampled.json'
FILENAME_CSV_IMG = 'img.csv'
FILENAME_CSV_MASK = 'mask.csv'
FILENAME_CSV_IMG_RESAMPLED = 'img_resampled.csv'
FILENAME_CSV_MASK_RESAMPLED = 'mask_resampled.csv'

FILENAME_VOXEL_INFO = 'voxelinfo.json'

KEYNAME_LABEL_OARS     = 'labels_oars'
KEYNAME_LABEL_EXTERNAL = 'labels_external'
KEYNAME_LABEL_TUMORS   = 'labels_tumors'
KEYNAME_LABEL_MISSING  = 'labels_missing'

DATAEXTRACTOR_WORKERS = 8

import itk
import numpy as np
import tensorflow as tf
import SimpleITK as sitk

DATATYPE_VOXEL_IMG = np.int16
DATATYPE_VOXEL_MASK = np.uint8

DATATYPE_SITK_VOXEL_IMG      = sitk.sitkInt16
DATATYPE_SITK_VOXEL_MASK     = sitk.sitkUInt8

DATATYPE_ITK_VOXEL_MASK      = itk.UC

DATATYPE_NP_INT32 = np.int32

DATATYPE_TF_STRING = tf.string
DATATYPE_TF_UINT8 = tf.uint8
DATATYPE_TF_INT16 = tf.int16
DATATYPE_TF_INT32 = tf.int32
DATATYPE_TF_FLOAT32 = tf.float32

DUMMY_LABEL = 255

# Keys - Volume params
KEYNAME_PIXEL_SPACING     = 'pixel_spacing'
KEYNAME_ORIGIN            = 'origin'
KEYNAME_SHAPE             = 'shape'
KEYNAME_INTERCEPT         = 'intercept'
KEYNAME_SLOPE             = 'slope'
KEYNAME_ZVALS             = 'z_vals'
KEYNAME_MEAN_MIDPOINT     = 'mean_midpoint'
KEYNAME_OTHERS            = 'others'
KEYNAME_INTERPOLATOR      = 'interpolator'
KEYNAME_INTERPOLATOR_IMG  = 'interpolator_img'
KEYNAME_INTERPOLATOR_MASK = 'interpolator_mask'
KEYNAME_SHAPE_ORIG        = 'shape_orig'
KEYNAME_SHAPE_RESAMPLED   = 'shape_resampled'

# Keys - Dataset params
KEY_VOXELRESO          = 'VOXEL_RESO'
KEY_LABEL_MAP          = 'LABEL_MAP'
KEY_LABEL_MAP_FULL     = 'LABEL_MAP_FULL'
KEY_LABEL_MAP_EXTERNAL = 'LABEL_MAP_EXTERNAL'
KEY_LABEL_COLORS       = 'LABEL_COLORS'
KEY_LABEL_WEIGHTS      = 'LABEL_WEIGHTS'
KEY_IGNORE_LABELS      = 'IGNORE_LABELS'
KEY_LABELID_BACKGROUND = 'LABELID_BACKGROUND'
KEY_LABELID_MIDPOINT   = 'LABELID_MIDPOINT' 
KEY_HU_MIN             = 'HU_MIN'
KEY_HU_MAX             = 'HU_MAX'
KEY_PREPROCESS         = 'PREPROCESS'
KEY_CROP               = 'CROP'
KEY_GRID_3D            = 'GRID_3D'

# Common Labels
LABELNAME_BACKGROUND = 'Background'

# Keys - Cropping
KEY_MIDPOINT_EXTENSION_W_LEFT   = 'MIDPOINT_EXTENSION_W_LEFT'    
KEY_MIDPOINT_EXTENSION_W_RIGHT  = 'MIDPOINT_EXTENSION_W_RIGHT' 
KEY_MIDPOINT_EXTENSION_H_BACK   = 'MIDPOINT_EXTENSION_H_BACK'  
KEY_MIDPOINT_EXTENSION_H_FRONT  = 'MIDPOINT_EXTENSION_H_FRONT'  
KEY_MIDPOINT_EXTENSION_D_TOP    = 'MIDPOINT_EXTENSION_D_TOP'   
KEY_MIDPOINT_EXTENSION_D_BOTTOM = 'MIDPOINT_EXTENSION_D_BOTTOM' 

# Keys - Gridding
KEY_GRID_SIZE                   = 'GRID_SIZE'
KEY_GRID_OVERLAP                = 'GRID_OVERLAP'
KEY_GRID_SAMPLER_PERC           = 'GRID_SAMPLER_PERC'
KEY_GRID_RANDOM_SHIFT_MAX       = 'GRID_RANDOM_SHIFT_MAX'
KEY_GRID_RANDOM_SHIFT_PERC      = 'GRID_RANDOM_SHIFT_PERC'

# Keys - .nrrd file keys
KEY_NRRD_PIXEL_SPACING = 'space directions'
KEY_NRRD_ORIGIN        = 'space origin'
KEY_NRRD_SHAPE         = 'sizes'

TYPE_VOXEL_ORIGSHAPE = 'orig'
TYPE_VOXEL_RESAMPLED = 'resampled'

MASK_TYPE_ONEHOT = 'one_hot'
MASK_TYPE_COMBINED = 'combined'

PREFETCH_BUFFER = 5

################################### HaN - MICCAI 2015 ###################################

DATASET_MICCAI2015   = 'miccai2015'
DATALOADER_MICCAI2015_TRAIN      = 'train'
DATALOADER_MICCAI2015_TRAIN_ADD  = 'train_additional'
DATALOADER_MICCAI2015_TEST       = 'test_offsite'
DATALOADER_MICCAI2015_TESTONSITE = 'test_onsite'

HaN_MICCAI2015 = {
    KEY_LABEL_MAP : {
        'Background':0
        , 'BrainStem':1 , 'Chiasm':2, 'Mandible':3
        , 'OpticNerve_L':4, 'OpticNerve_R':5
        , 'Parotid_L':6,'Parotid_R':7
        ,'Submandibular_L':8, 'Submandibular_R':9
    }
    , KEY_LABEL_COLORS : {
        0: [255,255,255,10]
        , 1:[0,110,254,255], 2: [225,128,128,255], 3:[254,0,128,255]
        , 4:[191,50,191,255], 5:[254,128,254,255]
        , 6: [182, 74, 74,255], 7:[128,128,0,255]
        , 8:[50,105,161,255], 9:[46,194,194,255]
    }
    , KEY_LABEL_WEIGHTS      : [1/4231347, 1/16453, 1/372, 1/32244, 1/444, 1/397, 1/16873, 1/17510, 1/4419, 1/4410] # avg voxel count
    , KEY_IGNORE_LABELS      : [0]
    , KEY_LABELID_BACKGROUND : 0 
    , KEY_LABELID_MIDPOINT   : 1
    , KEY_HU_MIN             : -125 # window_levl=50, window_width=350, 50 - (350/2) for soft tissue
    , KEY_HU_MAX             : 225  # window_levl=50, window_width=350, 50 + (350/2) for soft tissue
    , KEY_VOXELRESO          : (0.8, 0.8, 2.5) # [(0.8, 0.8, 2.5), (1.0, 1.0, 2.0) , (1,1,1), (1,1,2)]
    , KEY_GRID_3D : {
        TYPE_VOXEL_RESAMPLED:{
            '(0.8, 0.8, 2.5)':{
                KEY_GRID_SIZE                : [140,140,40]
                , KEY_GRID_OVERLAP           : [20,20,0]
                , KEY_GRID_SAMPLER_PERC      : 0.90
                , KEY_GRID_RANDOM_SHIFT_MAX  : 40
                , KEY_GRID_RANDOM_SHIFT_PERC : 0.5
            }
        }  
    }
    , KEY_PREPROCESS:{
        TYPE_VOXEL_RESAMPLED:{
            KEY_CROP: {
                '(0.8, 0.8, 2.5)':{
                    KEY_MIDPOINT_EXTENSION_W_LEFT    : 120 
                    ,KEY_MIDPOINT_EXTENSION_W_RIGHT  : 120 # 240
                    ,KEY_MIDPOINT_EXTENSION_H_BACK   : 66
                    ,KEY_MIDPOINT_EXTENSION_H_FRONT  : 174 # 240
                    ,KEY_MIDPOINT_EXTENSION_D_TOP    : 20
                    ,KEY_MIDPOINT_EXTENSION_D_BOTTOM : 60 # 80 [96(20-76) ]
                } 
            }
        }
    }  
}


PATIENT_MICCAI2015_TESTOFFSITE = 'HaN_MICCAI2015-test_offsite-{}_resample_True'
FILENAME_SAVE_CT_MICCAI2015    = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_img.nrrd'
FILENAME_SAVE_GT_MICCAI2015    = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_mask.nrrd'
FILENAME_SAVE_PRED_MICCAI2015  = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_maskpred.nrrd' 
FILENAME_SAVE_MIF_MICCAI2015   = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_maskpredmif.nrrd'
FILENAME_SAVE_ENT_MICCAI2015   = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_maskpredent.nrrd'
FILENAME_SAVE_STD_MICCAI2015   = 'nrrd_HaN_MICCAI2015-test_offsite-{}_resample_True_maskpredstd.nrrd'

PATIENTIDS_MICCAI2015_TEST   = ['0522c0555', '0522c0576', '0522c0598', '0522c0659', '0522c0661', '0522c0667', '0522c0669', '0522c0708', '0522c0727', '0522c0746']

################################### HaN - DeepMindTCIA ###################################

DATALOADER_DEEPMINDTCIA_TEST = 'test'
DATALOADER_DEEPMINDTCIA_VAL  = 'validation'
DATALOADER_DEEPMINDTCIA_ONC  = 'oncologist'
DATALOADER_DEEPMINDTCIA_RAD  = 'radiographer'
DATASET_DEEPMINDTCIA = 'deepmindtcia'


KEY_LABELMAP_MICCAI_DEEPMINDTCIA = KEY_LABEL_MAP + 'MICCAI_TCIADEEPMIND'
HaN_DeepMindTCIA = {
    KEY_LABELMAP_MICCAI_DEEPMINDTCIA: {
        'Background': 'Background'
        , 'Brainstem': 'BrainStem'
        , 'Mandible':'Mandible'
        , 'Optic-Nerve-Lt': 'OpticNerve_L', 'Optic-Nerve-Rt': 'OpticNerve_R'
        , 'Optic_Nerve_Lt': 'OpticNerve_L', 'Optic_Nerve_Rt': 'OpticNerve_R'
        , 'Parotid-Lt':'Parotid_L', 'Parotid-Rt':'Parotid_R'
        , 'Parotid_Lt':'Parotid_L', 'Parotid_Rt':'Parotid_R'
        , 'Submandibular-Lt': 'Submandibular_L', 'Submandibular-Rt':'Submandibular_R'
        , 'Submandibular_Lt': 'Submandibular_L', 'Submandibular_Rt':'Submandibular_R'
    }
    , KEY_LABEL_MAP : {
        'Background':0
        , 'BrainStem':1 , 'Chiasm':2, 'Mandible':3
        , 'OpticNerve_L':4, 'OpticNerve_R':5
        , 'Parotid_L':6,'Parotid_R':7
        ,'Submandibular_L':8, 'Submandibular_R':9
    }
    , KEY_LABEL_COLORS : {
        0: [255,255,255,10]
        , 1:[0,110,254,255], 2: [225,128,128,255], 3:[254,0,128,255]
        , 4:[191,50,191,255], 5:[254,128,254,255]
        , 6: [182, 74, 74,255], 7:[128,128,0,255]
        , 8:[50,105,161,255], 9:[46,194,194,255]
    }
    , KEY_LABEL_WEIGHTS      : []
    , KEY_IGNORE_LABELS      : [0]
    , KEY_LABELID_BACKGROUND : 0 
    , KEY_LABELID_MIDPOINT   : 1
    , KEY_HU_MIN             : -125 # window_levl=50, window_width=350, 50 - (350/2) for soft tissue
    , KEY_HU_MAX             : 225  # window_levl=50, window_width=350, 50 + (350/2) for soft tissue
    , KEY_VOXELRESO          : (0.8, 0.8, 2.5) # [(0.8, 0.8, 2.5), (1.0, 1.0, 2.0) , (1,1,1), (1,1,2)]
    , KEY_GRID_3D : {
        TYPE_VOXEL_RESAMPLED:{
            '(0.8, 0.8, 2.5)':{
                KEY_GRID_SIZE                : [140,140,40]
                , KEY_GRID_OVERLAP           : [20,20,0]
                , KEY_GRID_SAMPLER_PERC      : 0.90
                , KEY_GRID_RANDOM_SHIFT_MAX  : 40
                , KEY_GRID_RANDOM_SHIFT_PERC : 0.5
            }
        }  
    }
    , KEY_PREPROCESS:{
        TYPE_VOXEL_RESAMPLED:{
            KEY_CROP: {
                '(0.8, 0.8, 2.5)':{
                    KEY_MIDPOINT_EXTENSION_W_LEFT    : 120 
                    ,KEY_MIDPOINT_EXTENSION_W_RIGHT  : 120 # 240
                    ,KEY_MIDPOINT_EXTENSION_H_BACK   : 66
                    ,KEY_MIDPOINT_EXTENSION_H_FRONT  : 174 # 240
                    ,KEY_MIDPOINT_EXTENSION_D_TOP    : 20
                    ,KEY_MIDPOINT_EXTENSION_D_BOTTOM : 60 # 80 [96(20-76) ]
                } 
            }
        }
    }
}

PATIENTIDS_DEEPMINDTCIA_TEST = ['0522c0331', '0522c0416', '0522c0419', '0522c0629', '0522c0768', '0522c0770', '0522c0773', '0522c0845', 'TCGA-CV-7236', 'TCGA-CV-7243', 'TCGA-CV-7245', 'TCGA-CV-A6JO', 'TCGA-CV-A6JY', 'TCGA-CV-A6K0', 'TCGA-CV-A6K1']

PATIENT_DEEPMINDTCIA_TEST_ONC            = 'HaN_DeepMindTCIA-test-oncologist-{}_resample_True'
FILENAME_SAVE_CT_DEEPMINDTCIA_TEST_ONC   = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_img.nrrd'
FILENAME_SAVE_GT_DEEPMINDTCIA_TEST_ONC   = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_mask.nrrd'
FILENAME_SAVE_PRED_DEEPMINDTCIA_TEST_ONC = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_maskpred.nrrd'
FILENAME_SAVE_MIF_DEEPMINDTCIA_TEST_ONC  = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_maskpredmif.nrrd'
FILENAME_SAVE_ENT_DEEPMINDTCIA_TEST_ONC  = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_maskpredent.nrrd'
FILENAME_SAVE_STD_DEEPMINDTCIA_TEST_ONC  = 'nrrd_HaN_DeepMindTCIA-test-oncologist-{}_resample_True_maskpredstd.nrrd'

############################################################
#                    VISUALIZATION                         #
############################################################
FIGSIZE=(15,15)
IGNORE_LABELS = []
PREDICT_THRESHOLD_MASK = 0.6

ENT_MIN, ENT_MAX = 0.0, 0.5
MIF_MIN, MIF_MAX = 0.0, 0.1