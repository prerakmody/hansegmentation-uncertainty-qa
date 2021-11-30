# Import private libraries
import src.config as config
from src.model.trainer import Trainer,Validator

# Import public libraries
import os
import pdb
import traceback
import tensorflow as tf
from pathlib import Path



if __name__ == "__main__":

    exp_name = 'HansegmentationUncertaintyQA-Flipout-CE'

    data_dir = Path(config.MAIN_DIR).joinpath('medical_dataloader', '_data')
    resampled = True
    crop_init = True
    grid      = True
    batch_size = 2

    model = config.MODEL_FOCUSNET_FLIPOUT 

    # To train
    params = {
        'exp_name': exp_name
        , 'random_seed':42
        , 'dataloader':{
            'data_dir': data_dir
            , 'dir_type'   : [config.DATALOADER_MICCAI2015_TRAIN, config.DATALOADER_MICCAI2015_TRAIN_ADD]
            , 'resampled'  : resampled
            , 'crop_init'  : crop_init
            , 'grid'       : grid
            , 'random_grid': True
            , 'filter_grid': False
            , 'centred_prob'  : 0.3
            , 'batch_size'    : batch_size  
            , 'shuffle'       : 5  
            , 'prefetch_batch': 4  
            , 'parallel_calls': 3  
        }
        , 'model': {
            'name': model
            , 'kl_alpha_init'   : 0.01
            , 'kl_schedule'     : config.KL_DIV_FIXED # [config.KL_DIV_FIXED, config.KL_DIV_ANNEALING]
            , 'kl_scale_factor' : 154 # distribute the kl_div loss 'kl_scale_factor' times across an epoch i.e. dataset_len/batch_size=> 
            # (140,140,40)=308/2.0=154
            , 'optimizer' : config.OPTIMIZER_ADAM
            , 'init_lr'  : 0.001 
            , 'fixed_lr' : True 
            , 'epochs'     : 1500    
            , 'epochs_save': 50   
            , 'epochs_eval': 50 
            , 'epochs_viz' : 500
            , 'load_model':{
                'load':False, 'load_exp_name': None,  'load_epoch':-1, 'load_optimizer_lr':None
            }
            , 'profiler': {
                'profile': False
                , 'epochs': [2,3]
                , 'steps_per_epoch': 60
                , 'starting_step': 4
            }
            , 'model_tboard': False
        }
        , 'metrics' : {
            'logging_tboard': True
            # for full 3D volume
            , 'metrics_eval': {'Dice': config.LOSS_DICE}
            ## for smaller grid/patch
            , 'metrics_loss'  : {'CE': config.LOSS_CE} # [config.LOSS_CE, config.LOSS_DICE]
            , 'loss_weighted' : {'CE': True}
            , 'loss_mask'     : {'CE': True}
            , 'loss_combo'    : {'CE': 1.0}
        }
        , 'others': {
            'epochs_timer': 20
            , 'epochs_memory':5
        }
    }

    # Call the trainer
    trainer = Trainer(params)
    trainer.train()

    # To evaluate on MICCAI2015
    params = {
        'exp_name': exp_name
        , 'pid'           : os.getpid()
        , 'dataloader': {
            'data_dir'      : data_dir
            , 'resampled'     : resampled
            , 'grid'          : grid
            , 'crop_init'     : crop_init
            , 'batch_size'    : batch_size
            , 'prefetch_batch': 1
            , 'dir_type' : [config.DATALOADER_MICCAI2015_TEST] # [config.DATALOADER_MICCAI2015_TESTONSITE]
            , 'eval_type' : config.MODE_TEST
        }
        , 'model': {
            'name': model
            , 'load_epoch'    : 1000
            , 'MC_RUNS'       : 30
            , 'training_bool' : True # [True=dropout-at-test-time, False=no-dropout-at-test-time]
        }
        , 'save': True
    }