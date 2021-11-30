# Import internal libraries
import src.config as config
import src.model.models as models
import src.model.utils as modutils
import src.model.losses as losses
import src.dataloader.utils as datautils

# Import external libraries
import os
import gc
import pdb
import copy
import time
import tqdm
import datetime
import traceback
import numpy as np
import tensorflow as tf
from pathlib import Path
import tensorflow_probability as tfp

import matplotlib.pyplot as plt; 
import seaborn as sns;

_EPSILON         = tf.keras.backend.epsilon()
DEBUG            = True

############################################################
#                      MODEL RELATED                       #
############################################################

def set_lr(epoch, optimizer, init_lr):
    
    if epoch > 1 and epoch % 20 == 0:
        optimizer.lr.assign(optimizer.lr * 0.98)

############################################################
#                      METRICS RELATED                     #
############################################################
class ModelMetrics():
    
    def __init__(self, metric_type, params):

        self.params = params

        self.label_map = params['internal']['label_map']
        self.label_ids = params['internal']['label_ids']
        self.logging_tboard = params['metrics']['logging_tboard']
        self.metric_type = metric_type

        self.losses_obj = self.get_losses_obj(params)

        self.init_metrics(params)
        if self.logging_tboard:
            self.init_tboard_writers(params)

        self.reset_metrics(params)
        self.init_epoch0(params)
        self.reset_metrics(params)
    
    def get_losses_obj(self, params):
        losses_obj = {} 
        for loss_key in params['metrics']['metrics_loss']:
            if config.LOSS_DICE == params['metrics']['metrics_loss'][loss_key]:
                losses_obj[loss_key] = losses.loss_dice_3d_tf_func
            if config.LOSS_CE  == params['metrics']['metrics_loss'][loss_key]:
                losses_obj[loss_key] = losses.loss_ce_3d_tf_func
            if config.LOSS_CE_BASIC == params['metrics']['metrics_loss'][loss_key]:
                losses_obj[loss_key] = losses.loss_cebasic_3d_tf_func

        return losses_obj
    
    def init_metrics(self, params):
        """
        These are metrics derived from tensorflows library 
        """
        # Metrics for losses (during training for smaller grids)
        self.metrics_loss_obj = {}
        for metric_key in params['metrics']['metrics_loss']:
            self.metrics_loss_obj[metric_key] = {}
            self.metrics_loss_obj[metric_key]['total'] = tf.keras.metrics.Mean(name='Avg{}-{}'.format(metric_key, self.metric_type))
            if params['metrics']['metrics_loss'][metric_key] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_CE_BASIC]:
                for label_id in self.label_ids:
                    self.metrics_loss_obj[metric_key][label_id] = tf.keras.metrics.Mean(name='Avg{}-Label-{}-{}'.format(metric_key, label_id, self.metric_type))
        
        # Metrics for eval (for full 3D volume)
        self.metrics_eval_obj = {}
        for metric_key in params['metrics']['metrics_eval']:
            self.metrics_eval_obj[metric_key] = {}
            self.metrics_eval_obj[metric_key]['total'] = tf.keras.metrics.Mean(name='Avg{}-{}'.format(metric_key, self.metric_type))
            if params['metrics']['metrics_eval'][metric_key] in [config.LOSS_DICE]:
                for label_id in self.label_ids:
                    self.metrics_eval_obj[metric_key][label_id] = tf.keras.metrics.Mean(name='Avg{}-Label-{}-{}'.format(metric_key, label_id, self.metric_type))

        # Time Metrics
        self.metric_time_dataloader     = tf.keras.metrics.Mean(name='AvgTime-Dataloader-{}'.format(self.metric_type))
        self.metric_time_model_predict  = tf.keras.metrics.Mean(name='AvgTime-ModelPredict-{}'.format(self.metric_type))
        self.metric_time_model_loss     = tf.keras.metrics.Mean(name='AvgTime-ModelLoss-{}'.format(self.metric_type))
        self.metric_time_model_backprop = tf.keras.metrics.Mean(name='AvgTime-ModelBackProp-{}'.format(self.metric_type))        
    
    def reset_metrics(self, params):

        # Metrics for losses (during training for smaller grids)
        for metric_key in params['metrics']['metrics_loss']:
            self.metrics_loss_obj[metric_key]['total'].reset_states()
            if params['metrics']['metrics_loss'][metric_key] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_CE_BASIC]:
                for label_id in self.label_ids:
                    self.metrics_loss_obj[metric_key][label_id].reset_states()

        # Metrics for eval (for full 3D volume)
        for metric_key in params['metrics']['metrics_eval']:
            self.metrics_eval_obj[metric_key]['total'].reset_states()
            if params['metrics']['metrics_eval'][metric_key] in [config.LOSS_DICE]:
                for label_id in self.label_ids:
                    self.metrics_eval_obj[metric_key][label_id].reset_states()

        # Time Metrics
        self.metric_time_dataloader.reset_states()
        self.metric_time_model_predict.reset_states()
        self.metric_time_model_loss.reset_states()
        self.metric_time_model_backprop.reset_states()
    
    def init_tboard_writers(self, params):
        """
        These are tensorboard writer
        """
        # Writers for loss (during training for smaller grids)
        self.writers_loss_obj = {}
        for metric_key in params['metrics']['metrics_loss']:
            self.writers_loss_obj[metric_key] = {}
            self.writers_loss_obj[metric_key]['total'] = modutils.get_tensorboard_writer(params['exp_name'], suffix=self.metric_type + '-Loss')
            if params['metrics']['metrics_loss'][metric_key] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_CE_BASIC]:
                for label_id in self.label_ids:
                    self.writers_loss_obj[metric_key][label_id] = modutils.get_tensorboard_writer(params['exp_name'], suffix=self.metric_type + '-Loss-' + str(label_id))
        
        # Writers for eval (for full 3D volume)
        self.writers_eval_obj = {}
        for metric_key in params['metrics']['metrics_eval']:
            self.writers_eval_obj[metric_key] = {}
            self.writers_eval_obj[metric_key]['total'] = modutils.get_tensorboard_writer(params['exp_name'], suffix=self.metric_type + '-Eval')
            if params['metrics']['metrics_eval'][metric_key] in [config.LOSS_DICE]:
                for label_id in self.label_ids:
                    self.writers_eval_obj[metric_key][label_id] = modutils.get_tensorboard_writer(params['exp_name'], suffix=self.metric_type + '-Eval-' + str(label_id))

        # Time and other writers
        self.writer_lr                  = modutils.get_tensorboard_writer(params['exp_name'], suffix='LR')
        self.writer_time_dataloader     = modutils.get_tensorboard_writer(params['exp_name'], suffix='Time-Dataloader')
        self.writer_time_model_predict  = modutils.get_tensorboard_writer(params['exp_name'], suffix='Time-Model-Predict')
        self.writer_time_model_loss     = modutils.get_tensorboard_writer(params['exp_name'], suffix='Time-Model-Loss')
        self.writer_time_model_backprop = modutils.get_tensorboard_writer(params['exp_name'], suffix='Time-Model-Backprop')
    
    def init_epoch0(self, params):
        
        for metric_str in self.metrics_loss_obj:
            self.update_metric_loss(metric_str, 1e-6)
            if params['metrics']['metrics_loss'][metric_str] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_CE_BASIC]:
                self.update_metric_loss_labels(metric_str, {label_id: 1e-6 for label_id in self.label_ids})

        for metric_str in self.metrics_eval_obj:
            self.update_metric_eval_labels(metric_str, {label_id: 1e-6 for label_id in self.label_ids})
            
        self.update_metrics_time(time_dataloader=1e-6, time_predict=1e-6, time_loss=1e-6, time_backprop=1e-6)

        self.write_epoch_summary(epoch=0, label_map=self.label_map, params=None, eval_condition=True)

    def update_metrics_time(self, time_dataloader, time_predict, time_loss, time_backprop):
        if time_dataloader is not None:
            self.metric_time_dataloader.update_state(time_dataloader)
        if time_predict is not None:
            self.metric_time_model_predict.update_state(time_predict)
        if time_loss is not None:
            self.metric_time_model_loss.update_state(time_loss)
        if time_backprop is not None:
            self.metric_time_model_backprop.update_state(time_backprop)  

    def update_metric_loss(self, metric_str, metric_val):
        # Metrics for losses (during training for smaller grids)
        self.metrics_loss_obj[metric_str]['total'].update_state(metric_val)
    
    @tf.function
    def update_metric_loss_labels(self, metric_str, metric_vals_labels):
        # Metrics for losses (during training for smaller grids)

        for label_id in self.label_ids:
            if metric_vals_labels[label_id] > 0.0:
                self.metrics_loss_obj[metric_str][label_id].update_state(metric_vals_labels[label_id])
        
    def update_metric_eval(self, metric_str, metric_val):
        # Metrics for eval (for full 3D volume)
        self.metrics_eval_obj[metric_str]['total'].update_state(metric_val)
        
    def update_metric_eval_labels(self, metric_str, metric_vals_labels, do_average=False):
        # Metrics for eval (for full 3D volume)

        metric_avg = []
        for label_id in self.label_ids:
            if label_id in metric_vals_labels:
                if metric_vals_labels[label_id] > 0:
                    self.metrics_eval_obj[metric_str][label_id].update_state(metric_vals_labels[label_id])
                    if do_average:
                        if label_id > 0:
                            metric_avg.append(metric_vals_labels[label_id])
        
        if do_average:
            if len(metric_avg):
                self.metrics_eval_obj[metric_str]['total'].update_state(np.mean(metric_avg))

    def write_epoch_summary(self, epoch, label_map, params=None, eval_condition=False):
        
        if self.logging_tboard:
            # Metrics for losses (during training for smaller grids)
            for metric_str in self.metrics_loss_obj:
                modutils.make_summary('Loss/{}'.format(metric_str), epoch, writer1=self.writers_loss_obj[metric_str]['total'], value1=self.metrics_loss_obj[metric_str]['total'].result())
                if self.params['metrics']['metrics_loss'][metric_str] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_CE_BASIC]:
                    if len(self.metrics_loss_obj[metric_str]) > 1: # i.e. has label ids
                        for label_id in self.label_ids:
                            label_name, _ = modutils.get_info_from_label_id(label_id, label_map)
                            modutils.make_summary('Loss/{}/{}'.format(metric_str, label_name), epoch, writer1=self.writers_loss_obj[metric_str][label_id], value1=self.metrics_loss_obj[metric_str][label_id].result())
                
            # Metrics for eval (for full 3D volume)
            if eval_condition:
                for metric_str in self.metrics_eval_obj:
                    modutils.make_summary('Eval3D/{}'.format(metric_str), epoch, writer1=self.writers_eval_obj[metric_str]['total'], value1=self.metrics_eval_obj[metric_str]['total'].result())
                    if len(self.metrics_eval_obj[metric_str]) > 1: # i.e. has label ids
                        for label_id in self.label_ids:
                            label_name, _ = modutils.get_info_from_label_id(label_id, label_map)
                            modutils.make_summary('Eval3D/{}/{}'.format(metric_str, label_name), epoch, writer1=self.writers_eval_obj[metric_str][label_id], value1=self.metrics_eval_obj[metric_str][label_id].result())

            # Time Metrics
            modutils.make_summary('Info/Time/Dataloader'   , epoch, writer1=self.writer_time_dataloader    , value1=self.metric_time_dataloader.result())
            modutils.make_summary('Info/Time/ModelPredict' , epoch, writer1=self.writer_time_model_predict , value1=self.metric_time_model_predict.result())
            modutils.make_summary('Info/Time/ModelLoss'    , epoch, writer1=self.writer_time_model_loss    , value1=self.metric_time_model_loss.result())
            modutils.make_summary('Info/Time/ModelBackProp', epoch, writer1=self.writer_time_model_backprop, value1=self.metric_time_model_backprop.result())
            
            # Learning Rate
            if params is not None:
                if 'optimizer' in params:
                    modutils.make_summary('Info/LR', epoch, writer1=self.writer_lr, value1=params['optimizer'].lr)

    def update_pbar(self, pbar):
        desc_str = ''

        # Metrics for losses (during training for smaller grids)
        for metric_str in self.metrics_loss_obj:
            if len(desc_str): desc_str += ',' 

            if metric_str in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_CE_BASIC]:
                metric_avg = []
                for label_id in self.label_ids:
                    if label_id > 0:
                        label_result = self.metrics_loss_obj[metric_str][label_id].result().numpy()
                        if label_result > 0:
                            metric_avg.append(label_result)
                
                mean_val = 0
                if len(metric_avg):     
                    mean_val = np.mean(metric_avg)
                loss_text = '{}Loss:{:2f}'.format(metric_str, mean_val)
                desc_str += loss_text
        
        # GPU Memory
        if 1:
            try:
                if len(self.metrics_loss_obj) > 1:
                    desc_str = desc_str[:-1] # to remove the extra ','     
                desc_str += ',' + str(modutils.get_tf_gpu_memory())
            except:
                pass
        
        pbar.set_description(desc=desc_str, refresh=True)

def eval_3D_finalize(exp_name, patient_img, patient_gt, patient_pred_processed, patient_pred, patient_pred_std, patient_pred_ent, patient_pred_mif, patient_pred_unc
                        , patient_id_curr
                        , model_folder_epoch_patches 
                        , loss_labels_val, hausdorff_labels_val, hausdorff95_labels_val, msd_labels_vals
                        , spacing, label_map
                        , save=False):

    try:

        # Step 1 - Save 3D grid to visualize in 3D Slicer (drag-and-drop mechanism)
        if save:
            
            datautils.write_nrrd(str(Path(model_folder_epoch_patches).joinpath('nrrd_' + patient_id_curr)) + '_img.nrrd' , patient_img[:,:,:,0], spacing)
            datautils.write_nrrd(str(Path(model_folder_epoch_patches).joinpath('nrrd_' + patient_id_curr)) + '_mask.nrrd', np.argmax(patient_gt, axis=3),spacing)
            maskpred_labels = np.argmax(patient_pred_processed, axis=3) # not "np.argmax(patient_pred, axis=3)" since it does not contain any postprocessing
            datautils.write_nrrd(str(Path(model_folder_epoch_patches).joinpath('nrrd_' + patient_id_curr)) + '_maskpred.nrrd', maskpred_labels, spacing)

            maskpred_labels_probmean = np.take_along_axis(patient_pred, np.expand_dims(maskpred_labels,axis=-1), axis=-1)[:,:,:,0]
            datautils.write_nrrd(str(Path(model_folder_epoch_patches).joinpath('nrrd_' + patient_id_curr)) + '_maskpredmean.nrrd', maskpred_labels_probmean, spacing)

            if np.sum(patient_pred_std):
                maskpred_labels_std = np.take_along_axis(patient_pred_std, np.expand_dims(maskpred_labels,axis=-1), axis=-1)[:,:,:,0]
                datautils.write_nrrd(str(Path(model_folder_epoch_patches).joinpath('nrrd_' + patient_id_curr)) + '_maskpredstd.nrrd', maskpred_labels_std, spacing)

                maskpred_std_max = np.max(patient_pred_std, axis=-1)
                datautils.write_nrrd(str(Path(model_folder_epoch_patches).joinpath('nrrd_' + patient_id_curr)) + '_maskpredstdmax.nrrd', maskpred_std_max, spacing)

            if np.sum(patient_pred_ent):
                maskpred_ent     = patient_pred_ent # [H,W,D]
                datautils.write_nrrd(str(Path(model_folder_epoch_patches).joinpath('nrrd_' + patient_id_curr)) + '_maskpredent.nrrd', maskpred_ent, spacing)

            if np.sum(patient_pred_mif):
                maskpred_mif = patient_pred_mif
                datautils.write_nrrd(str(Path(model_folder_epoch_patches).joinpath('nrrd_' + patient_id_curr)) + '_maskpredmif.nrrd', maskpred_mif, spacing)

            if np.sum(patient_pred_unc):
                if len(patient_pred_unc.shape) == 4:
                    maskpred_labels_unc = np.take_along_axis(patient_pred_unc, np.expand_dims(maskpred_labels,axis=-1), axis=-1)[:,:,:,0] # [H,W,D,C] --> [H,W,D]
                else:
                    maskpred_labels_unc = patient_pred_unc
                datautils.write_nrrd(str(Path(model_folder_epoch_patches).joinpath('nrrd_' + patient_id_curr)) + '_maskpredunc.nrrd', maskpred_labels_unc, spacing)

            try:
                # Step 3.1.3.2 - PLot results for that patient
                f, axarr = plt.subplots(3,1, figsize=(15,10))
                boxplot_dice, boxplot_hausdorff, boxplot_hausdorff95 = {}, {}, {}
                boxplot_dice_mean_list = []
                for label_id in range(len(loss_labels_val)):
                    label_name, _ = modutils.get_info_from_label_id(label_id, label_map)
                    boxplot_dice[label_name] = [loss_labels_val[label_id]]
                    boxplot_hausdorff[label_name] = [hausdorff_labels_val[label_id]]
                    boxplot_hausdorff95[label_name] = [hausdorff95_labels_val[label_id]]
                    if label_id > 0 and loss_labels_val[label_id] > 0:
                        boxplot_dice_mean_list.append(loss_labels_val[label_id])
                
                axarr[0].boxplot(boxplot_dice.values())
                axarr[0].set_xticks(range(1, len(boxplot_dice)+1))
                axarr[0].set_xticklabels(boxplot_dice.keys())
                axarr[0].set_ylim([0.0,1.1])
                axarr[0].grid()
                axarr[0].set_title('DICE - Avg: {} \n w/o chiasm: {}'.format(
                    '%.4f' % (np.mean(boxplot_dice_mean_list))
                    , '%.4f' % (np.mean(boxplot_dice_mean_list[0:1] + boxplot_dice_mean_list[2:])) # avoid label_id=2
                    )
                )

                axarr[1].boxplot(boxplot_hausdorff.values())
                axarr[1].set_xticks(range(1,len(boxplot_hausdorff)+1))
                axarr[1].set_xticklabels(boxplot_hausdorff.keys())
                axarr[1].grid()
                axarr[1].set_title('Hausdorff')
                
                axarr[2].boxplot(boxplot_hausdorff95.values())
                axarr[2].set_xticks(range(1,len(boxplot_hausdorff95)+1))
                axarr[2].set_xticklabels(boxplot_hausdorff95.keys())
                axarr[2].set_title('95% Hausdorff')
                axarr[2].grid()

                plt.savefig(str(Path(model_folder_epoch_patches).joinpath('results_' + patient_id_curr + '.png')), bbox_inches='tight') # , bbox_inches='tight'
                plt.close()
            
            except:
                traceback.print_exc()
    
    except:
        traceback.print_exc()
        pdb.set_trace()
        
def get_ece(y_true, y_predict, patient_id, res_global, verbose=False):
    """
    Params
    ------
    y_true   : [H,W,D,C], np.array, binary 
    y_predict: [H,W,D,C], np.array, with softmax probability values
    - Ref: https://github.com/sirius8050/Expected-Calibration-Error/blob/master/ECE.py
         : On Calibration of Modern Neural Networks
    - Ref(future): https://github.com/yding5/AdaptiveBinning
                 : Revisiting the evaluation of uncertainty estimation and its application to explore model complexity-uncertainty trade-off
    """
    res = {}
    nan_value = -0.1

    if verbose: print (' - [get_ece()] patient_id: ', patient_id)

    try:
        
        # Step 0 - Init
        label_count = y_true.shape[-1]

        # Step 1 - Calculate o_predict
        o_true    = np.argmax(y_true, axis=-1)
        o_predict = np.argmax(y_predict, axis=-1)

        # Step 2 - Loop over different classes
        for label_id in range(label_count):
            
            if label_id > -1:

                if verbose: print (' --- [get_ece()] label_id: ', label_id)

                if label_id not in res_global: res_global[label_id] = {'o_predict_label':[], 'y_predict_label':[], 'o_true_label':[]} 
                
                # Step 2.1 - Get o_predict_label(label_ids), o_true_label(label_ids), y_predict_label(probs) [and append to global list]        
                ## NB: You are considering TP + FP here     
                o_true_label    = o_true[o_predict == label_id]
                o_predict_label = o_predict[o_predict == label_id]
                y_predict_label = y_predict[:,:,:,label_id][o_predict == label_id]
                res_global[label_id]['o_true_label'].extend(o_true_label.flatten().tolist())
                res_global[label_id]['o_predict_label'].extend(o_predict_label.flatten().tolist())
                res_global[label_id]['y_predict_label'].extend(y_predict_label.flatten().tolist())

                if len(o_true_label) and len(y_predict_label):

                    # Step 2.2 - Bin the probs and calculate their mean
                    y_predict_label_bin_ids = np.digitize(y_predict_label, np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]), right=False) - 1
                    y_predict_binned_vals   = [y_predict_label[y_predict_label_bin_ids == bin_id] for bin_id in range(label_count)] 
                    y_predict_bins_mean     = [np.mean(vals) if len(vals) else nan_value for vals in y_predict_binned_vals]

                    # Step 2.3 - Calculate the accuracy of each bin
                    o_predict_label_bins    = [o_predict_label[y_predict_label_bin_ids == bin_id] for bin_id in range(label_count)]
                    o_true_label_bins       = [o_true_label[y_predict_label_bin_ids == bin_id] for bin_id in range(label_count)]
                    y_predict_bins_accuracy = [np.sum(o_predict_label_bins[bin_id] == o_true_label_bins[bin_id])/len(o_predict_label_bins[bin_id]) if len(o_predict_label_bins[bin_id]) else nan_value for bin_id in range(label_count)]
                    y_predict_bins_len      = [len(o_predict_label_bins[bin_id]) for bin_id in range(label_count)]

                    # Step 2.4 - Wrapup
                    N             = np.prod(y_predict_label.shape)
                    ce            = np.array((np.array(y_predict_bins_len)/N)*(np.array(y_predict_bins_accuracy)-np.array(y_predict_bins_mean)))
                    ce[ce == 0]   = nan_value # i.e. y_predict_bins_accuracy[bin_id] == y_predict_bins_mean[bind_id] = nan_value
                    res[label_id] = ce
                    
                else:
                    res[label_id] = -1
                
                if 0:
                    if label_id == 1:
                        diff = np.array(y_predict_bins_accuracy)-np.array(y_predict_bins_mean)
                        print (' - [get_ece()][BStem] diff: ', ['%.4f' % each for each in diff])

                        # NB: This considers the whole volume
                        o_true_label    = y_true[:,:,:,label_id] # [1=this label, 0=other label]
                        o_predict_label = np.array(o_predict, copy=True)
                        o_predict_label[o_predict_label != label_id] = 0 
                        o_predict_label[o_predict_label == label_id] = 1 # [1 - predicted this label, 0 = predicted other label]
                        y_predict_label = y_predict[:,:,:,label_id]

                        # Step x.2 - Bin the probs and calculate their mean
                        y_predict_label_bin_ids = np.digitize(y_predict_label, np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]), right=False) - 1
                        y_predict_binned_vals   = [y_predict_label[y_predict_label_bin_ids == bin_id] for bin_id in range(label_count)] 
                        y_predict_bins_mean     = [np.mean(vals) if len(vals) else nan_value for vals in y_predict_binned_vals]

                        # Step x.3 - Calculate the accuracy of each bin
                        o_predict_label_bins    = [o_predict_label[y_predict_label_bin_ids == bin_id] for bin_id in range(label_count)]
                        o_true_label_bins       = [o_true_label[y_predict_label_bin_ids == bin_id] for bin_id in range(label_count)]
                        y_predict_bins_accuracy = [np.sum(o_predict_label_bins[bin_id] == o_true_label_bins[bin_id])/len(o_predict_label_bins[bin_id]) if len(o_predict_label_bins[bin_id]) else nan_value for bin_id in range(label_count)]
                        y_predict_bins_len      = [len(o_predict_label_bins[bin_id]) for bin_id in range(label_count)]
                        
                        N_new    = np.prod(y_predict_label.shape)
                        diff_new = np.array(y_predict_bins_accuracy)-np.array(y_predict_bins_mean)
                        ce_new   = np.array((np.array(y_predict_bins_len)/N_new)*(np.array(y_predict_bins_accuracy)-np.array(y_predict_bins_mean)))
                        print (' - [get_ece()][BStem] diff_new: ', ['%.4f' % each for each in diff_new])

                        pdb.set_trace()

                if verbose: 
                    print (' --- [get_ece()] y_predict_bins_accuracy: ', ['%.4f' % (each) for each in np.array(y_predict_bins_accuracy)])
                    print (' --- [get_ece()] CE : ', ['%.4f' % (each) for each in np.array(res[label_id])])
                    print (' --- [get_ece()] ECE: ', np.sum(np.abs(res[label_id][res[label_id] != nan_value])))

                    # Prob bars
                    # plt.hist(y_predict[:,:,:,label_id].flatten(), bins=10)
                    # plt.title('Softmax Probs (label={})\nPatient:{}'.format(label_id, patient_id))
                    # plt.show()

                    # GT Prob bars
                    # plt.bar(np.arange(len(y_predict_bins_len))/10.0 + 0.1, y_predict_bins_len, width=0.05)
                    # plt.title('Softmax Probs (GT) (label={})\nPatient:{}'.format(label_id, patient_id))
                    # plt.xlabel('Probabilities')
                    # plt.show()

                    # GT Probs (sorted) in plt.plot (with equally-spaced bins)
                    # from collections import Counter
                    # tmp = np.sort(y_predict_label)
                    # plt.plot(range(len(tmp)), tmp, color='orange')
                    # tmp_bins = np.digitize(tmp, np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01])) - 1
                    # tmp_bins_len = Counter(tmp_bins)
                    # boundary_start = 0
                    # plt.plot([0,0],[0.0,1.0], color='black', alpha=0.5, linestyle='dashed', label='Bins(equally-spaced)')
                    # for boundary in np.arange(0,len(tmp_bins_len)): plt.plot([boundary_start+tmp_bins_len[boundary], boundary_start+tmp_bins_len[boundary]], [0.0,1.0], color='black', alpha=0.5, linestyle='dashed'); boundary_start+=tmp_bins_len[boundary]
                    # plt.title('Sorted Softmax Probs (GT) (label={})\nPatient:{}'.format(label_id, patient_id))
                    # plt.legend()
                    # plt.show()

                    # GT Probs (sorted) in plt.plot (with equally-sized bins)
                    if label_id == 1:
                        Path('tmp').mkdir(parents=True, exist_ok=True)
                        tmp = np.sort(y_predict_label)
                        tmp_len = len(tmp)
                        plt.plot(range(len(tmp)), tmp, color='orange')
                        for boundary in np.arange(0,tmp_len, int(tmp_len//10)): plt.plot([boundary, boundary], [0.0,1.0], color='black', alpha=0.5, linestyle='dashed')
                        plt.plot([0,0],[0,0], color='black', alpha=0.5, linestyle='dashed', label='Bins(equally-sized)')
                        plt.title('Sorted Softmax Probs (GT) (label={})\nPatient:{}'.format(label_id, patient_id))
                        plt.legend()
                        # plt.show()
                        plt.savefig('./tmp/ECE_SortedProbs_label_{}_{}.png'.format(label_id, patient_id, ), bbox_inches='tight');plt.close()

                        # ECE plot
                        plt.plot(np.arange(11), np.arange(11)/10.0, linestyle='dashed', color='black', alpha=0.8)
                        plt.scatter(np.arange(len(y_predict_bins_mean)) + 0.5     , y_predict_bins_mean, alpha=0.5, color='g', marker='s', label='Mean Pred')
                        plt.scatter(np.arange(len(y_predict_bins_accuracy)) + 0.5 , y_predict_bins_accuracy, alpha=0.5, color='b', marker='x', label='Accuracy')
                        diff = np.array(y_predict_bins_accuracy)-np.array(y_predict_bins_mean)
                        for bin_id in range(len(y_predict_bins_accuracy)): plt.plot([bin_id + 0.5, bin_id + 0.5],[y_predict_bins_accuracy[bin_id], y_predict_bins_mean[bin_id]], color='pink')
                        plt.plot([bin_id + 0.5, bin_id + 0.5],[y_predict_bins_accuracy[bin_id], y_predict_bins_mean[bin_id]], color='pink', label='CE')
                        plt.xticks(ticks=np.arange(11), labels=np.arange(11)/10.0)
                        plt.title('CE (label={})\nPatient:{}'.format(label_id, patient_id))
                        plt.xlabel('Probability')
                        plt.ylabel('Accuracy')
                        plt.legend()
                        # plt.show()
                        plt.savefig('./tmp/ECE_label_{}_{}.png'.format(label_id, patient_id, ), bbox_inches='tight');plt.close()
                    
                    # pdb.set_trace()

    except:
        traceback.print_exc()
        pdb.set_trace()
    
    return res_global, res

def eval_3D_summarize(res, ece_global_obj, model, eval_type, deepsup_eval, label_map, model_folder_epoch_patches, times_mcruns, ttotal, save=False, verbose=False):

    try:
        
        pid = os.getpid()

        ###############################################################################
        #                                 Summarize                                   #
        ###############################################################################    

        # Step 1 - Summarize DICE + Surface Distances
        if 1:
            loss_labels_avg, loss_labels_std               = [], []
            hausdorff_labels_avg, hausdorff_labels_std     = [], []
            hausdorff95_labels_avg, hausdorff95_labels_std = [], []
            msd_labels_avg, msd_labels_std                 = [], []
            
            loss_labels_list                               = np.array([res[patient_id][config.KEY_DICE_LABELS] for patient_id in res])
            hausdorff_labels_list                          = np.array([res[patient_id][config.KEY_HD_LABELS] for patient_id in res])
            hausdorff95_labels_list                        = np.array([res[patient_id][config.KEY_HD95_LABELS] for patient_id in res])
            msd_labels_list                                = np.array([res[patient_id][config.KEY_MSD_LABELS] for patient_id in res])

            for label_id in range(loss_labels_list.shape[1]): 
                tmp_vals = loss_labels_list[:,label_id]
                loss_labels_avg.append(np.mean(tmp_vals[tmp_vals > 0]))
                loss_labels_std.append(np.std(tmp_vals[tmp_vals > 0]))

                if label_id > 0:
                    tmp_vals = hausdorff_labels_list[:,label_id]
                    hausdorff_labels_avg.append(np.mean(tmp_vals[tmp_vals > 0])) # avoids -1 for "erroneous" HD, and 0 for "not to be calculated" HD
                    hausdorff_labels_std.append(np.std(tmp_vals[tmp_vals > 0]))

                    tmp_vals = hausdorff95_labels_list[:,label_id]
                    hausdorff95_labels_avg.append(np.mean(tmp_vals[tmp_vals > 0]))
                    hausdorff95_labels_std.append(np.std(tmp_vals[tmp_vals > 0]))

                    tmp_vals = msd_labels_list[:,label_id]
                    msd_labels_avg.append(np.mean(tmp_vals[tmp_vals > 0]))
                    msd_labels_std.append(np.std(tmp_vals[tmp_vals > 0]))

                else:
                    hausdorff_labels_avg.append(0)
                    hausdorff_labels_std.append(0)
                    hausdorff95_labels_avg.append(0)
                    hausdorff95_labels_std.append(0)
                    msd_labels_avg.append(0)
                    msd_labels_std.append(0)

            loss_avg = np.mean([res[patient_id][config.KEY_DICE_AVG] for patient_id in res])
            print (' --------------------------- eval_type: ', eval_type)
            print (' - dice_labels_3D   : ', ['%.4f' % each for each in loss_labels_avg])
            print (' - dice_labels_3D   : ', ['%.4f' % each for each in loss_labels_std])
            print (' - dice_3D          : %.4f' % np.mean(loss_labels_avg))
            print (' - dice_3D (w/o bgd): %.4f' %  np.mean(loss_labels_avg[1:]))
            print (' - dice_3D (w/o bgd, w/o chiasm): %.4f' %  np.mean(loss_labels_avg[1:2] + loss_labels_avg[3:]))
            print ('')
            print (' - hausdorff_labels_3D   : ', ['%.4f' % each for each in hausdorff_labels_avg])
            print (' - hausdorff_labels_3D   :  ', ['%.4f' % each for each in hausdorff_labels_std])
            print (' - hausdorff_3D (w/o bgd): %.4f' %  np.mean(hausdorff_labels_avg[1:]))
            print (' - hausdorff_3D (w/o bgd, w/o chiasm): %.4f' %  np.mean(hausdorff_labels_avg[1:2] + hausdorff_labels_avg[3:]))
            print ('')
            print (' - hausdorff95_labels_3D   : ', ['%.4f' % each for each in hausdorff95_labels_avg])
            print (' - hausdorff95_labels_3D   : ', ['%.4f' % each for each in hausdorff95_labels_std])
            print (' - hausdorff95_3D (w/o bgd): %.4f' %  np.mean(hausdorff95_labels_avg[1:]))
            print (' - hausdorff95_3D (w/o bgd, w/o chiasm): %.4f' %  np.mean(hausdorff95_labels_avg[1:2] + hausdorff95_labels_avg[3:]))
            print ('')
            print (' - msd_labels_3D   : ', ['%.4f' % each for each in msd_labels_avg])
            print (' - msd_labels_3D   : ', ['%.4f' % each for each in msd_labels_std])
            print (' - msd_3D (w/o bgd): %.4f' %  np.mean(msd_labels_avg[1:]))
            print (' - msd_3D (w/o bgd, w/o chiasm): %.4f' %  np.mean(msd_labels_avg[1:2] + msd_labels_avg[3:]))

        # Step 2 - Summarize ECE
        if 1:
            print ('')
            gc.collect()
            nan_value = config.VAL_ECE_NAN
            ece_labels_obj = {}
            ece_labels = []
            label_count = len(ece_global_obj)
            pbar_desc_prefix = '[ECE]'
            ece_global_obj_keys = list(ece_global_obj.keys()) 
            res[config.KEY_PATIENT_GLOBAL] = {}
            with tqdm.tqdm(total=label_count, desc=pbar_desc_prefix, disable=True) as pbar_ece:
                for label_id in ece_global_obj_keys:
                    o_true_label    = np.array(ece_global_obj[label_id]['o_true_label'])
                    o_predict_label = np.array(ece_global_obj[label_id]['o_predict_label'])
                    y_predict_label = np.array(ece_global_obj[label_id]['y_predict_label'])
                    if label_id in ece_global_obj: del ece_global_obj[label_id]
                    gc.collect()

                    # Step 1.1 - Bin the probs and calculate their mean
                    y_predict_label_bin_ids = np.digitize(y_predict_label, np.array([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]), right=False) - 1
                    y_predict_binned_vals   = [y_predict_label[y_predict_label_bin_ids == bin_id] for bin_id in range(label_count)] 
                    y_predict_bins_mean     = [np.mean(vals) if len(vals) else nan_value for vals in y_predict_binned_vals]

                    # Step 1.2 - Calculate the accuracy of each bin
                    o_predict_label_bins    = [o_predict_label[y_predict_label_bin_ids == bin_id] for bin_id in range(label_count)]
                    o_true_label_bins       = [o_true_label[y_predict_label_bin_ids == bin_id] for bin_id in range(label_count)]
                    y_predict_bins_accuracy = [np.sum(o_predict_label_bins[bin_id] == o_true_label_bins[bin_id])/len(o_predict_label_bins[bin_id]) if len(o_predict_label_bins[bin_id]) else nan_value for bin_id in range(label_count)]
                    y_predict_bins_len      = [len(o_predict_label_bins[bin_id]) for bin_id in range(label_count)]

                    # Step 1.3 - Wrapup
                    N = np.prod(y_predict_label.shape)
                    ce = np.array((np.array(y_predict_bins_len)/N)*(np.array(y_predict_bins_accuracy)-np.array(y_predict_bins_mean)))
                    ce[ce == 0] = nan_value
                    ece_label = np.sum(np.abs(ce[ce != nan_value]))
                    ece_labels.append(ece_label)
                    ece_labels_obj[label_id] = {'y_predict_bins_mean':y_predict_bins_mean, 'y_predict_bins_accuracy':y_predict_bins_accuracy, 'ce':ce, 'ece':ece_label}

                    pbar_ece.update(1)
                    memory = pbar_desc_prefix + ' [' + str(modutils.get_memory(pid)) + ']'
                    pbar_ece.set_description(desc=memory, refresh=True)

                    res[config.KEY_PATIENT_GLOBAL][label_id] = {'ce':ce, 'ece':ece_label}

            print (' - ece_labels   : ', ['%.4f' % each for each in ece_labels])
            print (' - ece          : %.4f' % np.mean(ece_labels))
            print (' - ece (w/o bgd): %.4f' %  np.mean(ece_labels[1:]))
            print (' - ece (w/o bgd, w/o chiasm): %.4f' %  np.mean(ece_labels[1:2] + ece_labels[3:]))
            print ('')

            del ece_global_obj
            gc.collect()

        # Step 3 - Plot
        if 1:
            if save and not deepsup_eval:
                f, axarr = plt.subplots(3,1, figsize=(15,10))
                boxplot_dice, boxplot_hausdorff, boxplot_hausdorff95, boxplot_msd = {}, {}, {}, {}
                for label_id in range(len(loss_labels_list[0])):
                    label_name, _                   = modutils.get_info_from_label_id(label_id, label_map)
                    boxplot_dice[label_name]        = loss_labels_list[:,label_id]
                    boxplot_hausdorff[label_name]   = hausdorff_labels_list[:,label_id]
                    boxplot_hausdorff95[label_name] = hausdorff95_labels_list[:,label_id]
                    boxplot_msd[label_name]         = msd_labels_list[:,label_id]

                axarr[0].boxplot(boxplot_dice.values())
                axarr[0].set_xticks(range(1, len(boxplot_dice)+1))
                axarr[0].set_xticklabels(boxplot_dice.keys())
                axarr[0].set_ylim([0.0,1.1])
                axarr[0].set_title('DICE (Avg: {}) \n w/o chiasm:{}'.format( 
                    '%.4f' %  np.mean(loss_labels_avg[1:])
                    , '%.4f' %  np.mean(loss_labels_avg[1:2] + loss_labels_avg[3:])
                    )
                )

                axarr[1].boxplot(boxplot_hausdorff.values())
                axarr[1].set_xticks(range(1, len(boxplot_hausdorff)+1))
                axarr[1].set_xticklabels(boxplot_hausdorff.keys())
                axarr[1].set_ylim([0.0,10.0])
                axarr[1].set_title('Hausdorff')
                
                axarr[2].boxplot(boxplot_hausdorff95.values())
                axarr[2].set_xticks(range(1, len(boxplot_hausdorff95)+1))
                axarr[2].set_xticklabels(boxplot_hausdorff95.keys())
                axarr[2].set_ylim([0.0,6.0])
                axarr[2].set_title('95% Hausdorff')
                
                path_results = str(Path(model_folder_epoch_patches).joinpath('results_all.png'))
                plt.savefig(path_results, bbox_inches='tight')
                plt.close()

                f, axarr = plt.subplots(1, figsize=(15,10))
                axarr.boxplot(boxplot_dice.values())
                axarr.set_xticks(range(1, len(boxplot_dice)+1))
                axarr.set_xticklabels(boxplot_dice.keys())
                axarr.set_ylim([0.0,1.1])
                axarr.set_yticks(np.arange(0,1.1,0.05))
                axarr.set_title('DICE')
                axarr.grid()
                path_results = str(Path(model_folder_epoch_patches).joinpath('results_all_dice.png'))
                plt.savefig(path_results, bbox_inches='tight')
                plt.close()

                f, axarr = plt.subplots(1, figsize=(15,10))
                axarr.boxplot(boxplot_hausdorff95.values())
                axarr.set_xticks(range(1, len(boxplot_hausdorff95)+1))
                axarr.set_xticklabels(boxplot_hausdorff95.keys())
                axarr.set_title('95% HD')
                axarr.grid()
                path_results = str(Path(model_folder_epoch_patches).joinpath('results_all_hd95.png'))
                plt.savefig(path_results, bbox_inches='tight')
                plt.close()

                f, axarr = plt.subplots(1, figsize=(15,10))
                axarr.boxplot(boxplot_hausdorff.values())
                axarr.set_xticks(range(1, len(boxplot_hausdorff)+1))
                axarr.set_xticklabels(boxplot_hausdorff.keys())
                axarr.set_title('Hausdorff Distance')
                axarr.grid()
                path_results = str(Path(model_folder_epoch_patches).joinpath('results_all_hd.png'))
                plt.savefig(path_results, bbox_inches='tight')
                plt.close()

                f, axarr = plt.subplots(1, figsize=(15,10))
                axarr.boxplot(boxplot_msd.values())
                axarr.set_xticks(range(1, len(boxplot_msd)+1))
                axarr.set_xticklabels(boxplot_msd.keys())
                axarr.set_title('MSD')
                axarr.grid()
                path_results = str(Path(model_folder_epoch_patches).joinpath('results_all_msd.png'))
                plt.savefig(path_results, bbox_inches='tight')
                plt.close()

                # ECE
                for label_id in ece_labels_obj:
                    y_predict_bins_mean = ece_labels_obj[label_id]['y_predict_bins_mean']
                    y_predict_bins_accuracy = ece_labels_obj[label_id]['y_predict_bins_accuracy']
                    ece = ece_labels_obj[label_id]['ece']

                    plt.plot(np.arange(11), np.arange(11)/10.0, linestyle='dashed', color='black', alpha=0.8)
                    plt.scatter(np.arange(len(y_predict_bins_mean)) + 0.5     , y_predict_bins_mean, alpha=0.5, color='g', marker='s', label='Mean Pred')
                    plt.scatter(np.arange(len(y_predict_bins_accuracy)) + 0.5 , y_predict_bins_accuracy, alpha=0.5, color='b', marker='x', label='Accuracy')
                    for bin_id in range(len(y_predict_bins_accuracy)): plt.plot([bin_id + 0.5, bin_id + 0.5],[y_predict_bins_accuracy[bin_id], y_predict_bins_mean[bin_id]], color='pink')
                    plt.plot([bin_id + 0.5, bin_id + 0.5],[y_predict_bins_accuracy[bin_id], y_predict_bins_mean[bin_id]], color='pink', label='CE')
                    plt.xticks(ticks=np.arange(11), labels=np.arange(11)/10.0)
                    plt.title('CE (label={})\nECE: {}'.format(label_id, '%.5f' % (ece)))
                    plt.xlabel('Probability')
                    plt.ylabel('Accuracy')
                    plt.ylim([-0.15, 1.05])
                    plt.legend()
                    
                    # plt.show()
                    path_results = str(Path(model_folder_epoch_patches).joinpath('results_ece_label{}.png'.format(label_id)))
                    plt.savefig(str(path_results), bbox_inches='tight')
                    plt.close()

        # Step 5 - Save data as .json
        if 1:
            try:
                filename                      = str(Path(model_folder_epoch_patches).joinpath(config.FILENAME_EVAL3D_JSON))
                modutils.write_json(res, filename)

            except:
                traceback.print_exc()
                if DEBUG: pdb.set_trace()

        model.trainable=True
        print ('\n - [eval_3D] Avg of times_mcruns               : {:f} +- {:f}'.format(np.mean(times_mcruns), np.std(times_mcruns)))
        print (' - [eval_3D()] Total time passed (save={})  : {}s \n'.format(save, round(time.time() - ttotal, 2)))
        if verbose: pdb.set_trace()

        return loss_avg, {i:loss_labels_avg[i] for i in range(len(loss_labels_avg))}

    except:
        model.trainable=True
        traceback.print_exc()
        if DEBUG: pdb.set_trace()
        return -1, {}

def eval_3D_process_outputs(exp_name, res, ece_global_obj, patient_id_curr, meta1_batch, patient_img, patient_gt, patient_pred_vals, patient_pred_overlap, patient_pred_std, patient_pred_ent, patient_pred_mif, patient_pred_unc
            , deepsup_eval, model_folder_epoch_imgs, model_folder_epoch_patches, label_map, label_colors, t99, save=False, verbose=False):

    try:

        # Step 3.1.1 - Get stitched patient grid
        if verbose: t0 = time.time()
        patient_pred_ent          = patient_pred_ent/patient_pred_overlap     # [H,W,D]/[H,W,D]
        patient_pred_mif          = patient_pred_mif/patient_pred_overlap
        patient_pred_overlap      = np.expand_dims(patient_pred_overlap, -1)
        patient_pred              = patient_pred_vals/patient_pred_overlap    # [H,W,D,C]/[H,W,D,1]
        patient_pred_std          = patient_pred_std/patient_pred_overlap
        patient_pred_unc          = patient_pred_unc/patient_pred_overlap
        del patient_pred_vals
        del patient_pred_overlap
        patient_pred_unc = np.take_along_axis(patient_pred_unc, np.expand_dims(np.argmax(patient_pred, axis=-1),axis=-1), axis=-1)[:,:,:,0]

        gc.collect() # returns number of unreachable objects collected by GC
        patient_pred_postprocessed = losses.remove_smaller_components(patient_gt, patient_pred, meta=patient_id_curr, label_ids_small = [2,4,5])
        if verbose: print (' - [eval_3D_process_outputs()] Post-Process time    : ', time.time() - t0,'s')
        
        # Step 3.1.2 - Loss Calculation
        spacing = np.array([meta1_batch[4], meta1_batch[5], meta1_batch[6]])/100.0
        try:
            if verbose: t0 = time.time()
            loss_avg_val, loss_labels_val           = losses.dice_numpy(patient_gt, patient_pred_postprocessed)
            hausdorff_avg_val, hausdorff_labels_val, hausdorff95_avg_val, hausdorff95_labels_val, msd_avg_val, msd_labels_vals = losses.get_surface_distances(patient_gt, patient_pred_postprocessed, spacing)
            if verbose:
                print (' - [eval_3D_process_outputs()] DICE                 : ', ['%.4f' % (each) for each in loss_labels_val])
                print (' - [eval_3D_process_outputs()] HD95                 : ', ['%.4f' % (each) for each in hausdorff95_labels_val])

            if loss_avg_val != -1 and len(loss_labels_val):
                res[patient_id_curr] = {
                    config.KEY_DICE_AVG      : loss_avg_val
                    , config.KEY_DICE_LABELS : loss_labels_val
                    , config.KEY_HD_AVG      : hausdorff95_avg_val
                    , config.KEY_HD_LABELS   : hausdorff_labels_val 
                    , config.KEY_HD95_AVG    : hausdorff95_avg_val
                    , config.KEY_HD95_LABELS : hausdorff95_labels_val
                    , config.KEY_MSD_AVG     : msd_avg_val
                    , config.KEY_MSD_LABELS  : msd_labels_vals
                }
            else:
                print (' - [INFO][eval_3D_process_outputs()] patient_id: ', patient_id_curr)
            if verbose: print (' - [eval_3D_process_outputs()] Loss calculation time: ', time.time() - t0,'s')
        except:
            traceback.print_exc()
            if DEBUG: pdb.set_trace()

        # Step 3.1.3 - ECE calculation
        if verbose: t0 = time.time()
        ece_global_obj, ece_patient_obj = get_ece(patient_gt, patient_pred, patient_id_curr, ece_global_obj)
        res[patient_id_curr][config.KEY_ECE_LABELS] = ece_patient_obj
        if verbose: print (' - [eval_3D_process_outputs()] ECE time             : ', time.time() - t0,'s')

        # Step 3.1.5 - Save/Visualize
        if not deepsup_eval:
            if verbose: t0 = time.time()
            eval_3D_finalize(exp_name, patient_img, patient_gt, patient_pred_postprocessed, patient_pred, patient_pred_std, patient_pred_ent, patient_pred_mif, patient_pred_unc
                , patient_id_curr
                , model_folder_epoch_patches 
                , loss_labels_val, hausdorff_labels_val, hausdorff95_labels_val, msd_labels_vals
                , spacing, label_map
                , save=save)
            if verbose: print (' - [eval_3D_process_outputs()] Save as .nrrd time   : ', time.time() - t0,'s')

        if verbose: print (' - [eval_3D_process_outputs()] Total patient time   : ', time.time() - t99,'s')
            
        # Step 3.1.6
        del patient_img
        del patient_gt
        del patient_pred
        del patient_pred_std
        del patient_pred_ent
        del patient_pred_postprocessed
        del patient_pred_mif
        del patient_pred_unc
        gc.collect()

        return res, ece_global_obj
    
    except:
        traceback.print_exc()
        if DEBUG: pdb.set_trace()
        return res, ece_global_obj

def eval_3D_get_outputs(model, X, Y, training_bool, MC_RUNS, deepsup, deepsup_eval):

    # Step 0 - Init
    DO_KEYS = [config.KEY_MIF, config.KEY_ENT]
    # DO_KEYS = [config.KEY_STD]
    
    # Step 1 - Warm up model
    _           = model(X, training=training_bool)

    # Step 2 - Run Monte-Carlo predictions
    error_mcruns   = False
    try:
        tic_mcruns  = time.time()
        y_predict = tf.stack([model(X, training=training_bool) for _ in range(MC_RUNS)]) # [MC,B,H,W,D,C] # [Note] with model.trainable=False and model.training=True we get dropout at inference time
        toc_mcruns        = time.time()
    except tf.errors.ResourceExhaustedError as e:
        print (' - [eval_3D_get_outputs()] OOM error for MC_RUNS={}'.format(MC_RUNS))
        error_mcruns = True
    
    if error_mcruns:
        try:
            MC_RUNS     = 5
            tic_mcruns  = time.time()
            if deepsup:
                if deepsup_eval:
                    y_predict = tf.stack([model(X, training=training_bool)[0] for _ in range(MC_RUNS)]) # [MC,B,H,W,D,C] # [Note] with model.trainable=False and model.training=True we get dropout at inference time
                    X = X[:,::2,::2,::2,:]
                    Y = Y[:,::2,::2,::2,:]
                else:
                    y_predict = tf.stack([model(X, training=training_bool)[1] for _ in range(MC_RUNS)]) # [MC,B,H,W,D,C] # [Note] with model.trainable=False and model.training=True we get dropout at inference time    
            else:
                y_predict = tf.stack([model(X, training=training_bool) for _ in range(MC_RUNS)]) # [MC,B,H,W,D,C] # [Note] with model.trainable=False and model.training=True we get dropout at inference time
            toc_mcruns        = time.time()
        except tf.errors.ResourceExhaustedError as e:
            print (' - [eval_3D_get_outputs()] OOM error for MC_RUNS=5')
            import sys; sys.exit(1)

    # Step 3 - Calculate different metrics
    if config.KEY_MIF in DO_KEYS:
        y_predict_mif     = y_predict * tf.math.log(y_predict + _EPSILON)               # [MC,B,H,W,D,C]
        y_predict_mif     = tf.math.reduce_sum(y_predict_mif, axis=[0,-1])/MC_RUNS      # [MC,B,H,W,D,C] -> [B,H,W,D]
    else:
        y_predict_mif = []
    
    if config.KEY_STD in DO_KEYS:
        y_predict_std     = tf.math.reduce_std(y_predict, axis=0)                       # [MC,B,H,W,D,C] -> [B,H,W,D,C]
    else:
        y_predict_std     = []
    
    if config.KEY_PERC in DO_KEYS:
        y_predict_perc    = tfp.stats.percentile(y_predict, q=[30,70], axis=0, interpolation='nearest')
        y_predict_unc     = y_predict_perc[1] - y_predict_perc[0]
        del y_predict_perc
        gc.collect()
    else:
        y_predict_unc     = []
    
    y_predict         = tf.math.reduce_mean(y_predict, axis=0)    # [MC,B,H,W,D,C] -> [B,H,W,D,C]

    if config.KEY_ENT in DO_KEYS:
        y_predict_ent     = -1*tf.math.reduce_sum(y_predict * tf.math.log(y_predict + _EPSILON), axis=-1) # [B,H,W,D,C] -> # [B,H,W,D] ent = -p.log(p)
        y_predict_mif     = y_predict_ent + y_predict_mif                                                 # [B,H,W,D] + [B,H,W,D] = [B,H,W,D]; MI = ent + expectation(ent)
    else:
        y_predict_ent     = []
        y_predict_mif     = []

    return Y, y_predict, y_predict_std, y_predict_ent, y_predict_mif, y_predict_unc, toc_mcruns-tic_mcruns

def eval_3D(model, dataset_eval, dataset_eval_gen, params, show=False, save=False, verbose=False):
    
    try:
        
        # Step 0.0 - Variables under debugging
        pass

        # Step 0.1 - Extract params
        PROJECT_DIR  = config.PROJECT_DIR
        exp_name     = params['exp_name']
        pid          = params['pid']
        eval_type    = params['eval_type']
        batch_size   = 2
        epoch        = params['epoch']
        deepsup      = params['deepsup']
        deepsup_eval = params['deepsup_eval']
        label_map    = dict(dataset_eval.get_label_map())
        label_colors = dict(dataset_eval.get_label_colors())

        if verbose: print (''); print (' --------------------- eval_3D({}) ---------------------'.format(eval_type))
        
        # Step 0.2 - Init results array
        res                = {}
        ece_global_obj     = {}
        patient_grid_count = {}

        # Step 0.3 - Init temp variables
        patient_id_curr            = None
        w_grid, h_grid, d_grid     = None, None, None
        meta1_batch                = None
        patient_gt                 = None
        patient_img                = None
        patient_pred_overlap       = None
        patient_pred_vals          = None
        model_folder_epoch_patches = None
        model_folder_epoch_imgs    = None
        
        mc_runs       = params.get(config.KEY_MC_RUNS, None)
        training_bool = params.get(config.KEY_TRAINING_BOOL, None)
        model_folder_epoch_patches, model_folder_epoch_imgs = modutils.get_eval_folders(PROJECT_DIR, exp_name, epoch, eval_type, mc_runs, training_bool, create=True)

        # Step 0.4 - Debug vars
        ttotal,t0, t99         = time.time(), None, None
        times_mcruns           = []
        
        # Step 1 - Loop over dataset_eval (which provides patients & grids in an ordered manner)
        print ('')
        model.trainable  = False
        pbar_desc_prefix = 'Eval3D_{} [batch={}]'.format(eval_type, batch_size)
        training_bool    = params.get(config.KEY_TRAINING_BOOL,True) # [True, False]
        with tqdm.tqdm(total=len(dataset_eval), desc=pbar_desc_prefix, leave=False) as pbar_eval:
            for (X,Y,meta1,meta2) in dataset_eval_gen.repeat(1):
                
                MC_RUNS     = params.get(config.KEY_MC_RUNS,config.VAL_MC_RUNS_DEFAULT)
                Y, y_predict, y_predict_std, y_predict_ent, y_predict_mif, y_predict_unc, mcruns_time = eval_3D_get_outputs(model, X, Y, training_bool, MC_RUNS, deepsup, deepsup_eval)
                times_mcruns.append(mcruns_time)

                for batch_id in range(X.shape[0]):

                    # Step 2 - Get grid info
                    patient_id_running = meta2[batch_id].numpy().decode('utf-8')
                    if patient_id_running in patient_grid_count: patient_grid_count[patient_id_running] += 1
                    else: patient_grid_count[patient_id_running] = 1

                    meta1_batch = meta1[batch_id].numpy()
                    w_start, h_start, d_start = meta1_batch[1], meta1_batch[2], meta1_batch[3]
                    
                    # Step 3 - Check if its a new patient
                    if patient_id_running != patient_id_curr:

                        # Step 3.1 - Sort out old patient (patient_id_curr)
                        if patient_id_curr != None:

                            res, ece_global_obj = eval_3D_process_outputs(exp_name, res, ece_global_obj, patient_id_curr, meta1_batch, patient_img, patient_gt, patient_pred_vals, patient_pred_overlap, patient_pred_std, patient_pred_ent, patient_pred_mif, patient_pred_unc
                                    , deepsup_eval, model_folder_epoch_imgs, model_folder_epoch_patches, label_map, label_colors, t99, show=show, save=save, verbose=verbose)
                            
                        # Step 3.2 - Create variables for new patient
                        if verbose: t99 = time.time()
                        patient_id_curr   = patient_id_running
                        patient_scan_size = meta1_batch[7:10]
                        dataset_name      = patient_id_curr.split('-')[0]
                        dataset_this      = dataset_eval.get_subdataset(param_name=dataset_name)
                        w_grid, h_grid, d_grid = dataset_this.w_grid, dataset_this.h_grid, dataset_this.d_grid
                        if deepsup_eval: 
                            patient_scan_size      = patient_scan_size//2
                            w_grid, h_grid, d_grid = w_grid//2, h_grid//2, d_grid//2

                        patient_pred_size      = list(patient_scan_size) + [len(dataset_this.LABEL_MAP)]
                        patient_pred_overlap   = np.zeros(patient_scan_size, dtype=np.uint8)
                        patient_pred_ent       = np.zeros(patient_scan_size, dtype=np.float32)
                        patient_pred_mif       = np.zeros(patient_scan_size, dtype=np.float32)
                        patient_pred_vals      = np.zeros(patient_pred_size, dtype=np.float32)
                        patient_pred_std       = np.zeros(patient_pred_size, dtype=np.float32)
                        patient_pred_unc       = np.zeros(patient_pred_size, dtype=np.float32)
                        patient_gt             = np.zeros(patient_pred_size, dtype=np.float32)
                        if show or save:
                            patient_img = np.zeros(list(patient_scan_size) + [1], dtype=np.float32)
                        else:
                            patient_img = []

                    # Step 4 - If not new patient anymore, fill up data
                    if deepsup_eval:
                        w_start, h_start, d_start = w_start//2, h_start//2, d_start//2
                    patient_pred_vals[w_start:w_start + w_grid, h_start:h_start+h_grid, d_start:d_start+d_grid]    += y_predict[batch_id]
                    if len(y_predict_std):
                        patient_pred_std[w_start:w_start + w_grid, h_start:h_start+h_grid, d_start:d_start+d_grid] += y_predict_std[batch_id]
                    if len(y_predict_ent):
                        patient_pred_ent[w_start:w_start + w_grid, h_start:h_start+h_grid, d_start:d_start+d_grid]     += y_predict_ent[batch_id]
                    if len(y_predict_mif):
                        patient_pred_mif[w_start:w_start + w_grid, h_start:h_start+h_grid, d_start:d_start+d_grid]     += y_predict_mif[batch_id]
                    if len(y_predict_unc):
                        patient_pred_unc[w_start:w_start + w_grid, h_start:h_start+h_grid, d_start:d_start+d_grid] += y_predict_unc[batch_id]
                    
                    patient_pred_overlap[w_start:w_start + w_grid, h_start:h_start+h_grid, d_start:d_start+d_grid] += np.ones(y_predict[batch_id].shape[:-1], dtype=np.uint8)
                    patient_gt[w_start:w_start+w_grid, h_start:h_start+h_grid, d_start:d_start+d_grid] = Y[batch_id]
                    if show or save:
                        patient_img[w_start:w_start+w_grid, h_start:h_start+h_grid, d_start:d_start+d_grid] = X[batch_id]
        
                pbar_eval.update(batch_size)
                mem_used = modutils.get_memory(pid)
                memory = pbar_desc_prefix + ' [' + mem_used + ']'
                pbar_eval.set_description(desc=memory, refresh=True)

        # Step 3 - For last patient
        res, ece_global_obj = eval_3D_process_outputs(exp_name, res, ece_global_obj, patient_id_curr, meta1_batch, patient_img, patient_gt, patient_pred_vals, patient_pred_overlap, patient_pred_std, patient_pred_ent, patient_pred_mif, patient_pred_unc
                                    , deepsup_eval, model_folder_epoch_imgs, model_folder_epoch_patches, label_map, label_colors, t99, show=show, save=save, verbose=verbose)
        print ('\n - [eval_3D()] Time passed to accumulate grids & process patients: ', round(time.time() - ttotal, 2), 's')

        return eval_3D_summarize(res, ece_global_obj, model, eval_type, deepsup_eval, label_map, model_folder_epoch_patches, times_mcruns, ttotal, save=save, show=show, verbose=verbose)

    except:
        traceback.print_exc()
        if DEBUG: pdb.set_trace()
        model.trainable=True
        return -1, {} 

############################################################
#                           VAL                            #
############################################################

class Validator:

    def __init__(self, params):
        
        self.params = params

        # Dataloader Params
        self.data_dir  = self.params['dataloader']['data_dir']
        self.dir_type  = self.params['dataloader']['dir_type']
        self.grid      = self.params['dataloader']['grid']
        self.crop_init = self.params['dataloader']['crop_init']
        self.resampled = self.params['dataloader']['resampled']

        self._get_dataloader()
        self._get_model()
    
    def _set_dataloader(self):
        
        # Step 1 - Init params
        batch_size = self.params['dataloader']['batch_size']
        prefetch_batch = self.params['dataloader']['prefetch_batch']

        # Step 2 - Load dataset
        if self.dir_type  == config.DATALOADER_MICCAI2015_TEST:
            
            self.dataset_test_eval = datautils.get_dataloader_3D_test_eval(self.data_dir, dir_type=self.dir_type
                            , grid=self.grid, crop_init=self.crop_init
                            , resampled=self.resampled
                        )

        elif self.dir_type == config.DATALOADER_DEEPMINDTCIA_TEST:
            
            self.annotator_type = self.params['dataloader']['annotator_type']
            self.dataset_test_eval = datautils.get_dataloader_deepmindtcia(self.data_dir, dir_type=self.dir_type, annotator_type=self.annotator_type
                                                        , grid=self.grid, crop_init=self.crop_init, resampled=self.resampled
                                    )
        
        # Step 3 - Load dataloader/datagenerator
        self.datagen_test_eval = self.dataset_test_eval.generator().batch(batch_size).prefetch(prefetch_batch)

    def _set_model(self):

        # Step 1 - Init params
        exp_name    = self.params['exp_name']
        load_epoch  = self.params['model']['load_epoch']
        class_count = len(self.dataset_test_eval.datasets[0].LABEL_MAP.values())

        # Step 2 - Get model arch
        if self.params['model']['name'] == config.MODEL_FOCUSNET_DROPOUT:
            print (' - [Trainer][_models()] ModelFocusNetDropOut')
            self.model = models.ModelFocusNetDropOut(class_count=class_count, trainable=False)

        # Step 3 - Load model
        load_model_params = {'PROJECT_DIR': config.PROJECT_DIR
                                , 'exp_name': exp_name
                                , 'load_epoch': load_epoch
                                , 'optimizer': tf.keras.optimizers.Adam()
                            }
        modutils.load_model(self.model, load_type=config.MODE_TRAIN, params=load_model_params)
        print ('')
        print (' - [train.py][val()] Model({}) Loaded for {} at epoch-{} (validation purposes) !'.format(str(self.model), exp_name, load_epoch))
        print ('')

    def validate(self, verbose=False):
        
        save = self.params['save']
        loss_avg, loss_labels_avg = eval_3D(self.model, self.dataset_test_eval, self.datagen_test_eval, self.params, save=save, verbose=verbose)
    
############################################################
#                         TRAINER                          #
############################################################

class Trainer:

    def __init__(self, params):

        # Init
        self.params = params

        # Print
        self._train_preprint()

        # Random Seeds
        self._set_seed()

        # Set the dataloaders
        self._set_dataloaders()

        # Set the model
        self._set_model()

        # Set Metrics
        self._set_metrics()

        # Other flags
        self.write_model_done = False
        
    def _train_preprint(self):
        print ('')
        print (' -------------- {}    ({})'.format(self.params['exp_name'], str(datetime.datetime.now())))
        
        print ('')
        print (' DATALOADER ')
        print (' ---------- ')
        print (' - dir_type: ', self.params['dataloader']['dir_type'])

        print (' -- resampled: ', self.params['dataloader']['resampled'])
        print (' -- crop_init: ', self.params['dataloader']['crop_init'])
        print (' -- grid: ', self.params['dataloader']['grid'])
        print ('  --- filter_grid  : ', self.params['dataloader']['filter_grid'])
        print ('  --- random_grid  : ', self.params['dataloader']['random_grid'])
        print ('  --- centred_prob : ', self.params['dataloader']['centred_prob'])

        print (' -- batch_size: ', self.params['dataloader']['batch_size'])
        print ('  -- prefetch_batch : ', self.params['dataloader']['prefetch_batch'])
        print ('  -- parallel_calls : ', self.params['dataloader']['parallel_calls'])
        print ('  -- shuffle        : ', self.params['dataloader']['shuffle'])

        print ('')
        print (' MODEL ')
        print (' ----- ')
        print (' - Model: ', str(self.params['model']['name']))
        print (' -- Model TBoard: ', self.params['model']['model_tboard'])
        print (' -- Profiler: ', self.params['model']['profiler']['profile'])
        if self.params['model']['profiler']['profile']:
            print (' ---- Profiler Epochs: ', self.params['model']['profiler']['epochs'])
            print (' ---- Step Per Epochs: ', self.params['model']['profiler']['steps_per_epoch'])
        print (' - Optimizer: ', str(self.params['model']['optimizer']))
        print (' -- Init LR: ', self.params['model']['init_lr'])
        print (' -- Fixed LR: ', self.params['model']['fixed_lr'])
        
        print (' - Epochs: ', self.params['model']['epochs'])
        print (' -- Save   : every {} epochs'.format(self.params['model']['epochs_save']))
        print (' -- Eval3D : every {} epochs '.format(self.params['model']['epochs_eval']))
        print (' -- Viz3D  : every {} epochs '.format(self.params['model']['epochs_viz']))

        print ('')
        print (' METRICS ')
        print (' ------- ') 
        print (' - Logging-TBoard : ', self.params['metrics']['logging_tboard'])
        if not self.params['metrics']['logging_tboard']:
            print (' !!!!!!!!!!!!!!!!!!! NO LOGGING-TBOARD !!!!!!!!!!!!!!!!!!!')
            print ('')
        print (' - Eval           : ', self.params['metrics']['metrics_eval'])
        print (' - Loss           : ', self.params['metrics']['metrics_loss'])
        print (' -- Weighted Loss : ', self.params['metrics']['loss_weighted'])
        print (' -- Masked Loss   : ', self.params['metrics']['loss_mask'])
        print (' -- Combo         : ', self.params['metrics']['loss_combo'])

        print ('')
        print (' DEVOPS ')
        print (' ------ ')
        self.pid = os.getpid()
        print (' - OS-PID: ', self.pid)
        print (' - Seed: ', self.params['random_seed'])

        print ('')

    def _set_seed(self):
        np.random.seed(self.params['random_seed'])
        tf.random.set_seed(self.params['random_seed'])
    
    def _set_dataloaders(self):
        
        print ('')
        print (' DATALOADER OBJECTS')
        print (' ---------- ')

        # Params - Directories
        data_dir = self.params['dataloader']['data_dir']
        dir_type = self.params['dataloader']['dir_type']
        dir_type_eval = ['_'.join(dir_type)]

        # Params - Single volume
        resampled = self.params['dataloader']['resampled']
        crop_init = self.params['dataloader']['crop_init']
        grid = self.params['dataloader']['grid']
        filter_grid = self.params['dataloader']['filter_grid']
        random_grid = self.params['dataloader']['random_grid']
        centred_prob = self.params['dataloader']['centred_prob']

        # Params - Dataloader        
        batch_size = self.params['dataloader']['batch_size']
        prefetch_batch = self.params['dataloader']['prefetch_batch']
        parallel_calls = self.params['dataloader']['parallel_calls']
        shuffle_size = self.params['dataloader']['shuffle']
        
        # Params - Debug
        pass
        
        # Datasets
        self.dataset_train = datautils.get_dataloader_3D_train(data_dir, dir_type=dir_type
                                , grid=grid, crop_init=crop_init, filter_grid=filter_grid
                                , random_grid=random_grid
                                , resampled=resampled
                                , parallel_calls=parallel_calls
                                , centred_dataloader_prob=centred_prob
                                )
        self.dataset_train_eval = datautils.get_dataloader_3D_train_eval(data_dir, dir_type=dir_type_eval
                                    , grid=grid, crop_init=crop_init
                                    , resampled=resampled
                                    )
        self.dataset_test_eval = datautils.get_dataloader_3D_test_eval(data_dir
                                    , grid=grid, crop_init=crop_init
                                    , resampled=resampled
                                    )

        # Get labels Ids
        self.label_map = dict(self.dataset_train.get_label_map())
        self.label_ids = self.label_map.values()
        self.params['internal'] = {}
        self.params['internal']['label_map'] = self.label_map # for use in Metrics
        self.params['internal']['label_ids'] = self.label_ids # for use in Metrics
        self.label_weights = list(self.dataset_train.get_label_weights())

        # Generators
        # repeat() -> shuffle() --> batch() --> prefetch(): endlessly-output-data --> shuffle-within-buffer --> create-batch-from-shuffle --> prefetch-enough-batches
        self.dataset_train_gen = self.dataset_train.generator().repeat().shuffle(shuffle_size).batch(batch_size).apply(tf.data.experimental.prefetch_to_device(device='/GPU:0', buffer_size=prefetch_batch))
        self.dataset_train_eval_gen = self.dataset_train_eval.generator().batch(2).prefetch(4)
        self.dataset_test_eval_gen = self.dataset_test_eval.generator().batch(2).prefetch(4)
        
    def _set_model(self):

        print ('')
        print (' MODEL OBJECTS')
        print (' ---------- ')

        # Step 1 - Get class ids
        class_count = len(self.label_ids)
        
        # Step 2 - Get model arch
        if self.params['model']['name'] == config.MODEL_FOCUSNET_DROPOUT:
            print (' - [Trainer][_models()] ModelFocusNetDropOut')
            self.model = models.ModelFocusNetDropOut(class_count=class_count, trainable=True)
            self.kl_alpha_init = 0.0

        elif self.params['model']['name'] == config.MODEL_FOCUSNET_FLIPOUT:
            print (' - [Trainer][_models()] ModelFocusNetFlipOut')
            self.model = models.ModelFocusNetFlipOut(class_count=class_count, trainable=True)
            self.kl_schedule = self.params['model']['kl_schedule']
            self.kl_alpha_init = 0.1
            self.kl_alpha_increase_per_epoch = 0.0001 
            self.initial_epoch               = 250

        # Step 3 - Get optimizer
        if self.params['model']['optimizer'] == config.OPTIMIZER_ADAM:
            self.optimizer = tf.keras.optimizers.Adam(learning_rate=self.params['model']['init_lr'])

        # Step 4 - Load model (if needed)
        epochs = self.params['model']['epochs']
        if not self.params['model']['load_model']['load']:  
            # Step 4.1 - Set epoch range under non-loading situations
            self.epoch_range = range(1,epochs+1)
        else:

            # Step 4.2.1 - Some model-loading params
            load_epoch        = self.params['model']['load_model']['load_epoch']
            load_exp_name     = self.params['model']['load_model']['load_exp_name']
            load_optimizer_lr = self.params['model']['load_model']['load_optimizer_lr']
            load_model_params = {'PROJECT_DIR': config.PROJECT_DIR, 'load_epoch': load_epoch, 'optimizer':self.optimizer}
            
            print ('')
            print (' - [Trainer][_set_model()] Loading pretrained model')
            print (' - [Trainer][_set_model()] Model: ', self.model)
            
            # Step 4.2.2.1 - If loading is done from the same exp_name
            if load_exp_name is None:
                load_model_params['exp_name'] = self.params['exp_name']
                self.epoch_range = range(load_epoch+1, epochs+1)
                print (' - [Trainer][_set_model()] Training from epoch:{} to {}'.format(load_epoch, epochs))
            # Step 4.2.2.1 - If loading is done from another exp_name
            else:
                self.epoch_range = range(1, epochs+1)
                load_model_params['exp_name'] = load_exp_name
                print (' - [Trainer][_set_model()] Training from epoch:{} to {}'.format(1, epochs))

            print (' - [Trainer][_set_model()] exp_name: ', load_model_params['exp_name'])

            # Step 4.3 - Finally, load model from the checkpoint
            modutils.load_model(self.model, load_type=config.MODE_TRAIN, params=load_model_params)
            print (' - [Trainer][_set_model()] Model Loaded at epoch-{} !'.format(load_epoch))
            print (' -- [Trainer][_set_model()] Optimizer.lr : ', self.optimizer.lr.numpy())
            if load_optimizer_lr is not None:
                self.optimizer.lr.assign(load_optimizer_lr)
                print (' -- [Trainer][_set_model()] Optimizer.lr : ', self.optimizer.lr.numpy())
        
        # Step 5 - Create model weights
        _ = self.model(tf.random.normal((1,140,140,40,1))) # [NOTE: This needs to be the same as the input grid size in FlipOut]
        print (' -- [Trainer][_set_model()] Created model weights ')
        try:
            print (' --------------------------------------- ')
            print (self.model.summary(line_length=150))
            print (' --------------------------------------- ')
            count = 0
            for var in self.model.trainable_variables:
                print (' - var: ', var.name)
                count += 1
                if count > 20:
                    print (' ... ')
                    break 
        except:
            print (' - [Trainer][_set_model()] model.summary() failed')
            pass
        
    def _set_metrics(self):
        self.metrics = {}
        self.metrics[config.MODE_TRAIN] = ModelMetrics(metric_type=config.MODE_TRAIN, params=self.params)
        self.metrics[config.MODE_TEST]  = ModelMetrics(metric_type=config.MODE_TEST, params=self.params)

    def _set_profiler(self, epoch, epoch_step):
        """
        - Ref: https://www.tensorflow.org/guide/data_performance_analysis#analysis_workflow
        """
        exp_name = self.params['exp_name']

        if self.params['model']['profiler']['profile']:
            if epoch in self.params['model']['profiler']['epochs']:
                if epoch_step == self.params['model']['profiler']['starting_step']:
                    self.logdir = Path(config.MODEL_CHKPOINT_MAINFOLDER).joinpath(exp_name, config.MODEL_LOGS_FOLDERNAME, 'profiler', str(epoch))
                    tf.profiler.experimental.start(str(self.logdir))
                    print (' - tf.profiler.experimental.start(logdir)')
                    print ('')
                elif epoch_step == self.params['model']['profiler']['starting_step'] + self.params['model']['profiler']['steps_per_epoch']:
                    print (' - tf.profiler.experimental.stop()')
                    tf.profiler.experimental.stop()
                    print ('')

    @tf.function
    def _train_loss(self, Y, y_predict, meta1, mode):

        trainMetrics  = self.metrics[mode]
        metrics_loss  = self.params['metrics']['metrics_loss']
        loss_weighted = self.params['metrics']['loss_weighted']
        loss_mask     = self.params['metrics']['loss_mask']
        loss_combo    = self.params['metrics']['loss_combo']

        label_ids = self.label_ids
        label_weights = tf.cast(self.label_weights, dtype=tf.float32)

        loss_vals = tf.constant(0.0, dtype=tf.float32)
        mask = losses.get_mask(meta1[:,-len(label_ids):], Y)

        inf_flag = False
        nan_flag = False

        for metric_str in metrics_loss:
            
            weights = []
            if loss_weighted[metric_str]:
                weights = label_weights
            
            if not loss_mask[metric_str]:
                mask = tf.cast(tf.cast(mask + 1, dtype=tf.bool), dtype=tf.float32)
            
            if metrics_loss[metric_str] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_CE_BASIC]:   
                loss_val_train, loss_labellist_train, metric_val_report, metric_labellist_report = trainMetrics.losses_obj[metric_str](Y, y_predict, mask, weights=weights)
                if metrics_loss[metric_str] in [config.LOSS_DICE, config.LOSS_CE, config.LOSS_CE_BASIC]:
                    nan_list = tf.math.is_nan(loss_labellist_train)
                    nan_val  = tf.math.is_nan(loss_val_train)
                    inf_list = tf.math.is_inf(loss_labellist_train)
                    inf_val  = tf.math.is_inf(loss_val_train)
                    if nan_val or tf.math.reduce_any(nan_list):
                        nan_flag = True
                        tf.print (' - [ERROR][Trainer][_train_loss()] Loss NaN spotted: ', metric_str, ' || nan_list: ', nan_list, ' || nan_val: ', nan_val)
                        tf.print (' - [ERROR][Trainer][_train_loss()] Loss NaN spotted: ', metric_str, ' || mask: ', mask)
                        tf.print (' - [ERROR][Trainer][_train_loss()] Loss NaN spotted: ', metric_str, ' || loss_vals: ', loss_vals)
                    elif inf_val or tf.math.reduce_any(inf_list):
                        inf_flag = True
                        tf.print (' - [ERROR][Trainer][_train_loss()] Loss Inf spotted: ', metric_str, ' || loss_val_train: ', loss_val_train)
                        tf.print (' - [ERROR][Trainer][_train_loss()] Loss Inf spotted: ', metric_str, ' || inf_list: ', inf_list, ' || inf_val: ', inf_val)
                        tf.print (' - [ERROR][Trainer][_train_loss()] Loss Inf spotted: ', metric_str, ' || mask: ', mask)
                        tf.print (' - [ERROR][Trainer][_train_loss()] Loss NaN spotted: ', metric_str, ' || loss_vals: ', loss_vals)
                    else:
                        if len(metric_labellist_report):
                            trainMetrics.update_metric_loss_labels(metric_str, metric_labellist_report) # in sub-3D settings, this value is only indicative of performance
                        trainMetrics.update_metric_loss(metric_str, loss_val_train)

                        if 0: # [DEBUG]
                            tf.print(' - metric_str: ', metric_str, ' || loss_val_train: ', loss_val_train)

                        if metric_str in loss_combo:
                            loss_val_train = loss_val_train*loss_combo[metric_str]
                        loss_vals = tf.math.add(loss_vals, loss_val_train) # Averaged loss


        if nan_flag or inf_flag:
            loss_vals = 0.0 # no backprop when something was wrong

        return loss_vals
        
    @tf.function
    def _train_step(self, X, Y, meta1, meta2, kl_alpha):

        try:
            model           = self.model
            optimizer       = self.optimizer
            if self.params['model']['name'] == config.MODEL_FOCUSNET_FLIPOUT:
                kl_scale_fac    = self.params['model']['kl_scale_factor']
            else:
                kl_scale_fac = 0.0 
            
            y_predict         = None
            loss_vals         = None
            gradients         = None

            # Step 1 - Calculate loss
            with tf.GradientTape() as tape:
                
                t2 = tf.timestamp()
                y_predict = model(X, training=True)
                t2_ = tf.timestamp()

                loss_vals = self._train_loss(Y, y_predict, meta1, mode=config.MODE_TRAIN)

                if self.params['model']['name'] == config.MODEL_FOCUSNET_FLIPOUT:
                    print (' - [Trainer][_train_step()] Model FlipOut')
                    kl           = tf.math.add_n(model.losses)
                    kl_loss      = kl*kl_alpha/kl_scale_fac
                    loss_vals    = loss_vals + kl_loss
                
            # Step 2 - Calculate gradients
            t3 = tf.timestamp()
            if not tf.math.reduce_any(tf.math.is_nan(loss_vals)):
                all_vars = model.trainable_variables

                gradients = tape.gradient(loss_vals, all_vars) # dL/dW
                
                # Step 3 - Apply gradients
                optimizer.apply_gradients(zip(gradients, all_vars))

            else:
                tf.print('\n ====================== [NaN Error] ====================== ')
                tf.print(' - [ERROR][Trainer][_train_step()] Loss NaN spotted || loss_vals: ', loss_vals)
                tf.print(' - [ERROR][Trainer][_train_step()] meta2: ', meta2, ' || meta1: ', meta1)

            t3_ = tf.timestamp()
            return t2_-t2, t3-t2_, t3_-t3

        except tf.errors.ResourceExhaustedError as e:
            tf.print('\n ====================== [OOM Error] ====================== ')
            tf.print(' - [ERROR][Trainer][_train_step()]  meta2: ', meta2, ' || meta1: ', meta1)
            traceback.print_exc()
            return None, None, None

        except:
            tf.print('\n ====================== [Some Error] ====================== ')
            tf.print(' - [ERROR][Trainer][_train_step()]  meta2: ', meta2, ' || meta1: ', meta1)
            traceback.print_exc()
            return None, None, None
    
    def train(self):

        # Global params
        exp_name = self.params['exp_name']

        # Dataloader params
        batch_size = self.params['dataloader']['batch_size']

        # Model/Training params
        fixed_lr     = self.params['model']['fixed_lr']
        init_lr      = self.params['model']['init_lr']
        max_epoch    = self.params['model']['epochs']
        epoch_range  = iter(self.epoch_range)
        epoch_length = len(self.dataset_train)
        params_save_model = {'PROJECT_DIR': config.PROJECT_DIR, 'exp_name': exp_name, 'optimizer':self.optimizer}

        # Metrics params
        metrics_eval = self.params['metrics']['metrics_eval']
        trainMetrics = self.metrics[config.MODE_TRAIN]

        # KL Divergence Params
        kl_alpha = self.kl_alpha_init # [0.0, self.kl_alpha_init]
        
        # Eval Params
        params_eval = {'PROJECT_DIR': config.PROJECT_DIR, 'exp_name': exp_name, 'pid': self.pid
                            , 'eval_type': config.MODE_TRAIN, 'batch_size': batch_size}
        
        # Viz params
        epochs_save = self.params['model']['epochs_save']
        epochs_viz = self.params['model']['epochs_viz']
        epochs_eval = self.params['model']['epochs_eval']
        
        # Random vars
        t_start_time = time.time()

        try:
            
            epoch_step = 0
            epoch = None
            pbar = None
            t1 = time.time()
            for (X,Y,meta1,meta2) in self.dataset_train_gen:
                t1_ = time.time()

                try:
                    # Epoch starter code                        
                    if epoch_step == 0:

                        # Get Epoch
                        epoch = next(epoch_range)

                        # Metrics
                        trainMetrics.reset_metrics(self.params)
                        
                        # LR
                        if not fixed_lr:
                            set_lr(epoch, self.optimizer, init_lr)
                        self.model.trainable = True

                        # Calculate kl_alpha (only for FlipOut model)
                        if self.params['model']['name'] == config.MODEL_FOCUSNET_FLIPOUT:
                            if self.kl_schedule == config.KL_DIV_ANNEALING:
                                if epoch > self.initial_epoch:
                                    if epoch % self.kl_epochs_change == 0:
                                        kl_alpha = tf.math.minimum(self.kl_alpha_max, self.kl_alpha_init + (epoch - self.initial_epoch)/float(self.kl_epochs_change) * self.kl_alpha_increase_per_epoch)   
                        else:
                            kl_alpha = 0.0

                        # Pretty print
                        print ('')
                        print (' ===== [{}] EPOCH:{}/{} (LR={:3f}) =================='.format(exp_name, epoch, max_epoch, self.optimizer.lr.numpy()))

                        # Start a fresh pbar
                        pbar = tqdm.tqdm(total=epoch_length, desc='')

                    # Model Writing to tensorboard
                    if self.params['model']['model_tboard'] and self.write_model_done is False :
                        self.write_model_done = True 
                        modutils.write_model(self.model, X, self.params)
                    
                    # Start/Stop Profiling (after dataloader is kicked off)
                    self._set_profiler(epoch, epoch_step)

                    # Calculate loss and gradients from them
                    time_predict, time_loss, time_backprop = self._train_step(X, Y, meta1, meta2, kl_alpha)

                    # Update metrics (time + eval + plots)
                    time_dataloader = t1_ - t1
                    trainMetrics.update_metrics_time(time_dataloader, time_predict, time_loss, time_backprop)
                            
                    # Update looping stuff
                    epoch_step += batch_size
                    pbar.update(batch_size)
                    trainMetrics.update_pbar(pbar)
                    
                except:
                    modutils.print_exp_name(exp_name + '-' + config.MODE_TRAIN, epoch)
                    params_save_model['epoch'] = epoch
                    modutils.save_model(self.model, params_save_model)
                    traceback.print_exc()
                    pdb.set_trace()

                if epoch_step >= epoch_length:
                    
                    # Reset epoch-loop params
                    pbar.close()
                    epoch_step = 0

                    try:
                        # Model save
                        if epoch % epochs_save == 0:
                            params_save_model['epoch'] = epoch
                            modutils.save_model(self.model, params_save_model)
                        
                        # Eval on full 3D
                        if epoch % epochs_eval == 0:
                            self.params['epoch'] = epoch
                            save=False
                            if epoch > 0 and epoch % epochs_viz == 0:
                                save=True

                            self.model.trainable = False
                            for metric_str in metrics_eval:
                                if metrics_eval[metric_str] in [config.LOSS_DICE]:
                                    params_eval['epoch']        = epoch
                                    params_eval['eval_type']    = config.MODE_TRAIN
                                    eval_avg, eval_labels_avg = eval_3D(self.model, self.dataset_train_eval, self.dataset_train_eval_gen, params_eval, save=save)
                                    trainMetrics.update_metric_eval_labels(metric_str, eval_labels_avg, do_average=True)

                        # Test
                        if epoch % epochs_eval == 0:
                            self._test()
                            self.model.trainable = True

                        # Epochs summary/wrapup
                        eval_condition = epoch % epochs_eval == 0
                        trainMetrics.write_epoch_summary(epoch, self.label_map, {'optimizer':self.optimizer}, eval_condition)

                        if epoch > 0 and epoch % self.params['others']['epochs_timer'] == 0:
                            elapsed_seconds =  time.time() - t_start_time
                            print (' - Total time elapsed : {}'.format( str(datetime.timedelta(seconds=elapsed_seconds)) ))
                        if epoch % self.params['others']['epochs_memory'] == 0:
                            mem_before = modutils.get_memory(self.pid)
                            gc_n = gc.collect()
                            mem_after = modutils.get_memory(self.pid)
                            print(' - Unreachable objects collected by GC: {} || ({}) -> ({})'.format(gc_n, mem_before, mem_after))

                        # Break out of loop at end of all epochs
                        if epoch == max_epoch:
                            print ('\n\n - [Trainer][train()] All epochs finished')
                            break

                    except:
                        modutils.print_exp_name(exp_name + '-' + config.MODE_TRAIN, epoch)
                        params_save_model['epoch'] = epoch
                        modutils.save_model(self.model, params_save_model)
                        traceback.print_exc()
                        pdb.set_trace()
                 
                t1 = time.time() # reset dataloader time calculator

        except:
            modutils.print_exp_name(exp_name + '-' + config.MODE_TRAIN, epoch)
            traceback.print_exc()
            pdb.set_trace()

    def _test(self):
        
        exp_name = None
        epoch    = None
        try:

            # Step 1.1 - Params
            exp_name = self.params['exp_name']
            epoch    = self.params['epoch']
            deepsup  = self.params['model']['deepsup']

            metrics_eval = self.params['metrics']['metrics_eval']
            epochs_viz   = self.params['model']['epochs_viz']
            batch_size   = self.params['dataloader']['batch_size']
            
            # vars
            testMetrics = self.metrics[config.MODE_TEST]
            testMetrics.reset_metrics(self.params)
            params_eval = {'PROJECT_DIR': config.PROJECT_DIR, 'exp_name': exp_name, 'pid': self.pid
                            , 'eval_type': config.MODE_TEST, 'batch_size': batch_size
                            , 'epoch':epoch}
                
            # Step 2 - Eval on full 3D
            save=False
            if epoch > 0 and epoch % epochs_viz == 0:
                save=True
            for metric_str in metrics_eval:
                if metrics_eval[metric_str] in [config.LOSS_DICE]:
                    params_eval['eval_type']    = config.MODE_TEST
                    eval_avg, eval_labels_avg = eval_3D(self.model, self.dataset_test_eval, self.dataset_test_eval_gen, params_eval, save=save)
                    testMetrics.update_metric_eval_labels(metric_str, eval_labels_avg, do_average=True)

            testMetrics.write_epoch_summary(epoch, self.label_map, {}, True)

        except:
            modutils.print_exp_name(exp_name + '-' + config.MODE_TEST, epoch)
            traceback.print_exc()
            pdb.set_trace()

