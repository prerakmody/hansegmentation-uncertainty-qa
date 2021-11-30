# Import internal libraries
import src.config as config

# Import external libraries
import os
import pdb
import copy
import tqdm
import json
import psutil
import humanize
import traceback
import numpy as np
import matplotlib
from pathlib import Path
import tensorflow as tf

from src.config import PROJECT_DIR

############################################################
#                    MODEL RELATED                         #
############################################################
def save_model(model, params):
    """
    The phrase "Saving a TensorFlow model" typically means one of two things:
        - using the Checkpoints format (this code does checkpointing)
        - using the SavedModel format.
     - Ref: https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
        - ckpt_obj = tf.train.Checkpoint(optimizer=optimizer, model=model); ckpt_obj.save(file_prefix='my_model_ckpt')
        - tf.keras.Model.save_weights('my_model_save_weights')
        - tf.keras.Model.save('my_model_save')
      - Questions
        - What is the SavedModel format
            - SavedModel saves the execution graph.  
    """
    try:
        PROJECT_DIR = params['PROJECT_DIR']
        exp_name = params['exp_name']
        epoch = params['epoch']

        folder_name = config.MODEL_CHKPOINT_NAME_FMT.format(epoch)
        model_folder = Path(PROJECT_DIR).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, exp_name, folder_name)
        model_folder.mkdir(parents=True, exist_ok=True)
        model_path = Path(model_folder).joinpath(folder_name)
        
        optimizer = params['optimizer']
        ckpt_obj = tf.train.Checkpoint(optimizer=optimizer, model=model)
        ckpt_obj.save(file_prefix=model_path)

        if 0: # CHECK
            model.save(str(Path(model_path).joinpath('model_save'))) # SavedModel format: creates a folder "model_save" with assets/, variables/ (contains weights) and a saved_model.pb (model architecture)
            model.save_weights(str(Path(model_path).joinpath('model_save_weights')))

    except:
        traceback.print_exc()
        pdb.set_trace()

def load_model(model, load_type, params):
    """
     - Ref: https://www.tensorflow.org/api_docs/python/tf/train/Checkpoint
    """
    try:
        
        PROJECT_DIR = params['PROJECT_DIR']
        exp_name    = params['exp_name']
        load_epoch  = params['load_epoch']

        folder_name = config.MODEL_CHKPOINT_NAME_FMT.format(load_epoch)
        model_folder = Path(PROJECT_DIR).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, exp_name, folder_name)
        
        if load_type == config.MODE_TRAIN:
            if 'optimizer' in params:
                ckpt_obj = tf.train.Checkpoint(optimizer=params['optimizer'], model=model)
                # ckpt_obj.restore(save_path=tf.train.latest_checkpoint(str(model_folder))).assert_existing_objects_matched() # shows errors
                # ckpt_obj.restore(save_path=tf.train.latest_checkpoint(str(model_folder))).assert_consumed()
                ckpt_obj.restore(save_path=tf.train.latest_checkpoint(str(model_folder))).expect_partial()
            else:
                print (' - [ERROR][utils.load_model] Optimizer not passed !')
                pdb.set_trace()

        elif load_type in [config.MODE_VAL, config.MODE_TEST]:
            ckpt_obj = tf.train.Checkpoint(model=model)
            ckpt_obj.restore(save_path=tf.train.latest_checkpoint(str(model_folder))).expect_partial()
        
        elif load_type == config.MODE_VAL_NEW:
            model.load_weights(str(Path(model_folder).joinpath('model.h5')), by_name=True, skip_mismatch=False)

        else:
            print (' - [ERROR][utils.load_model] It should not be here!')
            pdb.set_trace()
            # tf.keras.Model.load_weights
            # tf.train.list_variables(tf.train.latest_checkpoint(str(model_folder)))

    except:
        traceback.print_exc()
        pdb.set_trace()

def get_tensorboard_writer(exp_name, suffix):
    try:
        import tensorflow as tf

        logdir = Path(config.MODEL_CHKPOINT_MAINFOLDER).joinpath(exp_name, config.MODEL_LOGS_FOLDERNAME, suffix)
        writer = tf.summary.create_file_writer(str(logdir))
        return writer

    except:
        traceback.print_exc()
        pdb.set_trace()

def make_summary(fieldname, epoch, writer1=None, value1=None, writer2=None, value2=None):
    try:
        import tensorflow as tf

        if writer1 is not None and value1 is not None:
            with writer1.as_default():
                tf.summary.scalar(fieldname, value1, epoch)
                writer1.flush()
        if writer2 is not None and value2 is not None:
            with writer2.as_default():
                tf.summary.scalar(fieldname, value2, epoch)
                writer2.flush()
    except:
        traceback.print_exc()
        pdb.set_trace()

def make_summary_hist(fieldname, epoch, writer1=None, value1=None, writer2=None, value2=None):
    try:
        import tensorflow as tf

        if writer1 is not None and value1 is not None:
            with writer1.as_default():
                tf.summary.histogram(fieldname, value1, epoch)
                writer1.flush()
        if writer2 is not None and value2 is not None:
            with writer2.as_default():
                tf.summary.histogram(fieldname, value2, epoch)
                writer2.flush()
    except:
        traceback.print_exc()
        pdb.set_trace()

def write_model(model, X, params, suffix='model'):
    """
     - Ref:
        - https://www.tensorflow.org/api_docs/python/tf/summary/trace_on 
        - https://www.tensorflow.org/api_docs/python/tf/summary/trace_export
        - https://www.tensorflow.org/api_docs/python/tf/keras/utils/plot_model
        - https://stackoverflow.com/questions/56690089/how-to-graph-tf-keras-model-in-tensorflow-2-0
    """

    # Step 1 - Start trace
    tf.summary.trace_on(graph=True, profiler=False)

    # Step 2 - Perform operation
    _ = write_model_trace(model, X)

    # Step 3 - Export trace
    writer = get_tensorboard_writer(params['exp_name'], suffix)
    with writer.as_default():
        tf.summary.trace_export(name=model.name, step=0, profiler_outdir=None)
        writer.flush()

    # Step 4 - Save as .png
    # tf.keras.utils.plot_model(model, to_file='model.png', show_shapes=True, expand_nested=True) # only works for the functional API. :(

@tf.function
def write_model_trace(model, X):
    return model(X)

def set_lr(epoch, optimizer):
    # if epoch == 200:
    #     optimizer.lr.assign(0.0001)
    pass

############################################################
#                      WRITE RELATED                       #
############################################################

class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)

def write_json(json_data, json_filepath):

    Path(json_filepath).parent.absolute().mkdir(parents=True, exist_ok=True)

    with open(str(json_filepath), 'w') as fp:
        json.dump(json_data, fp, indent=4, cls=NpEncoder)

############################################################
#                      DEBUG RELATED                       #
############################################################
def print_break_msg():
    print ('')
    print (' ========================= break operator applied here =========================')
    print ('')

def get_memory_usage(filename):
    mem = os.popen("ps aux | grep %s | awk '{sum=sum+$6}; END {print sum/1024 \" MB\"}'"% (filename)).read()
    return mem.rstrip()

def print_exp_name(exp_name, epoch):
    print ('')
    print (' [ERROR] ========================= {} (epoch={}) ========================='.format(exp_name, epoch))
    print ('')

def get_memory(pid):
    try:
        process = psutil.Process(pid)
        return humanize.naturalsize(process.memory_info().rss)
    except:
        return '-1'

def get_tf_gpu_memory():
    # https://www.tensorflow.org/api_docs/python/tf/config/experimental/get_memory_usage
    gpu_devices = tf.config.list_physical_devices('GPU')
    if len(gpu_devices):
        memory_bytes = tf.config.experimental.get_memory_usage(gpu_devices[0].name.split('/physical_device:')[-1])
        return 'GPU: {:2f}GB'.format(memory_bytes/1024.0/1024.0/1024.0)
    else:
        return '-1'

############################################################
#                    DATALOADER RELATED                    #
############################################################
def get_info_from_label_id(label_id, label_map, label_colors=None):
    """
    The label_id param has to be greater than 0
    """
    
    label_name = [label for label in label_map if label_map[label] == label_id]
    if len(label_name):
        label_name = label_name[0]
    else:
        label_name = None

    if label_colors is not None:
        label_color = np.array(label_colors[label_id])
        if np.any(label_color > 0):
            label_color = label_color/255.0
    else:
        label_color = None

    return label_name, label_color

def cmap_for_dataset(label_colors):
    cmap_me = matplotlib.colors.ListedColormap(np.array([*label_colors.values()])/255.0)
    norm = matplotlib.colors.BoundaryNorm(boundaries=range(0,cmap_me.N+1), ncolors=cmap_me.N)

    return cmap_me, norm

############################################################
#                      EVAL RELATED                        #
############################################################
def get_eval_folders(PROJECT_DIR, exp_name, epoch, mode, mc_runs=None, training_bool=None, create=False):
    folder_name                = config.MODEL_CHKPOINT_NAME_FMT.format(epoch)

    if mc_runs is not None and  training_bool is not None: # During manual eval
        if mc_runs == 1 and training_bool == False:
            model_folder_epoch_save    = Path(PROJECT_DIR).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, exp_name, folder_name, config.MODEL_IMGS_FOLDERNAME, mode + config.SUFFIX_DET)
        else:
            model_folder_epoch_save    = Path(PROJECT_DIR).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, exp_name, folder_name, config.MODEL_IMGS_FOLDERNAME, mode + config.SUFFIX_MC.format(mc_runs))
    else: # During automated training + eval
        model_folder_epoch_save    = Path(PROJECT_DIR).joinpath(config.MODEL_CHKPOINT_MAINFOLDER, exp_name, folder_name, config.MODEL_IMGS_FOLDERNAME, mode)
    
    model_folder_epoch_patches = Path(model_folder_epoch_save).joinpath('patches')
    model_folder_epoch_imgs    = Path(model_folder_epoch_save).joinpath('imgs')
    
    if create:
        Path(model_folder_epoch_patches).mkdir(parents=True, exist_ok=True)
        Path(model_folder_epoch_imgs).mkdir(parents=True, exist_ok=True)

    return model_folder_epoch_patches, model_folder_epoch_imgs

############################################################
#                             3D                           #
############################################################

def viz_model_output_3d_old(X, y_true, y_predict, y_predict_std, patient_id, path_save, label_map, label_colors, VMAX_STD=0.3):
    """
    X     : [H,W,D,1]
    y_true: [H,W,D,Labels]

    Takes only a single batch of data
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import skimage
        import skimage.measure
        
        Path(path_save).mkdir(exist_ok=True, parents=True)

        labels_ids = sorted(list(label_map.values()))
        cmap_me, norm_me = cmap_for_dataset(label_colors)
        cmap_img = 'rainbow'
        VMIN = 0
        VMAX_STD  = VMAX_STD
        VMAX_MEAN = 1.0

        slice_ids = list(range(X.shape[2]))
        # with tqdm.tqdm(total=len(slice_ids), leave=False, desc='ID:' + patient_id) as pbar_slices:
        with tqdm.tqdm(total=len(slice_ids), leave=False) as pbar_slices:
            for slice_id in range(X.shape[2]):
                
                # Data
                X_slice = X[:,:,slice_id,0]
                y_true_slice = y_true[:,:,slice_id,:]
                y_predict_slice = y_predict[:,:,slice_id,:]
                y_predict_std_slice = y_predict_std[:,:,slice_id,:]
                
                # Matplotlib figure
                filename = '\n'.join(patient_id.split('-'))
                suptitle_str = 'Slice: {}'.format(filename)
                fig     = plt.figure(figsize=(15,15), dpi=200)
                fig_std = plt.figure(figsize=(15,15), dpi=200)
                spec     = fig.add_gridspec(nrows=2 + np.ceil(len(labels_ids)/5).astype(int), ncols=5)
                spec_std = fig_std.add_gridspec(nrows=2 + np.ceil(len(labels_ids)/5).astype(int), ncols=5)
                fig.suptitle(suptitle_str + '\n Predictive Mean')
                fig_std.suptitle(suptitle_str + '\n Predictive Std')

                # Top two images
                ax3 = fig.add_subplot(spec[0, 4])
                ax3.imshow(X_slice, cmap='gray')
                ax3.axis('off')
                ax3.set_title('Raw data')
                ax3_std = fig_std.add_subplot(spec_std[0, 4])
                ax3_std.imshow(X_slice, cmap='gray')
                ax3_std.axis('off')
                ax3_std.set_title('Raw data')
                img_slice_mask_plot = np.zeros(y_true_slice[:,:,0].shape)
                
                # Other images
                i,j = 2,0
                for label_id in labels_ids:
                    if label_id not in config.IGNORE_LABELS:

                        # Get ground-truth and prediction slices
                        img_slice_mask_predict = copy.deepcopy(y_predict_slice[:,:,label_id])
                        img_slice_mask_predict_std = copy.deepcopy(y_predict_std_slice[:,:,label_id])
                        img_slice_mask_gt = copy.deepcopy(y_true_slice[:,:,label_id])
                        
                        # Plot prediction heatmap
                        if j >= 5:
                            i = i + 1
                            j = 0
                        ax = fig.add_subplot(spec[i,j])
                        ax_std = fig_std.add_subplot(spec_std[i,j])
                        j += 1
                        
                        # img_slice_mask_predict[img_slice_mask_predict > config.PREDICT_THRESHOLD_MASK] = label_id
                        ax.imshow(img_slice_mask_predict, interpolation='none', cmap=cmap_img, vmin=0, vmax=VMAX_MEAN)
                        ax_std.imshow(img_slice_mask_predict_std, interpolation='none', cmap=cmap_img, vmin=0, vmax=VMAX_STD)
                        
                        # Plot Gt contours
                        label, color = get_info_from_label_id(label_id, label_map, label_colors)
                        if label_id == 0: 
                            ax4 = fig.add_subplot(spec[1, 4])
                            fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap_img), ax=ax4)
                            ax4.axis('off')
                            ax4.set_title('Colorbar')

                            ax4_std = fig_std.add_subplot(spec_std[1, 4])
                            fig_std.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap_img, norm=matplotlib.colors.Normalize(vmin=0, vmax=VMAX_STD)), ax=ax4_std)
                            ax4_std.axis('off')
                            ax4_std.set_title('Colorbar')

                        else:
                            contours_mask = skimage.measure.find_contours(img_slice_mask_gt, level=0.99)
                            for _, contour_mask in enumerate(contours_mask):
                                ax3.plot(contour_mask[:, 1], contour_mask[:, 0], linewidth=2, color=color)
                                ax3_std.plot(contour_mask[:, 1], contour_mask[:, 0], linewidth=2, color=color)
                        
                        if label is not None:
                            ax.set_title(label + '(' + str(label_id) + ')', color=color)
                            ax_std.set_title(label + '(' + str(label_id) + ')', color=color)
                        ax.axis('off')
                        ax_std.axis('off')

                        # Gather GT mask
                        idxs_gt = np.argwhere(img_slice_mask_gt > 0)
                        img_slice_mask_plot[idxs_gt[:,0], idxs_gt[:,1]] = label_id

                # GT mask
                ax1 = fig.add_subplot(spec[0:2, 0:2])
                ax1.imshow(img_slice_mask_plot, cmap=cmap_me, norm=norm_me, interpolation='none')
                ax1.set_title('GT Mask')
                ax1.tick_params(labelsize=6)
                ax1_std = fig_std.add_subplot(spec_std[0:2, 0:2])
                ax1_std.imshow(img_slice_mask_plot, cmap=cmap_me, norm=norm_me, interpolation='none')
                ax1_std.set_title('GT Mask')
                ax1_std.tick_params(labelsize=6)

                # Predicted Mask 
                ax2 = fig.add_subplot(spec[0:2, 2:4])
                ax2.imshow(np.argmax(y_predict_slice, axis=2), cmap=cmap_me, norm=norm_me, interpolation='none')
                ax2.set_title('Predicted Mask (mean)')
                ax2.tick_params(labelsize=6)
                ax2_std = fig_std.add_subplot(spec_std[0:2, 2:4])
                ax2_std.imshow(np.argmax(y_predict_slice, axis=2), cmap=cmap_me, norm=norm_me, interpolation='none')
                ax2_std.set_title('Predicted Mask (mean)')
                ax2_std.tick_params(labelsize=6)

                # Show and save
                # path_savefig = Path(model_folder_epoch_images).joinpath(filename_meta.replace('.npy','.png'))
                path_savefig = Path(path_save).joinpath(patient_id + '_' + '%.3d' % (slice_id) + '_mean.png')
                fig.savefig(str(path_savefig), bbox_inches='tight')
                path_savefig_std = Path(path_save).joinpath(patient_id + '_' + '%.3d' % (slice_id) + '_std.png')
                fig_std.savefig(str(path_savefig_std), bbox_inches='tight')
                plt.close(fig=fig)
                plt.close(fig=fig_std)
                pbar_slices.update(1)

    except:
        traceback.print_exc()
        pdb.set_trace()

def viz_model_output_3d(exp_name, X, y_true, y_predict, y_predict_unc, patient_id, path_save, label_map, label_colors, vmax_unc=0.3, unc_title='', unc_savesufix=''):
    """
    X     : [H,W,D,1]
    y_true: [H,W,D,Labels]

    Takes only a single batch of data
    """
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        import skimage
        import skimage.measure
        
        Path(path_save).mkdir(exist_ok=True, parents=True)

        labels_ids = sorted(list(label_map.values()))
        cmap_me, norm_me = cmap_for_dataset(label_colors)
        cmap_img = 'rainbow'
        VMIN = 0
        VMAX_UNC  = vmax_unc
        VMAX_MEAN = 1.0

        slice_ids = list(range(X.shape[2]))
        # with tqdm.tqdm(total=len(slice_ids), leave=False, desc='ID:' + patient_id) as pbar_slices:
        with tqdm.tqdm(total=len(slice_ids), leave=False) as pbar_slices:
            for slice_id in range(X.shape[2]):

                if slice_id < 20:
                    pbar_slices.update(1)
                    continue
                
                # Data
                X_slice             = X[:,:,slice_id,0]
                y_true_slice        = y_true[:,:,slice_id,:]
                y_predict_slice     = y_predict[:,:,slice_id,:]
                y_predict_unc_slice = y_predict_unc[:,:,slice_id,:]
                
                # Matplotlib figure
                filename     = '\n'.join(patient_id.split('-'))
                suptitle_str = 'Exp: {}\nPatient: {}\nSlice: {}'.format(exp_name, filename, slice_id)
                fig          = plt.figure(figsize=(15,15), dpi=200)
                fig_unc      = plt.figure(figsize=(15,15), dpi=200)
                spec         = fig.add_gridspec(nrows=2 + np.ceil(len(labels_ids)/5).astype(int), ncols=5)
                spec_unc     = fig_unc.add_gridspec(nrows=2 + np.ceil(len(labels_ids)/5).astype(int), ncols=5)
                fig.suptitle(suptitle_str + '\n Predictive Mean')
                fig_unc.suptitle(suptitle_str + '\n {}'.format(unc_title))

                # Top two images
                ax3 = fig.add_subplot(spec[0, 4])
                ax3.imshow(X_slice, cmap='gray')
                ax3.axis('off')
                ax3.set_title('Raw data')
                ax3_unc = fig_unc.add_subplot(spec_unc[0, 4])
                ax3_unc.imshow(X_slice, cmap='gray')
                ax3_unc.axis('off')
                ax3_unc.set_title('Raw data')
                img_slice_mask_plot = np.zeros(y_true_slice[:,:,0].shape)
                
                # Other images
                i,j = 2,0
                for label_id in labels_ids:
                    if label_id not in config.IGNORE_LABELS:

                        # Get ground-truth and prediction slices
                        img_slice_mask_predict     = copy.deepcopy(y_predict_slice[:,:,label_id])
                        img_slice_mask_predict_unc = copy.deepcopy(y_predict_unc_slice[:,:,label_id])
                        img_slice_mask_gt          = copy.deepcopy(y_true_slice[:,:,label_id])
                        
                        # Plot prediction heatmap
                        if j >= 5:
                            i = i + 1
                            j = 0
                        ax = fig.add_subplot(spec[i,j])
                        # ax_unc = fig_unc.add_subplot(spec_unc[i,j])
                        j += 1
                        
                        # img_slice_mask_predict[img_slice_mask_predict > config.PREDICT_THRESHOLD_MASK] = label_id
                        ax.imshow(img_slice_mask_predict, interpolation='none', cmap=cmap_img, vmin=0, vmax=VMAX_MEAN)
                        # ax_unc.imshow(img_slice_mask_predict_unc, interpolation='none', cmap=cmap_img, vmin=0, vmax=VMAX_UNC)
                        
                        # Plot Gt contours
                        label, color = get_info_from_label_id(label_id, label_map, label_colors)
                        if label_id == 0: 
                            ax4 = fig.add_subplot(spec[1, 4])
                            fig.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap_img), ax=ax4)
                            ax4.axis('off')
                            ax4.set_title('Colorbar')

                            ax4_unc = fig_unc.add_subplot(spec_unc[1, 4])
                            fig_unc.colorbar(matplotlib.cm.ScalarMappable(cmap=cmap_img, norm=matplotlib.colors.Normalize(vmin=0, vmax=VMAX_UNC)), ax=ax4_unc)
                            ax4_unc.axis('off')
                            ax4_unc.set_title('Colorbar')

                        else:
                            contours_mask = skimage.measure.find_contours(img_slice_mask_gt, level=0.99)
                            for _, contour_mask in enumerate(contours_mask):
                                ax3.plot(contour_mask[:, 1], contour_mask[:, 0], linewidth=2, color=color)
                                ax3_unc.plot(contour_mask[:, 1], contour_mask[:, 0], linewidth=2, color=color)
                        
                        if label is not None:
                            ax.set_title(label + '(' + str(label_id) + ')', color=color)
                            # ax_unc.set_title(label + '(' + str(label_id) + ')', color=color)
                        ax.axis('off')
                        # ax_unc.axis('off')

                        # Gather GT mask
                        idxs_gt = np.argwhere(img_slice_mask_gt > 0)
                        img_slice_mask_plot[idxs_gt[:,0], idxs_gt[:,1]] = label_id

                        # if 'ent' in unc_savesufix:
                        #     ax_unc.cla()

                # GT mask
                ax1 = fig.add_subplot(spec[0:2, 0:2])
                ax1.imshow(img_slice_mask_plot, cmap=cmap_me, norm=norm_me, interpolation='none')
                ax1.set_title('GT Mask')
                ax1.tick_params(labelsize=6)
                ax1_unc = fig_unc.add_subplot(spec_unc[0:2, 0:2])
                ax1_unc.imshow(img_slice_mask_plot, cmap=cmap_me, norm=norm_me, interpolation='none')
                ax1_unc.set_title('GT Mask')
                ax1_unc.tick_params(labelsize=6)

                # Predicted Mask 
                ax2 = fig.add_subplot(spec[0:2, 2:4])
                ax2.imshow(np.argmax(y_predict_slice, axis=2), cmap=cmap_me, norm=norm_me, interpolation='none')
                ax2.set_title('Predicted Mask (mean)')
                ax2.tick_params(labelsize=6)
                ax2_unc = fig_unc.add_subplot(spec_unc[0:2, 2:4])
                ax2_unc.imshow(np.argmax(y_predict_slice, axis=2), cmap=cmap_me, norm=norm_me, interpolation='none')
                ax2_unc.set_title('Predicted Mask (mean)')
                ax2_unc.tick_params(labelsize=6)

                # Specifically for entropy
                if unc_savesufix in ['stdmax', 'ent', 'mif']:

                    ax_unc_gt = fig_unc.add_subplot(spec[2:4, 0:2])
                    ax_unc_gt.imshow(img_slice_mask_predict_unc, interpolation='none', cmap=cmap_img, vmin=0, vmax=VMAX_UNC)
                    slice_binary_gt  = img_slice_mask_plot
                    slice_binary_gt[slice_binary_gt > 0] = 1
                    ax_unc_gt.imshow(slice_binary_gt, cmap='gray', interpolation='none', alpha=0.3)

                    ax_unc_pred = fig_unc.add_subplot(spec[2:4, 2:4])
                    ax_unc_pred.imshow(img_slice_mask_predict_unc, interpolation='none', cmap=cmap_img, vmin=0, vmax=VMAX_UNC)
                    slice_binary_pred     = np.argmax(y_predict_slice, axis=2)
                    slice_binary_pred[slice_binary_pred > 0] = 1
                    ax_unc_pred.imshow(slice_binary_pred, cmap='gray', interpolation='none', alpha=0.3)

                # Show and save
                # path_savefig = Path(model_folder_epoch_images).joinpath(filename_meta.replace('.npy','.png'))
                path_savefig = Path(path_save).joinpath(patient_id + '_' + '%.3d' % (slice_id) + '_mean.png')
                fig.savefig(str(path_savefig), bbox_inches='tight')
                path_savefig_unc = Path(path_save).joinpath(patient_id + '_' + '%.3d' % (slice_id) + '_{}.png'.format(unc_savesufix))
                fig_unc.savefig(str(path_savefig_unc), bbox_inches='tight')
                plt.close(fig=fig)
                plt.close(fig=fig_unc)
                pbar_slices.update(1)

    except:
        traceback.print_exc()
        pdb.set_trace()
    