# Import internal libraries
import src.config as config

# Import external libraries
import pdb
import copy
import time
import skimage
import skimage.util
import traceback
import numpy as np
from pathlib import Path
import SimpleITK as sitk 
import tensorflow as tf
import tensorflow_addons as tfa
try: import tensorflow_probability as tfp
except: pass
import matplotlib.pyplot as plt

_EPSILON = tf.keras.backend.epsilon()
MAX_FUNC_TIME = 300 

############################################################
#                           UTILS                          #
############################################################

@tf.function
def get_mask(mask_1D, Y):
    # mask_1D: [[1,0,0,0, ...., 1]] - [B,L] something like this
    # Y : [B,H,W,D,L] 
    if 0:
        mask = tf.expand_dims(tf.expand_dims(tf.expand_dims(mask_1D, axis=1),axis=1),axis=1) # mask.shape=[B,1,1,1,L]
        mask = tf.tile(mask, multiples=[1,Y.shape[1],Y.shape[2],Y.shape[3],1]) # mask.shape = [B,H,W,D,L]
        mask = tf.cast(mask, tf.float32)
        return mask
    else:
        return mask_1D

def get_largest_component(y, verbose=True):
    """
    Takes as input predicted predicted probs and returns a binary mask with the largest component for each label

    Params
    ------
    y: [H,W,D,L] --> predicted probabilities of np.float32 
    - Ref: https://simpleitk.readthedocs.io/en/v1.2.4/Documentation/docs/source/filters.html
    """

    try:
        
        label_count = y.shape[-1]
        y = np.argmax(y, axis=-1)
        y = np.concatenate([np.expand_dims(y == label_id, axis=-1) for label_id in range(label_count)],axis=-1)
        ccFilter = sitk.ConnectedComponentImageFilter()
        
        for label_id in range(y.shape[-1]):

            if label_id > 0 and label_id not in []: # Note: pointless to do it for background
                if verbose: t0 = time.time()
                y_label = y[:,:,:,label_id] # [H,W,D]

                component_img = ccFilter.Execute(sitk.GetImageFromArray(y_label.astype(np.uint8)))
                component_array = sitk.GetArrayFromImage(component_img) # will contain pseduo-labels given to different components
                component_count = ccFilter.GetObjectCount()
                
                if component_count >= 2: # background and foreground
                    component_sizes = np.bincount(component_array.flatten()) # count the voxels belong to different components
                    component_sizes_sorted = np.asarray(sorted(component_sizes, reverse=True))
                    if verbose: print ('\n - [INFO][losses.get_largest_component()] label_id: ', label_id, ' || sizes: ', component_sizes_sorted)
                
                    component_largest_sortedidx = np.argwhere(component_sizes == component_sizes_sorted[1])[0][0] # Note: idx=1 as idx=0 is background # Risk: for optic nerves
                    y_label_mask = (component_array == component_largest_sortedidx).astype(np.float32)
                    y[:,:,:,label_id] = y_label_mask
                    if verbose: print (' - [INFO][losses.get_largest_component()]: label_id: ', label_id, '(',round(time.time() - t0,3),'s)')
                else:
                    if verbose: print (' - [INFO][losses.get_largest_component()] label_id: ', label_id, ' has only background!!')

                # [TODO]: set other components as background (i.e. label=0)

        y = y.astype(np.float32)
        return y

    except:
        traceback.print_exc()
        pdb.set_trace()

def remove_smaller_components(y_true, y_pred, meta='', label_ids_small = [], verbose=False):
    """
    Takes as input predicted probs and returns a binary mask by removing some of the smallest components for each label

    Params
    ------
    y: [H,W,D,L] --> predicted probabilities of np.float32
    - Ref: https://simpleitk.readthedocs.io/en/v1.2.4/Documentation/docs/source/filters.html
    """
    t0 = time.time()

    try:
        
        # Step 0 - Preprocess by selecting one voxel per class
        y_pred_copy = copy.deepcopy(y_pred) # [H,W,D,L] with probs
        label_count = y_pred_copy.shape[-1]
        y_pred_copy = np.argmax(y_pred_copy, axis=-1) # [H,W,D]
        y_pred_copy = np.concatenate([np.expand_dims(y_pred_copy == label_id, axis=-1) for label_id in range(label_count)],axis=-1) # [H,W,D,L] as a binary mask
        
        for label_id in range(y_pred_copy.shape[-1]):

            if label_id > 0: # Note: pointless to do it for background
                if verbose: t0 = time.time()
                y_label = y_pred_copy[:,:,:,label_id] # [H,W,D]

                # Step 1 - Get different components
                ccFilter = sitk.ConnectedComponentImageFilter()
                component_img = ccFilter.Execute(sitk.GetImageFromArray(y_label.astype(np.uint8)))
                component_array = sitk.GetArrayFromImage(component_img) # will contain pseduo-labels given to different components
                component_count = ccFilter.GetObjectCount()
                component_sizes = np.bincount(component_array.flatten()) # count the voxels belong to different components    

                # Step 2 - Evaluate each component
                if component_count >= 1: # at least a foreground (along with background)

                    # Step 2.1 - Sort them on the basis of voxel count
                    component_sizes_sorted = np.asarray(sorted(component_sizes, reverse=True))
                    if verbose:
                        print ('\n - [INFO][losses.get_largest_component()] label_id: ', label_id, ' || sizes: ', component_sizes_sorted)
                        print (' - [INFO][losses.get_largest_component()] unique_comp_labels: ', np.unique(component_array)) 
                        
                    # Step 2.1 - Remove really small components for good Hausdorff calculation
                    component_sizes_sorted_unique = np.unique(component_sizes_sorted[::-1]) # ascending order
                    for comp_size_id, comp_size in enumerate(component_sizes_sorted_unique):
                        if label_id in label_ids_small:
                            if comp_size <= config.MIN_SIZE_COMPONENT:
                                components_labels = [each[0] for each in np.argwhere(component_sizes == comp_size)]
                                for component_label in components_labels:
                                    component_array[component_array == component_label] = 0
                        else:
                            if comp_size_id < len(component_sizes_sorted_unique) - 2: # set to 0 except background and foreground
                                components_labels = [each[0] for each in np.argwhere(component_sizes == comp_size)]
                                for component_label in components_labels:
                                    component_array[component_array == component_label] = 0
                    if verbose: print (' - [INFO][losses.get_largest_component()] unique_comp_labels: ', np.unique(component_array))
                    y_pred_copy[:,:,:,label_id] = component_array.astype(np.bool).astype(np.float32)
                    if verbose: print (' - [INFO][losses.get_largest_component()] label_id: ', label_id, '(',round(time.time() - t0,3),'s)')

                    if 0:
                        # Step 1 - Hausdorff
                        y_true_label = y_true[:,:,:,label_id]
                        y_pred_label = y_pred_copy[:,:,:,label_id]

                        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
                        hausdorff_distance_filter.Execute(sitk.GetImageFromArray(y_true_label.astype(np.uint8)), sitk.GetImageFromArray(y_pred_label.astype(np.uint8)))
                        print (' - hausdorff: ', hausdorff_distance_filter.GetHausdorffDistance())
                        
                        # Step 2 - 95% Hausdorff
                        y_true_contour = sitk.LabelContour(sitk.GetImageFromArray(y_true_label.astype(np.uint8)), False)
                        y_pred_contour = sitk.LabelContour(sitk.GetImageFromArray(y_pred_label.astype(np.uint8)), False)
                        y_true_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_true_contour, squaredDistance=False, useImageSpacing=True)) # i.e. euclidean distance
                        y_pred_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_pred_contour, squaredDistance=False, useImageSpacing=True))
                        dist_y_pred = sitk.GetArrayViewFromImage(y_pred_distance_map)[sitk.GetArrayViewFromImage(y_true_distance_map)==0] # pointless?
                        dist_y_true = sitk.GetArrayViewFromImage(y_true_distance_map)[sitk.GetArrayViewFromImage(y_pred_distance_map)==0]
                        print (' - 95 hausdorff:', np.percentile(dist_y_true,95), np.percentile(dist_y_pred,95))

                else:
                    print (' - [INFO][losses.get_largest_component()] for meta: {} || label_id: {} has only background!! ({}) '.format(meta, label_id, component_sizes))
            
            if time.time() - t0 > MAX_FUNC_TIME:
                print (' - [INFO][losses.get_largest_component()] Taking too long: ', round(time.time() - t0,2),'s')

        y = y_pred_copy.astype(np.float32)
        return y

    except:
        traceback.print_exc()
        pdb.set_trace()

def get_hausdorff(y_true, y_pred, spacing, verbose=False):
    """
    :param y_true: [H, W, D, L]
    :param y_pred: [H, W, D, L] 
    - Ref: https://simpleitk.readthedocs.io/en/master/filters.html?highlight=%20HausdorffDistanceImageFilter()#simpleitk-filters
    - Ref: http://insightsoftwareconsortium.github.io/SimpleITK-Notebooks/Python_html/34_Segmentation_Evaluation.html
    """

    try:
        hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()

        hausdorff_labels = []
        for label_id in range(y_pred.shape[-1]):
            
            y_true_label = y_true[:,:,:,label_id] # [H,W,D]
            y_pred_label = y_pred[:,:,:,label_id] # [H,W,D]

            # Calculate loss (over all pixels)
            if np.sum(y_true_label) > 0:
                if label_id > 0:
                    try:
                        if np.sum(y_true_label) > 0:
                            y_true_sitk    = sitk.GetImageFromArray(y_true_label.astype(np.uint8))
                            y_pred_sitk    = sitk.GetImageFromArray(y_pred_label.astype(np.uint8))
                            y_true_sitk.SetSpacing(tuple(spacing))
                            y_pred_sitk.SetSpacing(tuple(spacing))
                            hausdorff_distance_filter.Execute(y_true_sitk, y_pred_sitk)
                            hausdorff_labels.append(hausdorff_distance_filter.GetHausdorffDistance())
                            if verbose: print (' - [INFO][get_hausdorff()] label_id: {} || hausdorff: {}'.format(label_id, hausdorff_labels[-1]))
                        else:
                            hausdorff_labels.append(-1)    
                    except:
                        print (' - [ERROR][get_hausdorff()] label_id: {}'.format(label_id))
                        hausdorff_labels.append(-1)    
                else:
                    hausdorff_labels.append(0)
            else:
                hausdorff_labels.append(0)
        
        hausdorff_labels = np.array(hausdorff_labels)
        hausdorff = np.mean(hausdorff_labels[hausdorff_labels>0])
        return hausdorff, hausdorff_labels
    
    except:
        traceback.print_exc()
        pdb.set_trace()

def get_surface_distances(y_true, y_pred, spacing, verbose=False):
    
    """
    :param y_true: [H, W, D, L] --> binary mask of np.float32
    :param y_pred: [H, W, D, L] --> also, binary mask of np.float32
    - Ref: https://discourse.itk.org/t/computing-95-hausdorff-distance/3832/3
    - Ref: https://git.lumc.nl/mselbiallyelmahdy/jointregistrationsegmentation-via-crossstetch/-/blob/master/lib/label_eval.py
    """

    try:
        hausdorff_labels   = []
        hausdorff95_labels = []
        msd_labels         = []
        for label_id in range(y_pred.shape[-1]):
            
            y_true_label = y_true[:,:,:,label_id] # [H,W,D]
            y_pred_label = y_pred[:,:,:,label_id] # [H,W,D]

            # Calculate loss (over all pixels)
            if np.sum(y_true_label) > 0:

                if label_id > 0:
                    if np.sum(y_pred_label) > 0:
                        y_true_sitk    = sitk.GetImageFromArray(y_true_label.astype(np.uint8))
                        y_pred_sitk    = sitk.GetImageFromArray(y_pred_label.astype(np.uint8))
                        y_true_sitk.SetSpacing(tuple(spacing))
                        y_pred_sitk.SetSpacing(tuple(spacing))
                        y_true_contour = sitk.LabelContour(y_true_sitk, False, backgroundValue=0)
                        y_pred_contour = sitk.LabelContour(y_pred_sitk, False, backgroundValue=0)
                        
                        y_true_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_true_sitk, squaredDistance=False, useImageSpacing=True))
                        y_pred_distance_map = sitk.Abs(sitk.SignedMaurerDistanceMap(y_pred_sitk, squaredDistance=False, useImageSpacing=True))
                        dist_y_true         = sitk.GetArrayFromImage(y_true_distance_map*sitk.Cast(y_pred_contour, sitk.sitkFloat32))
                        dist_y_pred         = sitk.GetArrayFromImage(y_pred_distance_map*sitk.Cast(y_true_contour, sitk.sitkFloat32))
                        dist_y_true         = dist_y_true[dist_y_true != 0]
                        dist_y_pred         = dist_y_pred[dist_y_pred != 0]
                            
                        if len(dist_y_true):
                            msd_labels.append(np.mean(np.array(list(dist_y_true) + list(dist_y_pred))))
                            if len(dist_y_true) and len(dist_y_pred):
                                hausdorff_labels.append( np.max( [np.max(dist_y_true), np.max(dist_y_pred)] ) )
                                hausdorff95_labels.append(np.max([np.percentile(dist_y_true, 95), np.percentile(dist_y_pred, 95)]))
                            elif len(dist_y_true) and not len(dist_y_pred):
                                hausdorff_labels.append(np.max(dist_y_true))
                                hausdorff95_labels.append(np.percentile(dist_y_true, 95))
                            elif not len(dist_y_true) and not len(dist_y_pred):
                                hausdorff_labels.append(np.max(dist_y_pred))
                                hausdorff95_labels.append(np.percentile(dist_y_pred, 95))
                        else:
                            hausdorff_labels.append(-1)
                            hausdorff95_labels.append(-1)
                            msd_labels.append(-1)
                    
                    else:
                        hausdorff_labels.append(-1)
                        hausdorff95_labels.append(-1)
                        msd_labels.append(-1)
                
                else:
                    hausdorff_labels.append(0)
                    hausdorff95_labels.append(0)
                    msd_labels.append(0)
            
            else:
                hausdorff_labels.append(0)
                hausdorff95_labels.append(0)
                msd_labels.append(0)
        
        hausdorff_labels   = np.array(hausdorff_labels)
        hausdorff_mean     = np.mean(hausdorff_labels[hausdorff_labels > 0])
        hausdorff95_labels = np.array(hausdorff95_labels)
        hausdorff95_mean   = np.mean(hausdorff95_labels[hausdorff95_labels > 0])
        msd_labels         = np.array(msd_labels)
        msd_mean           = np.mean(msd_labels[msd_labels > 0])
        return hausdorff_mean, hausdorff_labels, hausdorff95_mean, hausdorff95_labels, msd_mean, msd_labels

    except:
        traceback.print_exc()
        return -1, [], -1, []

def dice_numpy_slice(y_true_slice, y_pred_slice):
    """
    Specifically designed for 2D slices
    
    Params
    ------
    y_true_slice: [H,W]
    y_pred_slice: [H,W]
    """

    sum_true = np.sum(y_true_slice)
    sum_pred = np.sum(y_pred_slice)
    if sum_true > 0 and sum_pred > 0:
        num = 2 * np.sum(y_true_slice *y_pred_slice)
        den = sum_true + sum_pred
        return num/den
    elif sum_true > 0 and sum_pred == 0:
        return 0
    elif sum_true == 0 and sum_pred > 0:
        return -0.1
    else:
        return -1

def dice_numpy(y_true, y_pred):
    """
    :param y_true: [H, W, D, L]
    :param y_pred: [H, W, D, L] 
    - Ref: V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    - Ref: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py
    """

    dice_labels = []
    for label_id in range(y_pred.shape[-1]):
        
        y_true_label = y_true[:,:,:,label_id]        # [H,W,D]
        y_pred_label = y_pred[:,:,:,label_id] + 1e-8 # [H,W,D]

        # Calculate loss (over all pixels)
        if np.sum(y_true_label) > 0:
            num = 2*np.sum(y_true_label * y_pred_label)
            den = np.sum(y_true_label + y_pred_label)
            dice_label = num/den
        else:
            dice_label = -1.0

        dice_labels.append(dice_label)
    
    dice_labels = np.array(dice_labels)
    dice = np.mean(dice_labels[dice_labels>0])
    return dice, dice_labels

############################################################
#                           LOSSES                         #
############################################################

@tf.function
def loss_dice_3d_tf_func(y_true, y_pred, label_mask, weights=[], verbose=False):

    """
    Calculates soft-DICE loss

    :param y_true: [B, H, W, C, L]
    :param y_pred: [B, H, W, C, L] 
    :param label_mask: [B,L] 
    - Ref: V-Net: Fully Convolutional Neural Networks for Volumetric Medical Image Segmentation
    - Ref: https://gist.github.com/jeremyjordan/9ea3032a32909f71dd2ab35fe3bacc08#file-soft_dice_loss-py
    """
    
    # Step 0 - Init
    dice_labels = []
    label_mask = tf.cast(label_mask, dtype=tf.float32) # [B,L]

    # Step 1 - Get DICE of each sample in each label
    y_pred      = y_pred + _EPSILON
    dice_labels = (2*tf.math.reduce_sum(y_true * y_pred, axis=[1,2,3]))/(tf.math.reduce_sum(y_true + y_pred, axis=[1,2,3])) # [B,H,W,D,L] -> [B,L]
    dice_labels = dice_labels*label_mask # if mask of a label (e.g. background) has been explicitly set to 0, do not consider its loss
        
    # Step 2 - Mask results on the basis of ground truth availability
    label_mask             = tf.where(tf.math.greater(label_mask,0), label_mask, _EPSILON) # to handle division by 0
    dice_for_train         = None
    dice_labels_for_train  = None
    dice_labels_for_report = tf.math.reduce_sum(dice_labels,axis=0) / tf.math.reduce_sum(label_mask, axis=0)
    dice_for_report        = tf.math.reduce_mean(tf.math.reduce_sum(dice_labels,axis=1) / tf.math.reduce_sum(label_mask, axis=1))
    
    # Step 3 - Weighted DICE
    if len(weights):
        label_weights = weights / tf.math.reduce_sum(weights) # nomalized
        dice_labels_w = dice_labels * label_weights
        dice_labels_for_train = tf.math.reduce_sum(dice_labels_w,axis=0) / tf.math.reduce_sum(label_mask, axis=0) 
        dice_for_train = tf.math.reduce_mean(tf.math.reduce_sum(dice_labels_w,axis=1) / tf.math.reduce_sum(label_mask, axis=1))
    else:
        dice_labels_for_train = dice_labels_for_report
        dice_for_train = dice_for_report

    # Step 4 - Return results
    return 1.0 - dice_for_train, 1.0 - dice_labels_for_train, dice_for_report, dice_labels_for_report

@tf.function
def loss_ce_3d_tf_func(y_true, y_pred, label_mask, weights=[], verbose=False):
    """
    Calculates cross entropy loss

    Params
    ------
    y_true    : [B, H, W, C, L]
    y_pred    : [B, H, W, C, L] 
    label_mask: [B,L]
    - Ref: https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
    """
    
    # Step 0 - Init
    loss_labels = []
    label_mask = tf.cast(label_mask, dtype=tf.float32)
    y_pred     = y_pred + _EPSILON

    # Step 1.1 - Foreground loss
    loss_labels = -1.0 * y_true * tf.math.log(y_pred) # [B,H,W,D,L]
    loss_labels = label_mask * tf.math.reduce_sum(loss_labels, axis=[1,2,3]) # [B,H,W,D,L] --> [B,L]

    # Step 1.2 - Background loss
    y_pred_neg        = 1 - y_pred + _EPSILON
    loss_labels_neg   = -1.0 * (1 - y_true) * tf.math.log(y_pred_neg) # [B,H,W,D,L]
    loss_labels_neg   = label_mask * tf.math.reduce_sum(loss_labels_neg, axis=[1,2,3]) 
    loss_labels       = loss_labels + loss_labels_neg
    
    # Step 2 - Mask results on the basis of ground truth availability
    label_mask = tf.where(tf.math.greater(label_mask,0), label_mask, _EPSILON) # for reasons of division
    loss_for_train = None
    loss_labels_for_train = None
    loss_labels_for_report = tf.math.reduce_sum(loss_labels,axis=0) / tf.math.reduce_sum(label_mask, axis=0)
    loss_for_report = tf.math.reduce_mean(tf.math.reduce_sum(loss_labels,axis=1) / tf.math.reduce_sum(label_mask, axis=1))

    # Step 3 - Weighted DICE
    if len(weights):
        label_weights = weights / tf.math.reduce_sum(weights) # nomalized    
        loss_labels_w = loss_labels * label_weights
        loss_labels_for_train = tf.math.reduce_sum(loss_labels_w,axis=0) / tf.math.reduce_sum(label_mask, axis=0) 
        loss_for_train = tf.math.reduce_mean(tf.math.reduce_sum(loss_labels_w,axis=1) / tf.math.reduce_sum(label_mask, axis=1))
    else:
        loss_labels_for_train = loss_labels_for_report
        loss_for_train = loss_for_report
    
    # Step 4 - Return results
    return loss_for_train, loss_labels_for_train, loss_for_report, loss_labels_for_report

@tf.function
def loss_cebasic_3d_tf_func(y_true, y_pred, label_mask, weights=[], verbose=False):
    """
    Calculates cross entropy loss

    Params
    ------
    y_true    : [B, H, W, C, L]
    y_pred    : [B, H, W, C, L] 
    label_mask: [B,L]
    - Ref: https://www.dlology.com/blog/multi-class-classification-with-focal-loss-for-imbalanced-datasets/
    """
    
    # Step 0 - Init
    loss_labels = []
    label_mask = tf.cast(label_mask, dtype=tf.float32)
    y_pred     = y_pred + _EPSILON

    # Step 1 - Foreground loss
    loss_labels = -1.0 * y_true * tf.math.log(y_pred) # [B,H,W,D,L]
    loss_labels = label_mask * tf.math.reduce_sum(loss_labels, axis=[1,2,3]) # [B,H,W,D,L] --> [B,L]
    
    # Step 2 - Mask results on the basis of ground truth availability
    label_mask = tf.where(tf.math.greater(label_mask,0), label_mask, _EPSILON) # for reasons of division
    loss_for_train = None
    loss_labels_for_train = None
    loss_labels_for_report = tf.math.reduce_sum(loss_labels,axis=0) / tf.math.reduce_sum(label_mask, axis=0)
    loss_for_report = tf.math.reduce_mean(tf.math.reduce_sum(loss_labels,axis=1) / tf.math.reduce_sum(label_mask, axis=1))

    # Step 3 - Weighted DICE
    if len(weights):
        label_weights = weights / tf.math.reduce_sum(weights) # nomalized    
        loss_labels_w = loss_labels * label_weights
        loss_labels_for_train = tf.math.reduce_sum(loss_labels_w,axis=0) / tf.math.reduce_sum(label_mask, axis=0) 
        loss_for_train = tf.math.reduce_mean(tf.math.reduce_sum(loss_labels_w,axis=1) / tf.math.reduce_sum(label_mask, axis=1))
    else:
        loss_labels_for_train = loss_labels_for_report
        loss_for_train = loss_for_report
    
    # Step 4 - Return results
    return loss_for_train, loss_labels_for_train, loss_for_report, loss_labels_for_report