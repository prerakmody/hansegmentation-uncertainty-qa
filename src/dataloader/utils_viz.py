import pdb
import copy
import traceback
import numpy as np
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt

import medloader.dataloader.utils as utils
import medloader.dataloader.config as config

def cmap_for_dataset(dataset):
    # dataset = ZipDataset
    LABEL_COLORS = dataset.get_label_colors()
    cmap_me = matplotlib.colors.ListedColormap(np.array([*LABEL_COLORS.values()])/255.0)
    norm = matplotlib.colors.BoundaryNorm(boundaries=range(0,cmap_me.N+1), ncolors=cmap_me.N)

    return cmap_me, norm

############################################################
#                             2D                           #
############################################################

def get_info_from_label_id(label_id, LABEL_MAP, LABEL_COLORS=None):
    """
    The label_id param has to be greater than 0
    """

    label_name = [label for label in LABEL_MAP if LABEL_MAP[label] == label_id]
    if len(label_name):
        label_name = label_name[0]
    else:
        label_name = None

    label_color = None
    if LABEL_COLORS is not None:
        label_color = np.array(LABEL_COLORS[label_id])
        if np.any(label_color > 1):
            label_color = label_color/255.0

    return label_name, label_color

def viz_slice_raw_batch(slices_raw, slices_mask, meta, dataset):
    try:

        slices_raw = slices_raw[:,:,:,0]
        batch_size = slices_raw.shape[0]

        LABEL_COLORS = getattr(config, dataset.name)['LABEL_COLORS']
        cmap_me = matplotlib.colors.ListedColormap(np.array([*LABEL_COLORS.values()])/255.0)
        norm = matplotlib.colors.BoundaryNorm(boundaries=range(0,cmap_me.N+1), ncolors=cmap_me.N)
        # for i in range(cmap_me.N): print (i, ':', norm.__call__(i))
        
        f, axarr = plt.subplots(2,batch_size, figsize=config.FIGSIZE)
        if batch_size == 1:
            axarr = axarr.reshape(2,batch_size)
        
        # Loop over all batches
        for batch_id in range(batch_size):
            slice_raw= slices_raw[batch_id].numpy()
            idxs_update = np.argwhere(slice_raw < -200)
            slice_raw[idxs_update[:,0], idxs_update[:,1]] = -200
            idxs_update = np.argwhere(slice_raw > 400)
            slice_raw[idxs_update[:,0], idxs_update[:,1]] = 400
            axarr[1,batch_id].imshow(slice_raw, cmap='gray')
            axarr[0,batch_id].imshow(slice_raw, cmap='gray') 

            # Create a mask
            label_mask_show = np.zeros(slices_mask[batch_id,:,:,0].shape)
            for label_id in range(slices_mask.shape[-1]):
                label_mask = slices_mask[batch_id,:,:,label_id].numpy()
                if np.sum(label_mask) > 0:
                    label_idxs = np.argwhere(label_mask > 0)
                    if np.any(label_mask_show[label_idxs[:,0], label_idxs[:,1]] > 0):
                        print (' - [utils.viz_slice_raw_batch()] Label overload by label_id:{}'.format(label_id))

                    label_mask_show[label_idxs[:,0], label_idxs[:,1]] = label_id

                    if len(label_idxs) < 20:
                        label_name, _ = get_info_from_label_id(label_id, dataset)
                        print (' - [utils.viz_slice_raw_batch()] label_id:{} || label: {} || count: {}'.format(label_id+1, label_name, len(label_idxs)))
                    

            # Show the mask
            axarr[0,batch_id].imshow(label_mask_show.astype(np.float32), cmap=cmap_me, norm=norm, interpolation='none', alpha=0.3)

            # Add legend
            for label_id_mask in np.unique(label_mask_show):
                if label_id_mask != 0:
                    label, color = get_info_from_label_id(label_id_mask, dataset)
                    axarr[0,batch_id].scatter(0,0, color=color, label=label + '(' + str(int(label_id_mask))+')')
                leg = axarr[0,batch_id].legend(fontsize=8)
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(5.0)       
            
            # Add title
            if meta is not None and dataset is not None:
                filename = '\n'.join(meta[batch_id].numpy().decode('utf-8').split('-'))
                axarr[0,batch_id].set_title(dataset.name + '\n' + filename + '\n(BatchId={})'.format(batch_id))

        mng = plt.get_current_fig_manager()
        try:mng.window.showMaximized()
        except:pass
        try:mng.window.maxsize()
        except:pass
        plt.show()

        
    except:
        traceback.print_exc()
        pdb.set_trace()

def viz_slice_raw_batch_datasets(slices_raw, slices_mask, meta1, meta2, datasets):
    try:

        slices_raw = slices_raw[:,:,:,0]
        batch_size = slices_raw.shape[0]
        
        f, axarr = plt.subplots(2,batch_size, figsize=config.FIGSIZE)
        if batch_size == 1:
            axarr = axarr.reshape(2,batch_size)
        
        # Loop over all batches
        for batch_id in range(batch_size):
            axarr[1,batch_id].imshow(slices_raw[batch_id], cmap='gray')
            axarr[0,batch_id].imshow(slices_raw[batch_id], cmap='gray') 

            meta2_batchid = meta2[batch_id].numpy().decode('utf-8').split('-')[0]
            dataset_batch = ''
            for dataset in datasets:
                if dataset.name == meta2_batchid:
                    dataset_batch = dataset

            LABELID_BACKGROUND = getattr(config, dataset_batch.name)['LABELID_BACKGROUND']
            LABEL_COLORS = getattr(config, dataset_batch.name)['LABEL_COLORS']
            cmap_me = matplotlib.colors.ListedColormap(np.array([*LABEL_COLORS.values()])/255.0)
            norm = matplotlib.colors.BoundaryNorm(boundaries=range(1,12), ncolors=cmap_me.N)

            # Create a mask
            label_mask_show = np.zeros(slices_mask[batch_id,:,:,0].shape)
            for label_id in range(slices_mask.shape[-1]):
                label_mask = slices_mask[batch_id,:,:,label_id].numpy()
                if np.sum(label_mask) > 0:
                    label_idxs = np.argwhere(label_mask > 0)
                    if label_id == slices_mask.shape[-1] - 1: 
                        label_id_actual = LABELID_BACKGROUND
                    else:      
                        label_id_actual = label_id + 1

                    if np.any(label_mask_show[label_idxs[:,0], label_idxs[:,1]] > 0):
                        print (' - [utils.viz_slice_raw_batch()] Label overload by label_id:{}'.format(label_id))
                    if len(label_idxs) < 20:
                        label_name, _ = get_info_from_label_id(label_id_actual, dataset_batch)
                        print (' - [utils.viz_slice_raw_batch()] label_id:{} || label: {} || count: {}'.format(label_id+1, label_name, len(label_idxs)))
                    label_mask_show[label_idxs[:,0], label_idxs[:,1]] = label_id_actual
                    # label_name, label_color = get_info_from_label_id(label_id+1)

            # Show the mask
            # axarr[0,batch_id].imshow(label_mask_show, alpha=0.5, cmap=cmap_me, norm=norm)
            axarr[0,batch_id].imshow(label_mask_show.astype(np.float32), cmap=cmap_me, norm=norm)

            # Add legend
            for label_id_mask in np.unique(label_mask_show):
                if label_id_mask != 0:
                    label, color = get_info_from_label_id(label_id_mask, dataset_batch)
                    axarr[0,batch_id].scatter(0,0, color=color, label=label + '(' + str(int(label_id_mask))+')')
                leg = axarr[0,batch_id].legend(fontsize=8)
                for legobj in leg.legendHandles:
                    legobj.set_linewidth(5.0)       
            
            # Add title
            filepath = dataset_batch.paths_raw[meta1[batch_id]]
            if len(filepath) == 2:
                filepath = filepath[0] + ' ' + filepath[1]
            filename = Path(filepath).parts[-1]
            axarr[0,batch_id].set_title(dataset_batch.name + '\n' + filename + '\n(BatchId={})'.format(batch_id))

        mng = plt.get_current_fig_manager()
        mng.window.showMaximized()
        plt.show()

        
    except:
        traceback.print_exc()
        pdb.set_trace()

def viz_slice(slice_raw, slice_mask, meta=None, dataset=None):
    """
    - slice_raw: [H,W]
    - slice_mask: [H,W]
    """
    try:
        import matplotlib.pyplot as plt
        f, axarr = plt.subplots(2,2, figsize=config.FIGSIZE)
        axarr[0,0].imshow(slice_raw, cmap='gray') 
        axarr[0,1].imshow(slice_mask)
        axarr[1,0].hist(slice_raw)
        axarr[1,1].imshow(slice_raw, cmap='gray')
        axarr[1,1].imshow(slice_mask, alpha=0.5)

        axarr[0,0].set_title('Raw image')
        axarr[0,1].set_title('Raw mask')
        axarr[1,0].set_title('Raw Value Histogram')
        axarr[1,1].set_title('Raw + Mask')

        if meta is not None and dataset is not None:
            filename = Path(dataset.paths_raw[meta[0]]).parts[-1]
            plt.suptitle(filename)

        try:
            mng = plt.get_current_fig_manager()
            mng.window.showMaximized()
        except: pass
        
        plt.show()

    except:
        traceback.print_exc()
        pdb.set_trace()

############################################################
#                             3D                           #
############################################################

def viz_3d_slices(voxel_img, voxel_mask, dataset, meta1, meta2, plots=4):
    """
    voxel_img : [B,H,W,D, C=1]
    voxel_mask: [B,H,W,D, C]
    """
    try:
        cmap_me, norm = cmap_for_dataset(dataset)

        for batch_id in range(voxel_img.shape[0]):
            voxel_img_batch = voxel_img[batch_id,:,:,:,0]*(dataset.HU_MAX - dataset.HU_MIN) + dataset.HU_MIN
            voxel_mask_batch = np.argmax(voxel_mask[batch_id,:,:,:,:], axis=-1)
            height = voxel_img_batch.shape[-1]

            f,axarr = plt.subplots(2,plots)
            for plt_idx, z_idx in enumerate(np.random.choice(height, plots, replace=False)):
                axarr[0][plt_idx].imshow(voxel_img_batch[:,:,z_idx], cmap='gray')
                axarr[1][plt_idx].imshow(voxel_img_batch[:,:,z_idx], cmap='gray', alpha=0.2)
                axarr[1][plt_idx].imshow(voxel_mask_batch[:,:,z_idx], cmap=cmap_me, norm=norm, interpolation='none')
                axarr[0][plt_idx].set_title('Slice: {}/{}'.format(z_idx+1, height))
            
            name, patient_id, study_id = utils.get_name_patient_study_id(meta2[batch_id])
            if study_id is not None:
                filename = patient_id + '\n' + study_id
            else:
                filename = patient_id
            plt.suptitle(filename)
            plt.show()

                
    except:
        traceback.print_exc()
        pdb.set_trace()

def viz_3d_data(voxel_imgs, voxel_masks, meta1, meta2, dataset):
    """
    - voxel_imgs: [B,H,W,D]
    """

    try:
        """
        Ref: https://plotly.com/python/visualizing-mri-volume-slices/
        """
        import plotly.graph_objects as go

        voxel_imgs = voxel_imgs.numpy()
        for batch_id, voxel_img in enumerate(voxel_imgs):

            # Plot Volume
            r,c,d = voxel_img.shape
            d_ = d - 1
            
            fig = go.Figure(
                    frames=[
                        go.Frame(data=go.Surface(
                                        z=(d_ - k) * np.ones((r, c)),
                                        surfacecolor=np.flipud(voxel_img[:,:,d_ - k]),
                                        # surfacecolor=voxel_img[:,:,d_ - k],
                                        cmin=-1000, cmax=3000
                                    )
                                , name=str(k) # you need to name the frame for the animation to behave properly
                            ) for k in range(d)
                        ]
                    )
            
            # Add data to be displayed before animation starts
            fig.add_trace(go.Surface(
                            z=d_ * np.ones((r, c))
                            , surfacecolor=np.flipud(voxel_img[:,:,d_])
                            # , surfacecolor=voxel_img[:,:,d_]
                            , colorscale='Gray'
                            # , cmin=config.HU_MIN, cmax=config.HU_MIN
                            , cmin=-1000, cmax=3000
                            , colorbar=dict(thickness=20, ticklen=4)
                        )
                    )

            def frame_args(duration):
                return {
                        "frame": {"duration": duration},
                        "mode": "immediate",
                        "fromcurrent": True,
                        "transition": {"duration": duration, "easing": "linear"},
                    }

            sliders = [
                {
                    "pad": {"b": 10, "t": 60},
                    "len": 0.9,
                    "x": 0.1,
                    "y": 0,
                    "steps": [
                        {
                            "args": [[f.name], frame_args(0)],
                            "label": str(k),
                            "method": "animate",
                        }
                        for k, f in enumerate(fig.frames)
                    ],
                }
            ]

            fig.update_layout(
                title='Slices in volumetric data',
                width=600,
                height=600,
                scene=dict(
                            # zaxis=dict(range=[0, d*0.1], autorange=False),
                            zaxis=dict(range=[0, d], autorange=False),
                            aspectratio=dict(x=1, y=1, z=1),
                            ),
                updatemenus = [
                    {
                        "buttons": [
                            {
                                "args": [None, frame_args(50)],
                                "label": "&#9654;", # play symbol
                                "method": "animate",
                            },
                            {
                                "args": [[None], frame_args(0)],
                                "label": "&#9724;", # pause symbol
                                "method": "animate",
                            },
                        ],
                        "direction": "left",
                        "pad": {"r": 10, "t": 70},
                        "type": "buttons",
                        "x": 0.1,
                        "y": 0,
                    }
                ],
                sliders=sliders
            )
            fig.show()

            # Plot Mask (3D)

    except:
        traceback.print_exc()
        pdb.set_trace()

def viz_3d_mask(voxel_masks, dataset, meta1, meta2, label_map_full=False):
    """
    Expects a [B,H,W,D] shaped mask
    """
    try:
        import plotly.graph_objects as go
        import skimage
        import skimage.measure

        LABEL_MAP = dataset.get_label_map(label_map_full=label_map_full)
        LABEL_COLORS = dataset.get_label_colors()
        label_ids = meta1[:,-len(LABEL_MAP):].numpy()
        if np.sum(label_ids) < len(meta1): # if <= batch_size: return 0
            print (' - [viz_3d_mask()] No labels present: np.sum(label_ids): ', np.sum(label_ids))
            return 0

        # Step 1 - Loop over all batch_ids
        print (' ------------------------ VIZ 3D ------------------------')
        for batch_id, voxel_mask in enumerate(voxel_masks):
            fig = go.Figure()
            label_ids = np.unique(voxel_mask)
            print (' - label_ids: ', label_ids)

            import tensorflow as tf
            # voxel_mask_ = tf.image.rot90(voxel_mask, k=3)
            # voxel_mask_ = tf.transpose(tf.reverse(voxel_mask, [0]), [1,0,2])
            voxel_mask_ = voxel_mask
            print (' -voxel_mask: ', voxel_mask.shape)

            # Step 2 - Loop over all label_ids
            for i_, label_id in enumerate(label_ids):
                
                if label_id == 0 : continue
                name, color = get_info_from_label_id(label_id, LABEL_MAP, LABEL_COLORS)
                print (' - label_id: ', label_id, '(',name,')')

                # Get surface information
                voxel_mask_tmp = np.array(copy.deepcopy(voxel_mask_)).astype(config.DATATYPE_VOXEL_MASK)
                voxel_mask_tmp[voxel_mask_tmp != label_id] = 0
                verts, faces, _, _ = skimage.measure.marching_cubes(voxel_mask_tmp, step_size=1)
                
                # https://plotly.github.io/plotly.py-docs/generated/plotly.graph_objects.Mesh3d.html
                visible=True
                fig.add_trace(
                    go.Mesh3d(
                        x=verts[:,0], y=verts[:,1], z=verts[:,2]
                        , i=faces[:,0],j=faces[:,1],k=faces[:,2]
                        , color='rgb({},{},{})'.format(*color)
                        , name=name, showlegend=True
                        , visible=visible
                        # , lighting=go.mesh3d.Lighting(ambient=0.5)
                    )
                )
            
            fig.update_layout(
                scene = dict(
                    xaxis = dict(nticks=10, range=[0,voxel_mask.shape[0]], title='X-axis'),
                    yaxis = dict(nticks=10, range=[0,voxel_mask.shape[1]]),
                    zaxis = dict(nticks=10, range=[0,voxel_mask.shape[2]]),
                )
                ,width=700,
                margin=dict(r=20, l=50, b=10, t=50)
            )
            fig.update_layout(legend_title_text='Labels', showlegend=True)
            fig.update_layout(scene_aspectmode='cube')
            fig.update_layout(title_text='{} (BatchId={})'.format(meta2, batch_id))
            fig.show()

    except:
        traceback.print_exc()
        pdb.set_trace()
