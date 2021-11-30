# Import private libraries
import src.config as config
import src.dataloader.utils as utils


# Import public libraries
import pdb
import math
import traceback
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import SimpleITK as sitk
import tensorflow as tf
import tensorflow_addons as tfa

class Rotate3D:

    def __init__(self):
        self.name = 'Rotate3D'

    @tf.function
    def execute(self, x, y, meta1, meta2):
        """
        Rotates a 3D image along the z-axis by some random angle
        - Ref: https://www.tensorflow.org/api_docs/python/tf/image/rot90
        
        Parameters
        ----------
        x: tf.Tensor
            This is the 3D image of dtype=tf.int16 and shape=(H,W,C,1)
        y: tf.Tensor
            This is the 3D mask of dtype=tf.uint8 and shape=(H,W,C,Labels)
        meta1 = tf.Tensor
            This contains some indexing and meta info. Irrelevant to this function
        meta2 = tf.Tensor
            This contains some string information on patient identification. Irrelevant to this function
        """
        
        try:
            if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= 0.2:
                rotate_count =  tf.random.uniform([], minval=1, maxval=4, dtype=tf.dtypes.int32)
                
                # k=3 (rot=270) (anti-clockwise)
                if rotate_count == 3:
                    xtmp = tf.transpose(tf.reverse(x, [0]), [1,0,2,3])
                    ytmp = tf.transpose(tf.reverse(y, [0]), [1,0,2,3])

                # k = 1 (rot=90) (anti-clockwise)
                elif rotate_count == 1:
                    xtmp = tf.reverse(tf.transpose(x, [1,0,2,3]), [0])
                    ytmp = tf.reverse(tf.transpose(y, [1,0,2,3]), [0])

                # k=2 (rot=180) (clock-wise)
                elif rotate_count == 2:
                    xtmp = tf.reverse(x, [0,1])
                    ytmp = tf.reverse(y, [0,1])
                
                else:
                    xtmp = x
                    ytmp = y
                
                return xtmp, ytmp, meta1, meta2
                # return xtmp.read_value(), ytmp.read_value(), meta1, meta2
                
            else:
                return x, y, meta1, meta2
        except:
            traceback.print_exc()
            return x, y, meta1, meta2
    
    @tf.function
    def execute2(self, x_moving, x_fixed, y_moving, y_fixed, meta1, meta2):

        try:
            if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= 0.5:
                rotate_count =  tf.random.uniform([], minval=1, maxval=4, dtype=tf.dtypes.int32)
                
                # k=3 (rot=270) (anti-clockwise)
                if rotate_count == 3:
                    x_moving_tmp = tf.transpose(tf.reverse(x_moving, [0]), [1,0,2,3])
                    x_fixed_tmp = tf.transpose(tf.reverse(x_fixed, [0]), [1,0,2,3])
                    y_moving_tmp = tf.transpose(tf.reverse(y_moving, [0]), [1,0,2,3])
                    y_fixed_tmp = tf.transpose(tf.reverse(y_fixed, [0]), [1,0,2,3])

                # k = 1 (rot=90) (anti-clockwise)
                elif rotate_count == 1:
                    x_moving_tmp = tf.reverse(tf.transpose(x_moving, [1,0,2,3]), [0])
                    x_fixed_tmp = tf.reverse(tf.transpose(x_fixed, [1,0,2,3]), [0])
                    y_moving_tmp = tf.reverse(tf.transpose(y_moving, [1,0,2,3]), [0])
                    y_fixed_tmp = tf.reverse(tf.transpose(y_fixed, [1,0,2,3]), [0])

                # k=2 (rot=180) (clock-wise)
                elif rotate_count == 2:
                    x_moving_tmp = tf.reverse(x_moving, [0,1])
                    x_fixed_tmp = tf.reverse(x_fixed, [0,1])
                    y_moving_tmp = tf.reverse(y_moving, [0,1])
                    y_fixed_tmp = tf.reverse(y_fixed, [0,1])
                
                else:
                    x_moving_tmp = x_moving
                    x_fixed_tmp = x_fixed
                    y_moving_tmp = y_moving
                    y_fixed_tmp = y_fixed
                
                return x_moving_tmp, x_fixed_tmp, y_moving_tmp, y_fixed_tmp, meta1, meta2
                
            else:
                return x_moving, x_fixed, y_moving, y_fixed, meta1, meta2

        except:
            tf.print(' - [ERROR][Rotate3D][execute2]')
            return x_moving, x_fixed, y_moving, y_fixed, meta1, meta2

class Rotate3DSmall:

    def __init__(self, label_map, mask_type, prob=0.2, angle_degrees=15, interpolation='bilinear'):
        
        self.name = 'Rotate3DSmall'
        
        self.label_ids = label_map.values()
        self.class_count = len(label_map)
        self.mask_type = mask_type

        self.prob = prob
        self.angle_degrees = angle_degrees
        self.interpolation = interpolation
    
    @tf.function
    def execute(self, x, y, meta1, meta2):
        """
         - Ref: https://www.tensorflow.org/addons/api_docs/python/tfa/image/rotate

        """

        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)

        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.prob:

            angle_radians = tf.random.uniform([], minval=math.radians(-self.angle_degrees), maxval=math.radians(self.angle_degrees)
                                    , dtype=tf.dtypes.float32)
            

            if self.mask_type == config.MASK_TYPE_ONEHOT:
                """
                 - x: [H,W,D,1]
                 - y: [H,W,D,class]
                """
                
                x = tf.expand_dims(x[:,:,:,0], axis=0) # [1,H,W,D]
                x = tf.transpose(x) # [D,W,H,1] 
                x = tfa.image.rotate(x, angle_radians, interpolation='bilinear') # [D,W,H,1]
                x = tf.transpose(x) # [1,H,W,D]
                x = tf.expand_dims(x[0], axis=-1) # [H,W,D,1]
                
                y = tf.concat([
                        tf.expand_dims(
                            tf.transpose( # [1,H,W,D]
                                tfa.image.rotate( # [D,W,H,1]
                                    tf.transpose( # [D,W,H,1]
                                        tf.expand_dims(y[:,:,:,class_id], axis=0) # [1,H,W,D]
                                    )
                                    , angle_radians, interpolation='bilinear'
                                )
                            )[0] # [H,W,D]
                            , axis=-1 # [H,W,D,1]
                        ) for class_id in range(self.class_count)
                    ], axis=-1) # [H,W,D,10]
                y = tf.where(tf.math.greater_equal(y,0.5), 1.0, y)
                y = tf.where(tf.math.less(y,0.5), 0.0, y)

            elif self.mask_type == config.MASK_TYPE_COMBINED:
                """
                 - x: [H,W,D]
                 - y: [H,W,D]
                """
                
                x = tf.expand_dims(x,axis=0) # [1,H,W,D]
                x = tf.transpose(x) # [D,W,H,1]
                x = tfa.image.rotate(x, angle_radians, interpolation=self.interpolation) # [D,W,H,1]
                x = tf.transpose(x) # [1,H,W,D]
                x = x[0] # [H,W,D]

                y = tf.concat([tf.expand_dims(tf.math.equal(y, label), axis=-1) for label in self.label_ids], axis=-1) # [H,W,D,L]
                y = tf.cast(y, dtype=tf.float32)
                y = tf.concat([
                        tf.expand_dims(
                            tf.transpose( # [1,H,W,D]
                                tfa.image.rotate( # [D,W,H,1]
                                    tf.transpose( # [D,W,H,1]
                                        tf.expand_dims(y[:,:,:,class_id], axis=0) # [1,H,W,D]
                                    )
                                    , angle_radians, interpolation='bilinear'
                                )
                            )[0] # [H,W,D]
                            , axis=-1 # [H,W,D,1]
                        ) for class_id in range(self.class_count)
                    ], axis=-1) # [H,W,D,L]
                y = tf.where(tf.math.greater_equal(y,0.5), 1.0, y)
                y = tf.where(tf.math.less(y,0.5), 0.0, y) 
                y = tf.math.argmax(y, axis=-1) # [H,W,D]

            x = tf.cast(x, dtype=tf.float32)
            y = tf.cast(y, dtype=tf.float32)
           
        else:
            x = tf.cast(x, dtype=tf.float32)
            y = tf.cast(y, dtype=tf.float32)
        
        return x, y, meta1, meta2

class Rotate3DSmallZ:

    def __init__(self, label_map, mask_type, prob=0.2, angle_degrees=5, interpolation='bilinear'):
        
        self.name = 'Rotate3DSmallZ'
        
        self.label_ids = label_map.values()
        self.class_count = len(label_map)
        self.mask_type = mask_type

        self.prob = prob
        self.angle_degrees = angle_degrees
        self.interpolation = interpolation
    
    @tf.function
    def execute(self, x, y, meta1, meta2):
        """
        Params
        ------
        x: [H,W,D,1]
        y: [H,W,D,C]
         - Ref: https://www.tensorflow.org/addons/api_docs/python/tfa/image/rotate

        """

        x = tf.cast(x, dtype=tf.float32)
        y = tf.cast(y, dtype=tf.float32)

        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.prob:

            angle_radians = tf.random.uniform([], minval=math.radians(-self.angle_degrees), maxval=math.radians(self.angle_degrees)
                                    , dtype=tf.dtypes.float32)
            
            if self.mask_type == config.MASK_TYPE_ONEHOT:
                """
                 - x: [H,W,D,1]
                 - y: [H,W,D,class]
                """
                
                x = tfa.image.rotate(x, angle_radians, interpolation='bilinear') # [H,W,D,1]
                
                y = tf.concat([
                                tfa.image.rotate(
                                    tf.expand_dims(y[:,:,:,class_id], axis=-1) # [H,W,D,1]
                                    , angle_radians, interpolation='bilinear'
                                )
                                for class_id in range(self.class_count)
                            ], axis=-1) # [H,W,D,10]
                y = tf.where(tf.math.greater_equal(y,0.5), 1.0, y)
                y = tf.where(tf.math.less(y,0.5), 0.0, y)

            elif self.mask_type == config.MASK_TYPE_COMBINED:
                pass

            x = tf.cast(x, dtype=tf.float32)
            y = tf.cast(y, dtype=tf.float32)
           
        else:
            x = tf.cast(x, dtype=tf.float32)
            y = tf.cast(y, dtype=tf.float32)
        
        return x, y, meta1, meta2

class Translate:

    def __init__(self, label_map, translations=[40,40], prob=0.2):

        self.translations = translations
        self.prob         = prob
        self.label_ids    = label_map.values()
        self.class_count  = len(label_map)
        self.name         = 'Translate' 



    @tf.function
    def execute(self,x,y,meta1,meta2):
        """
        Params
        ------
        x: [H,W,D,1]
        y: [H,W,D,class]

        Ref
        ---
        - tfa.image.translate(image): image= (num_images, num_rows, num_columns, num_channels)
        """
        
        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.prob:
            
            translate_x = tf.random.uniform([], minval=-self.translations[0], maxval=self.translations[0], dtype=tf.dtypes.int32)
            translate_y = tf.random.uniform([], minval=-self.translations[1], maxval=self.translations[1], dtype=tf.dtypes.int32)

            x = tf.expand_dims(x[:,:,:,0], axis=0) # [1,H,W,D]
            x = tf.transpose(x) # [D,W,H,1] 
            x = tfa.image.translate(x, [translate_x, translate_y], interpolation='bilinear') # [D,W,H,1]; x=(num_images, num_rows, num_columns, num_channels)
            x = tf.transpose(x) # [1,H,W,D]
            x = tf.expand_dims(x[0], axis=-1) # [H,W,D,1]
            
            y = tf.concat([
                    tf.expand_dims(
                        tf.transpose( # [1,H,W,D]
                            tfa.image.translate( # [D,W,H,1]
                                tf.transpose( # [D,W,H,1]
                                    tf.expand_dims(y[:,:,:,class_id], axis=0) # [1,H,W,D]
                                )
                                , [translate_x, translate_y], interpolation='bilinear'
                            )
                        )[0] # [H,W,D]
                        , axis=-1 # [H,W,D,1]
                    ) for class_id in range(self.class_count)
                ], axis=-1) # [H,W,D,10]

        return x,y,meta1,meta2

class Noise:

    def __init__(self, x_shape, mean=0.0, std=0.1, prob=0.2):
        
        self.mean    = mean
        self.std     = std 
        self.prob    = prob
        self.x_shape = x_shape
        self.name    = 'Noise'
    
    @tf.function
    def execute(self,x,y,meta1,meta2):
        """
        Params
        ------
        x: [H,W,D,1]
        y: [H,W,D,class]
        """

        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.prob:
            
            x = x + tf.random.normal(self.x_shape, self.mean, self.std)

        return x,y,meta1,meta2

class Deform2Punt5D:

    def __init__(self, img_shape, label_map, grid_points=50, stddev=2.0, div_factor=2, prob=0.2, debug=False):
        """
        img_shape = [H,W] or [H,W,D]
        """

        self.name = 'Deform2Punt5D'

        if debug:
            import os
            import psutil
            import pynvml 
            pynvml.nvmlInit()
            self.device_id = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.process = psutil.Process(os.getpid())

        self.img_shape = img_shape[:2] # no deformation in z-dimension if img=3D
        self.depth = img_shape[-1]
        self.grid_points = grid_points
        self.stddev = stddev
        self.div_factor = int(div_factor)
        self.debug = debug
        self.label_ids = label_map.values()
        self.prob = prob

        self.flattened_grid_locations = []

        self._get_grid_control_points()
        self._get_grid_locations(*self.img_shape)
    
    def _get_grid_control_points(self):
        
        # Step 1 - Define grid shape using control grid spacing & final image shape
        grid_shape = np.zeros(len(self.img_shape), dtype=int)

        for idx in range(len(self.img_shape)):
            num_elem = float(self.img_shape[idx])
            if num_elem % 2 == 0:
                grid_shape[idx] = np.ceil( (num_elem - 1) / (2*self.grid_points) + 0.5) * 2 + 2
            else:
                grid_shape[idx] = np.ceil((num_elem - 1) / (2*self.grid_points)) * 2 + 3

        coords = []
        for i, size in enumerate(grid_shape):
            coords.append(tf.linspace(-(size - 1) / 2*self.grid_points, (size - 1) / 2*self.grid_points, size))
        permutation = np.roll(np.arange(len(coords) + 1), -1)
        self.grid_control_points_orig =  tf.cast(tf.transpose(tf.meshgrid(*coords, indexing="ij"), permutation), dtype=tf.float32)
        self.grid_control_points_orig += tf.expand_dims(tf.expand_dims(tf.constant(self.img_shape, dtype=tf.float32)/2.0,0),0)
        self.grid_control_points = tf.reshape(self.grid_control_points_orig, [-1,2])
    
    def _get_grid_locations(self, image_height, image_width):
        
        image_height = image_height//self.div_factor
        image_width = image_width//self.div_factor

        y_range = np.linspace(0, image_height - 1, image_height)
        x_range = np.linspace(0, image_width - 1, image_width)
        y_grid, x_grid = np.meshgrid(y_range, x_range, indexing="ij") # grid_x, grid_y = tf.meshgrid(tf.range(width), tf.range(height))
        grid_locations = np.stack((y_grid, x_grid), -1)
        
        flattened_grid_locations = np.reshape(grid_locations, [image_height * image_width, 2])
        self.flattened_grid_locations = tf.cast(tf.expand_dims(flattened_grid_locations, 0), dtype=tf.float32)

        return flattened_grid_locations
    
    @tf.function
    def _get_dense_flow(self, grid_control_points, grid_control_points_new):
        """
        params
        -----
        grid_control_points: [B, points, 2]
        grid_control_points_new: [B, points, 2]

        return
        dense_flows: [1,H,W,2]
        """
        
        # Step 1 - Get flows
        source_control_point_locations = tf.cast(grid_control_points/self.div_factor, dtype=tf.float32)
        dest_control_point_locations = tf.cast(grid_control_points_new/self.div_factor, dtype=tf.float32)
        control_point_flows = dest_control_point_locations - source_control_point_locations

        # Step 2 - Get dense flow via bspline interp  
        if self.debug:
            import pynvml
            tf.print (' - [_get_dense_flow()] GPU mem: ', '%.4f' % (pynvml.nvmlDeviceGetMemoryInfo(self.device_id).used/1024/1024/1000),'GB), ')
            tf.print (' - [_get_dense_flow()] img size: ', self.img_shape)
            tf.print (' - [_get_dense_flow()] img size for bspline interp: ', self.img_shape[0]//self.div_factor, self.img_shape[1]//self.div_factor)
        flattened_flows = tfa.image.interpolate_spline(
                dest_control_point_locations,
                control_point_flows,
                self.flattened_grid_locations,
                order=2
        )
        if self.debug:
            import pynvml
            tf.print (' - [_get_dense_flow()] GPU mem: ', '%.4f' % (pynvml.nvmlDeviceGetMemoryInfo(self.device_id).used/1024/1024/1000),'GB), ')

        # Step 3 - Reshape dense flow to original image size
        dense_flows = tf.reshape(flattened_flows, [1, self.img_shape[0]//self.div_factor, self.img_shape[1]//self.div_factor, 2])
        dense_flows = tf.image.resize(dense_flows, (self.img_shape[0], self.img_shape[1]))
        if self.debug:
            print (' - [_get_dense_flow()] dense flow: ', dense_flows.shape)

        return dense_flows
        
    def execute2D(self, x, y, show=True):
        """
        x = [H,W]
        y = [H,W,L]
        """
        
        # Step 1 - Get new control points and dense flow for each slice
        grid_control_points_new = self.grid_control_points +  tf.random.normal(self.grid_control_points.shape, 0, self.stddev)
        x_flow = self._get_dense_flow(tf.expand_dims(self.grid_control_points,0), tf.expand_dims(grid_control_points_new,0)) # [1,H,W,2]

        # Step 2 - Transform
        x_tf = tf.expand_dims(tf.expand_dims(x, -1), 0) # [1,H,W,1]
        y_tf = tf.expand_dims(y, 0) # [1,H,W,L]
        if 0:
            x_tf, x_flow = tfa.image.sparse_image_warp(x_tf, tf.expand_dims(self.grid_control_points,0), tf.expand_dims(grid_control_points_new,0))
        else:
            x_tf = tfa.image.dense_image_warp(x_tf, x_flow) # [1,H,W,1]
            y_tf = tf.concat([
                tfa.image.dense_image_warp(
                        tf.expand_dims(y_tf[:,:,:,class_id],-1), x_flow
                    ) for class_id in self.label_ids]
                , axis=-1
            ) # [B,H,W,L]
            y_tf = tf.where(tf.math.greater_equal(y_tf, 0.5), 1.0, y_tf)
            y_tf = tf.where(tf.math.less(y_tf, 0.5), 0.0, y_tf)


        # Step 99 - Show
        if show:
            f,axarr = plt.subplots(2,2, figsize=(15,10))
            axarr[0][0].imshow(x, cmap='gray') 
            y_plot = tf.argmax(y, axis=-1) # [H,W]
            axarr[0][0].imshow(y_plot, alpha=0.5)
            grid_og = self.grid_control_points_orig
            axarr[0][0].plot(grid_og[:,:,1], grid_og[:,:,0], 'y-.', alpha=0.2)
            axarr[0][0].plot(tf.transpose(grid_og[:,:,1]), tf.transpose(grid_og[:,:,0]), 'y-.', alpha=0.2)
            axarr[0][0].set_xlim([0, self.img_shape[1]])
            axarr[0][0].set_ylim([0, self.img_shape[0]])

            axarr[0][1].imshow(x_tf[0,:,:,0], cmap='gray')
            y_tf_plot = tf.argmax(y_tf, axis=-1)[0,:,:] # [H,W]
            axarr[0][1].imshow(y_tf_plot, alpha=0.5)
            grid_new = tf.reshape(grid_control_points_new, self.grid_control_points_orig.shape)
            axarr[0][1].plot(grid_new[:,:,1], grid_new[:,:,0], 'y-.', alpha=0.2)
            axarr[0][1].plot(tf.transpose(grid_new[:,:,1]), tf.transpose(grid_new[:,:,0]), 'y-.', alpha=0.2)
            axarr[0][1].set_xlim([0, self.img_shape[1]])
            axarr[0][1].set_ylim([0, self.img_shape[0]])

            axarr_dy = axarr[1][0].imshow(x_flow[0,:,:,1], vmin=-7.5, vmax=7.5, interpolation='none', cmap='rainbow')
            f.colorbar(axarr_dy, ax=axarr[1][0], extend='both')
            axarr_dx = axarr[1][1].imshow(x_flow[0,:,:,0], vmin=-7.5, vmax=7.5, interpolation='none', cmap='rainbow')
            f.colorbar(axarr_dx, ax=axarr[1][1], extend='both')

            plt.suptitle('Image Size: {} \n Control Points: {}\n StdDev: {}\nDiv Factor={}'.format(self.img_shape, self.grid_points, self.stddev, self.div_factor))
            plt.show()
            pdb.set_trace()

        return x_tf, y_tf

    @tf.function
    def execute(self, x, y, meta1, meta2, show=False):
        """
        x = [H,W,D,1]
        y = [H,W,D,L]
        No deformation in z-axis
        """

        if tf.random.uniform([], minval=0, maxval=1, dtype=tf.dtypes.float32) <= self.prob:
            if self.debug:
                tf.print (' - [Deform2Punt5D()][execute()] img_shape: ', self.img_shape, ' || depth: ', self.depth)
        
            # Step 1 - Get new control points and dense flow for each slice
            grid_control_points_new = self.grid_control_points +  tf.random.normal(self.grid_control_points.shape, 0, self.stddev)
            x_flow = self._get_dense_flow(tf.expand_dims(self.grid_control_points,0), tf.expand_dims(grid_control_points_new,0)) # [1,H,W,2]

            # Step 2 - Transform
            if show:
                idx = 70
                import matplotlib.pyplot as plt
                f,axarr = plt.subplots(2,2, figsize=(15,10))
                
                x_slice = x[:,:,idx,0] # [H,W]
                axarr[0][0].imshow(x_slice, cmap='gray') 
                y_slice = tf.argmax(y[:,:,idx,:], axis=-1) # [H,W]
                axarr[0][0].imshow(y_slice, cmap='gray', alpha=0.5)
                grid_og = self.grid_control_points_orig
                axarr[0][0].plot(grid_og[:,:,1], grid_og[:,:,0], 'y-.', alpha=0.2)
                axarr[0][0].plot(tf.transpose(grid_og[:,:,1]), tf.transpose(grid_og[:,:,0]), 'y-.', alpha=0.5)
                axarr[0][0].set_xlim([0, self.img_shape[1]])
                axarr[0][0].set_ylim([0, self.img_shape[0]])
                
            x = tf.transpose(x, [2,0,1,3]) # [H,W,D,1] -> [D,H,W,1]
            y = tf.transpose(y, [2,0,1,3]) # [H,W,D,L] -> [D,H,W,L]
            
            if self.debug:
                import pynvml
                tf.print (' - [Deform2Punt5D][execute()] GPU mem: ', '%.4f' % (pynvml.nvmlDeviceGetMemoryInfo(self.device_id).used/1024/1024/1000),'GB, ')
            x_flow_repeat = tf.repeat(x_flow, self.depth, axis=0) # [B,H,W,2] or [D,H,W,2]
            if self.debug:
                tf.print (' - [execute()] x_tf: ', x.shape, ' || x_flow_repeat: ', x_flow_repeat.shape)
            x = tfa.image.dense_image_warp(x, x_flow_repeat) 
            x =  tf.transpose(x, [1,2,0,3]) # [D,H,W,1] -> [H,W,D,1]
            y = tf.concat([
                    tfa.image.dense_image_warp(
                            tf.expand_dims(y[:,:,:,class_id],-1), x_flow_repeat
                        ) for class_id in self.label_ids]
                    , axis=-1
                ) # [D,H,W,L]
            y = tf.where(tf.math.greater_equal(y, 0.5), 1.0, y)
            y = tf.where(tf.math.less(y, 0.5), 0.0, y)
            y =  tf.transpose(y, [1,2,0,3]) # [D,H,W,L] --> [H,W,D,L]
            if self.debug:
                import pynvml
                tf.print (' - [execute()] GPU mem: ', '%.4f' % (pynvml.nvmlDeviceGetMemoryInfo(self.device_id).used/1024/1024/1000),'GB, ')
            
            if show:
                
                x_slice = x[:,:,idx,0] # [H,W]
                axarr[0][1].imshow(x_slice, cmap='gray')
                y_slice = tf.argmax(y[:,:,idx,:], axis=-1) # [H,W]
                axarr[0][1].imshow(y_slice, cmap='gray', alpha=0.5)
                grid_new = tf.reshape(grid_control_points_new, self.grid_control_points_orig.shape)
                axarr[0][1].plot(grid_new[:,:,1], grid_new[:,:,0], 'y-.', alpha=0.2)
                axarr[0][1].plot(tf.transpose(grid_new[:,:,1]), tf.transpose(grid_new[:,:,0]), 'y-.', alpha=0.5)
                axarr[0][1].set_xlim([0, self.img_shape[1]])
                axarr[0][1].set_ylim([0, self.img_shape[0]])

                axarr[1][0].imshow(x_flow[0,:,:,1], cmap='gray')
                axarr[1][1].imshow(x_flow[0,:,:,0], cmap='gray')
                plt.show()

        # return x_tf, y, meta1, meta2
        return x, y, meta1, meta2
