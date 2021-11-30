# Import internal libraries
import src.config as config

# Import external libraries
import pdb
import traceback
import numpy as np
import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow_probability as tfp

############################################################
#                     3D MODEL BLOCKS                      #
############################################################

class ConvBlock3D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation='relu'
                    , trainable=False
                    , pool=False
                    , name=''):
        super(ConvBlock3D, self).__init__(name='{}_ConvBlock3D'.format(name))

        # Step 0 - Init
        self.pool      = pool
        self.filters   = filters
        self.trainable = trainable
        if type(filters) == int:
            filters = [filters]
        
        # Step 1 - Conv Blocks
        self.conv_layer = tf.keras.Sequential()
        for filter_id, filter_count in enumerate(filters):
            self.conv_layer.add(
                tf.keras.layers.Conv3D(filters=filter_count, kernel_size=kernel_size, strides=strides, padding=padding
                        , dilation_rate=dilation_rate
                        , activation=activation
                        , name='Conv_{}'.format(filter_id))
            )
            self.conv_layer.add(tf.keras.layers.BatchNormalization(trainable=trainable))
            # https://www.tensorflow.org/api_docs/python/tf/keras/layers/BatchNormalization
            ## with the argument training=False (which is the default), the layer normalizes its output using a moving average of the mean and standard deviation of the batches it has seen during training
            # if filter_id == 0 and dropout is not None:
            #     self.conv_layer.add(tf.keras.layers.Dropout(rate=dropout, name='DropOut'))
        
        # Step 2 - Pooling Blocks
        if self.pool:
            self.pool_layer = tf.keras.layers.MaxPooling3D((2,2,2), strides=(2,2,2), name='Pool')
            
    @tf.function
    def call(self, x):
        
        x = self.conv_layer(x)
        
        if self.pool:
            return x, self.pool_layer(x)
        else:
            return x

class ConvBlock3DSERes(tf.keras.layers.Layer):
    """
    For channel-wise attention
    """

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation='relu'
                    , trainable=False
                    , pool=False
                    , squeeze_ratio=None
                    , init=False
                    , name=''):

        super(ConvBlock3DSERes, self).__init__(name='{}_ConvBlock3DSERes'.format(name))

        # Step 0 - Init
        self.init = init
        self.trainable = trainable

        # Step 1 - Init (to get equivalent feature map count)
        if self.init:
            self.convblock_filterequalizer = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                                    , activation='relu'
                                                    )

        # Step 2- Conv Block
        self.convblock_res = ConvBlock3D(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
                            , dilation_rate=dilation_rate
                            , activation=activation
                            , trainable=trainable
                            , pool=False
                            , name=name
                            )

        # Step 3 - Squeeze Block
        """
        Ref: https://github.com/imkhan2/se-resnet/blob/master/se_resnet.py
        """
        self.squeeze_ratio = squeeze_ratio
        if self.squeeze_ratio is not None:
            self.seblock = tf.keras.Sequential()
            self.seblock.add(tf.keras.layers.GlobalAveragePooling3D())
            self.seblock.add(tf.keras.layers.Reshape(target_shape=(1,1,1,filters[0])))
            self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0]//squeeze_ratio, kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                    , activation='relu'))
            self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                    , activation='sigmoid'))

        self.pool = pool
        if self.pool:
            self.pool_layer = tf.keras.layers.MaxPooling3D((2,2,2), strides=(2,2,2), name='{}_Pool'.format(name))

    @tf.function
    def call(self, x):
        
        if self.init:
            x = self.convblock_filterequalizer(x)

        x_res = self.convblock_res(x)

        if self.squeeze_ratio is not None:
            x_se = self.seblock(x_res) # squeeze and then get excitation factor
            x_res = tf.math.multiply(x_res, x_se) # excited block

        y = x + x_res

        if self.pool:
            return y, self.pool_layer(y)
        else:
            return y


class ConvBlock3DDropOut(tf.keras.layers.Layer):
    
    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation='relu'
                    , trainable=False
                    , dropout=None
                    , pool=False
                    , name=''):
        super(ConvBlock3DDropOut, self).__init__(name='{}_ConvBlock3DDropOut'.format(name))

        self.pool      = pool
        self.filters   = filters
        self.trainable = trainable

        if type(filters) == int:
            filters = [filters]
        
        self.conv_layer = tf.keras.Sequential()
        for filter_id, filter_count in enumerate(filters):
            
            if dropout is not None:
                self.conv_layer.add(tf.keras.layers.Dropout(rate=dropout, name='DropOut_{}'.format(filter_id))) # before every conv layer (could also be after every layer?)
            
            self.conv_layer.add(
                tf.keras.layers.Conv3D(filters=filter_count, kernel_size=kernel_size, strides=strides, padding=padding
                        , dilation_rate=dilation_rate
                        , activation=activation
                        , name='Conv_{}'.format(filter_id))
            )
            self.conv_layer.add(tf.keras.layers.BatchNormalization(trainable=trainable))
            
        if self.pool:
            self.pool_layer = tf.keras.layers.MaxPooling3D((2,2,2), strides=(2,2,2), name='Pool')
            
    @tf.function
    def call(self, x):
        
        x = self.conv_layer(x)
        
        if self.pool:
            return x, self.pool_layer(x)
        else:
            return x

class ConvBlock3DSEResDropout(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation='relu'
                    , trainable=False
                    , dropout=None
                    , pool=False
                    , squeeze_ratio=None
                    , init=False
                    , name=''):

        super(ConvBlock3DSEResDropout, self).__init__(name='{}_ConvBlock3DSEResDropout'.format(name))

        self.init = init
        self.trainable = trainable

        if self.init:
            self.convblock_filterequalizer = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                                    , activation='relu'
                                                    )

        self.convblock_res = ConvBlock3DDropOut(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
                            , dilation_rate=dilation_rate
                            , activation=activation
                            , trainable=trainable
                            , dropout=dropout
                            , pool=False
                            , name=name
                            )

        """
        Ref: https://github.com/imkhan2/se-resnet/blob/master/se_resnet.py
        """
        self.squeeze_ratio = squeeze_ratio
        if self.squeeze_ratio is not None:
            self.seblock = tf.keras.Sequential()
            self.seblock.add(tf.keras.layers.GlobalAveragePooling3D())
            self.seblock.add(tf.keras.layers.Reshape(target_shape=(1,1,1,filters[0])))
            self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0]//squeeze_ratio, kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                    , activation='relu'))
            self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                    , activation='sigmoid'))

        self.pool = pool
        if self.pool:
            self.pool_layer = tf.keras.layers.MaxPooling3D((2,2,2), strides=(2,2,2), name='{}_Pool'.format(name))

    @tf.function
    def call(self, x):
        
        if self.init:
            x = self.convblock_filterequalizer(x)

        x_res = self.convblock_res(x)

        if self.squeeze_ratio is not None:
            x_se = self.seblock(x_res) # squeeze and then get excitation factor
            x_res = tf.math.multiply(x_res, x_se) # excited block

        y = x + x_res

        if self.pool:
            return y, self.pool_layer(y)
        else:
            return y


class UpConvBlock3D(tf.keras.layers.Layer):

    def __init__(self, filters, kernel_size=(2,2,2), strides=(2, 2, 2), padding='same', trainable=False, name=''):
        super(UpConvBlock3D, self).__init__(name='{}_UpConv3D'.format(name))
        
        self.trainable = trainable
        self.upconv_layer = tf.keras.Sequential()
        self.upconv_layer.add(tf.keras.layers.Conv3DTranspose(filters, kernel_size, strides, padding=padding
                        , activation='relu'
                        , kernel_regularizer=None
                        , name='UpConv_{}'.format(self.name))
                    )
        # self.upconv_layer.add(tf.keras.layers.BatchNormalization(trainable=trainable))
    
    @tf.function
    def call(self, x):
        return self.upconv_layer(x)


class ConvBlock3DFlipOut(tf.keras.layers.Layer):
    """
    Ref
    - https://www.tensorflow.org/probability/api_docs/python/tfp/layers/Convolution3DFlipout
    """

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation='relu'
                    , trainable=False
                    , pool=False
                    , name=''):
        super(ConvBlock3DFlipOut, self).__init__(name='{}ConvBlock3DFlipOut'.format(name))

        self.pool = pool
        self.filters = filters

        if type(filters) == int:
            filters = [filters]
        
        self.conv_layer = tf.keras.Sequential()
        for filter_id, filter_count in enumerate(filters):
            self.conv_layer.add(
                tfp.layers.Convolution3DFlipout(filters=filter_count, kernel_size=kernel_size, strides=strides, padding=padding
                        , dilation_rate=dilation_rate
                        , activation=activation
                        # , kernel_prior_fn=?
                        , name='Conv3DFlip_{}'.format(filter_id))
            )
            self.conv_layer.add(tfa.layers.GroupNormalization(groups=filter_count//2, trainable=trainable))
            
        if self.pool:
            self.pool_layer = tf.keras.layers.MaxPooling3D((2,2,2), strides=(2,2,2), name='Pool')
            
    def call(self, x):
        
        x = self.conv_layer(x)
        
        if self.pool:
            return x, self.pool_layer(x)
        else:
            return x

class ConvBlock3DSEResFlipOut(tf.keras.layers.Layer):
    """
    For channel-wise attention
    """

    def __init__(self, filters, kernel_size=(3,3,3), strides=(1, 1, 1), padding='same'
                    , dilation_rate=(1,1,1)
                    , activation='relu'
                    , trainable=False
                    , pool=False
                    , squeeze_ratio=None
                    , init=False
                    , name=''):

        super(ConvBlock3DSEResFlipOut, self).__init__(name='{}ConvBlock3DSEResFlipOut'.format(name))

        self.init = init
        if self.init:
            self.convblock_filterequalizer = tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                                    , activation='relu'
                                                    , kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None)

        self.convblock_res = ConvBlock3DFlipOut(filters=filters, kernel_size=kernel_size, strides=strides, padding=padding
                            , dilation_rate=dilation_rate
                            , activation=activation
                            , trainable=trainable
                            , pool=False
                            , name=name
                            )

        """
        Ref: https://github.com/imkhan2/se-resnet/blob/master/se_resnet.py
        """
        self.squeeze_ratio = squeeze_ratio
        if self.squeeze_ratio is not None:
            self.seblock = tf.keras.Sequential()
            self.seblock.add(tf.keras.layers.GlobalAveragePooling3D())
            self.seblock.add(tf.keras.layers.Reshape(target_shape=(1,1,1,filters[0])))
            self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0]//squeeze_ratio, kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                    , activation='relu'
                                    , kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None))
            self.seblock.add(tf.keras.layers.Conv3D(filters=filters[0], kernel_size=(1,1,1), strides=(1,1,1), padding='same'
                                    , activation='sigmoid'
                                    , kernel_regularizer=None, bias_regularizer=None, activity_regularizer=None))

        self.pool = pool
        if self.pool:
            self.pool_layer = tf.keras.layers.MaxPooling3D((2,2,2), strides=(2,2,2), name='{}_Pool'.format(name))

    def call(self, x):
        
        if self.init:
            x = self.convblock_filterequalizer(x)

        x_res = self.convblock_res(x)

        if self.squeeze_ratio is not None:
            x_se = self.seblock(x_res) # squeeze and then get excitation factor
            x_res = tf.math.multiply(x_res, x_se) # excited block

        y = x + x_res

        if self.pool:
            return y, self.pool_layer(y)
        else:
            return y


############################################################
#                        3D MODELS                         #
############################################################

class ModelFocusNetDropOut(tf.keras.Model):
    
    def __init__(self, class_count, trainable=False, verbose=False):
        """
        Params
        ------
        class_count: to know how many class activation maps are needed
        trainable: set to False when BNorm does not need to have its parameters recalculated e.g. during testing
        """
        super(ModelFocusNetDropOut, self).__init__(name='ModelFocusNetDropOut')

        # Step 0 - Init
        self.verbose = verbose 
        self.trainable = trainable

        dropout     = [None, 0.25, 0.25, 0.25, 0.25, 0.25, None, None]
        filters     = [[16,16], [32,32]]
        dilation_xy = [1, 2, 3, 6, 12, 18]
        dilation_z  = [1, 1, 1, 1, 1 , 1]


        # Se-Res Blocks
        self.convblock1 = ConvBlock3DSEResDropout(filters=filters[0], kernel_size=(3,3,1), dilation_rate=(dilation_xy[0], dilation_xy[0], dilation_z[0]), trainable=trainable, dropout=dropout[0], pool=True , squeeze_ratio=2, name='Block1') # Dim/2 (e.g. 240/2=120)(rp=(3,5,10),(1,1,2))
        self.convblock2 = ConvBlock3DSEResDropout(filters=filters[0], kernel_size=(3,3,3), dilation_rate=(dilation_xy[1], dilation_xy[1], dilation_z[1]), trainable=trainable, dropout=dropout[0], pool=False, squeeze_ratio=2, name='Block2') # Dim/2 (e.g. 240/2=120)(rp=(14,18) ,(4,6))
        
        # Dense ASPP
        self.convblock3 = ConvBlock3DDropOut(filters=filters[1], dilation_rate=(dilation_xy[2], dilation_xy[2], dilation_z[2]), trainable=trainable, dropout=dropout[1], pool=False, name='Block3_ASPP') # Dim/2 (e.g. 240/2=120) (rp=(24,30),(8,10))
        self.convblock4 = ConvBlock3DDropOut(filters=filters[1], dilation_rate=(dilation_xy[3], dilation_xy[3], dilation_z[3]), trainable=trainable, dropout=dropout[2], pool=False, name='Block4_ASPP') # Dim/2 (e.g. 240/2=120) (rp=(42,54),(12,14))
        self.convblock5 = ConvBlock3DDropOut(filters=filters[1], dilation_rate=(dilation_xy[4], dilation_xy[4], dilation_z[4]), trainable=trainable, dropout=dropout[3], pool=False, name='Block5_ASPP') # Dim/2 (e.g. 240/2=120) (rp=(78,102),(16,18))
        self.convblock6 = ConvBlock3DDropOut(filters=filters[1], dilation_rate=(dilation_xy[5], dilation_xy[5], dilation_z[5]), trainable=trainable, dropout=dropout[4], pool=False, name='Block6_ASPP') # Dim/2 (e.g. 240/2=120) (rp=(138,176),(20,22))
        self.convblock7 = ConvBlock3DSEResDropout(filters=filters[1], dilation_rate=(dilation_xy[0], dilation_xy[0], dilation_z[0]), trainable=trainable, dropout=dropout[5], pool=False, squeeze_ratio=2, init=True, name='Block7') # Dim/2 (e.g. 240/2=120) (rp=(178,180),(24,26))

        # Upstream
        self.convblock8 = ConvBlock3DSEResDropout(filters=filters[1], dilation_rate=(dilation_xy[0], dilation_xy[0], dilation_z[0]), trainable=trainable, dropout=dropout[6], pool=False, squeeze_ratio=2, init=True, name='Block8') # Dim/2 (e.g. 240/2=120)

        self.upconvblock9 = UpConvBlock3D(filters=filters[0][0], trainable=trainable, name='Block9_1') # Dim/1 (e.g. 240/1=240)
        self.convblock9 = ConvBlock3DSEResDropout(filters=filters[0], dilation_rate=(dilation_xy[0], dilation_xy[0], dilation_z[0]), trainable=trainable, dropout=dropout[7], pool=False, squeeze_ratio=2, init=True, name='Block9') # Dim/1 (e.g. 240/1=240)
        
        # Final
        self.convblock10 = tf.keras.layers.Conv3D(filters=class_count, strides=(1,1,1), kernel_size=(3,3,3), padding='same'
                                , dilation_rate=(1,1,1)
                                , activation='softmax'
                                , name='Block10')

    @tf.function
    def call(self, x):
        
        # SE-Res Blocks
        conv1, pool1 = self.convblock1(x)
        conv2         = self.convblock2(pool1)
        
        # Dense ASPP
        conv3 = self.convblock3(conv2)
        conv3_op = tf.concat([conv2, conv3], axis=-1)
        
        conv4 = self.convblock4(conv3_op)
        conv4_op = tf.concat([conv3_op, conv4], axis=-1)
        
        conv5 = self.convblock5(conv4_op)
        conv5_op = tf.concat([conv4_op, conv5], axis=-1)
        
        conv6 = self.convblock6(conv5_op)
        conv6_op = tf.concat([conv5_op, conv6], axis=-1)
        
        conv7 = self.convblock7(conv6_op)
        
        # Upstream
        conv8 = self.convblock8(tf.concat([pool1, conv7], axis=-1))
        
        up9 = self.upconvblock9(conv8)
        conv9 = self.convblock9(tf.concat([conv1, up9], axis=-1))
        
        # Final
        conv10 = self.convblock10(conv9)

        if self.verbose:
            print (' ---------- Model: ', self.name)
            print (' - x: ', x.shape)
            print (' - conv1: ', conv1.shape)
            print (' - conv2: ', conv2.shape)
            print (' - conv3_op: ', conv3_op.shape)
            print (' - conv4_op: ', conv4_op.shape)
            print (' - conv5_op: ', conv5_op.shape)
            print (' - conv6_op: ', conv6_op.shape)
            print (' - conv7: ', conv7.shape)
            print (' - conv8: ', conv8.shape)
            print (' - conv9: ', conv9.shape)
            print (' - conv10: ', conv10.shape)


        return conv10

class ModelFocusNetFlipOut(tf.keras.Model):

    def __init__(self, class_count, trainable=False, verbose=False):
        super(ModelFocusNetFlipOut, self).__init__(name='ModelFocusNetFlipOut')

        # Step 0 - Init
        self.verbose = verbose
        
        filters  = [[16,16], [32,32]]
        dilation_xy = [1, 2, 3, 6, 12, 18]
        dilation_z  = [1, 1, 1, 1, 1 , 1]
        
        # Se-Res Blocks
        self.convblock1 = ConvBlock3DSERes(filters=filters[0], kernel_size=(3,3,1), dilation_rate=(dilation_xy[0], dilation_xy[0], dilation_z[0]), trainable=trainable, pool=True , squeeze_ratio=2, name='Block1') # Dim/2 (e.g. 96/2=48, 240/2=120)(rp=(3,5,10),(1,1,2))
        self.convblock2 = ConvBlock3DSERes(filters=filters[0]                     , dilation_rate=(dilation_xy[1], dilation_xy[1], dilation_z[1]), trainable=trainable, pool=False, squeeze_ratio=2, name='Block2') # Dim/2 (e.g. 96/2=48, 240/2=120)(rp=(14,18) ,(4,6))

        # Dense ASPP
        self.convblock3 = ConvBlock3DFlipOut(filters=filters[1], dilation_rate=(dilation_xy[2], dilation_xy[2], dilation_z[2]), trainable=trainable, pool=False, name='Block3_ASPP') # Dim/2 (e.g. 96/2=48, 240/2=120) (rp=(24,30),(16,18))
        self.convblock4 = ConvBlock3DFlipOut(filters=filters[1], dilation_rate=(dilation_xy[3], dilation_xy[3], dilation_z[3]), trainable=trainable, pool=False, name='Block4_ASPP') # Dim/2 (e.g. 96/2=48, 240/2=120) (rp=(42,54),(20,22))
        self.convblock5 = ConvBlock3DFlipOut(filters=filters[1], dilation_rate=(dilation_xy[4], dilation_xy[4], dilation_z[4]), trainable=trainable, pool=False, name='Block5_ASPP') # Dim/2 (e.g. 96/2=48, 240/2=120) (rp=(78,102),(24,26))
        self.convblock6 = ConvBlock3DFlipOut(filters=filters[1], dilation_rate=(dilation_xy[5], dilation_xy[5], dilation_z[5]), trainable=trainable, pool=False, name='Block6_ASPP') # Dim/2 (e.g. 96/2=48, 240/2=120) (rp=(138,176),(28,30))
        self.convblock7 = ConvBlock3DSEResFlipOut(filters=filters[1], dilation_rate=(dilation_xy[0], dilation_xy[0], dilation_z[0]), trainable=trainable, pool=False, squeeze_ratio=2, init=True, name='Block7') # Dim/2 (e.g. 96/2=48) (rp=(176,180),(32,34))

        # Upstream
        self.convblock8 = ConvBlock3DSERes(filters=filters[1], dilation_rate=(dilation_xy[0], dilation_xy[0], dilation_z[0]), trainable=trainable, pool=False, squeeze_ratio=2, init=True, name='Block8') # Dim/2 (e.g. 96/2=48)

        self.upconvblock9 = UpConvBlock3D(filters=filters[0][0], trainable=trainable, name='Block9_1') # Dim/1 (e.g. 96/1 = 96)
        self.convblock9 = ConvBlock3DSERes(filters=filters[0], dilation_rate=(dilation_xy[0], dilation_xy[0], dilation_z[0]), trainable=trainable, pool=False, squeeze_ratio=2, init=True, name='Block9') # Dim/1 (e.g. 96/1 = 96)
        
        self.convblock10 = tf.keras.layers.Conv3D(filters=class_count, strides=(1,1,1), kernel_size=(3,3,3), padding='same'
                                , dilation_rate=(1,1,1)
                                , activation='softmax'
                                , name='Block10')

    # @tf.function (cant call model.losses if this is enabled)
    def call(self, x):
        
        # SE-Res Blocks
        conv1, pool1 = self.convblock1(x)
        conv2        = self.convblock2(pool1)
        
        # Dense ASPP
        conv3 = self.convblock3(conv2)
        conv3_op = tf.concat([conv2, conv3], axis=-1)
        
        conv4 = self.convblock4(conv3_op)
        conv4_op = tf.concat([conv3_op, conv4], axis=-1)
        
        conv5 = self.convblock5(conv4_op)
        conv5_op = tf.concat([conv4_op, conv5], axis=-1)
        
        conv6 = self.convblock6(conv5_op)
        conv6_op = tf.concat([conv5_op, conv6], axis=-1)
        
        conv7 = self.convblock7(conv6_op)
        
        # Upstream
        conv8 = self.convblock8(tf.concat([pool1, conv7], axis=-1))
        
        up9 = self.upconvblock9(conv8)
        conv9 = self.convblock9(tf.concat([conv1, up9], axis=-1))
        
        # Final
        conv10 = self.convblock10(conv9)

        if self.verbose:
            print (' ---------- Model: ', self.name)
            print (' - x: ', x.shape)
            print (' - conv1: ', conv1.shape)
            print (' - conv2: ', conv2.shape)
            print (' - conv3_op: ', conv3_op.shape)
            print (' - conv4_op: ', conv4_op.shape)
            print (' - conv5_op: ', conv5_op.shape)
            print (' - conv6_op: ', conv6_op.shape)
            print (' - conv7: ', conv7.shape)
            print (' - conv8: ', conv8.shape)
            print (' - conv9: ', conv9.shape)
            print (' - conv10: ', conv10.shape)


        return conv10

############################################################
#                            MAIN                          #
############################################################

if __name__ == "__main__":
    
    X         = tf.random.normal((2,140,140,40,1))

    print ('\n ------------------- ModelFocusNetDropOut ------------------- ')
    model     = ModelFocusNetDropOut(class_count=10, trainable=True)
    y_predict = model(X, training=True)
    model.summary()

    print ('\n ------------------- ModelFocusNetFlipOut ------------------- ')
    model     = ModelFocusNetFlipOut(class_count=10, trainable=True)
    y_predict = model(X, training=True)
    model.summary()