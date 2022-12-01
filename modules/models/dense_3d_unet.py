import logging as log

import tensorflow as tf
from tensorflow.keras import Model, Input
from tensorflow.keras.layers import Layer, BatchNormalization, concatenate, Conv3D, MaxPooling3D, UpSampling3D

class ConvBlock(Model):
    def __init__(self, filters:list, kernel_size=(3, 3, 3), bn_momentums:list=None):
        super(ConvBlock, self).__init__()
        
        filter_1, filter_2 = filters
        self.conv_1 = Conv3D(filter_1, kernel_size, activation='relu', padding='same')
        self.conv_2 = Conv3D(filter_2, kernel_size, activation='relu', padding='same')

        if bn_momentums is not None : # Is it equivalent to `training=False` as arg ? 
            bn_momentum_1, bn_momentum_2 = bn_momentums   
            self.bn_1 = BatchNormalization(momentum=bn_momentum_1)
            self.bn_2 = BatchNormalization(momentum=bn_momentum_2)
        else:
            self.bn_1 = None
            self.bn_2 = None

    def call(self, input):
        x = self.conv_1(input)
        if self.bn_1 is not None: x = self.bn_1(x)
        x = concatenate([input, x], axis=-1)
        x = self.conv_2(x)
        if self.bn_2 is not None: x = self.bn_2(x)
        x = concatenate([input, x], axis=-1)

        return x

# TODO : Write a class implementing Model  
def DenseUnet3D():
    from tensorflow.keras import Input
    from tensorflow.keras.models import Model

    # Parameters of the model
    n_filters = 4
    depth = 4
    ## Batch normalization settings
    bn_momentum = 0.99

    inputs = Input((None, None, None, 1)) # Input layer

    x = inputs

    encoder_convs = []
    for i in range(depth):
        x = ConvBlock(
                filters=[n_filters * 2 ** i, n_filters * 2 ** i],
                bn_momentums=[bn_momentum, bn_momentum]
            )(x)
        encoder_convs.append(x)

        x = MaxPooling3D(pool_size=(2, 2, 1))(x)

    x = ConvBlock(
            filters=[n_filters * 2 ** depth, n_filters * 2 ** depth],
            bn_momentums=[bn_momentum, bn_momentum]
        )(x)

    for i in reversed(range(0, depth)): # Equ. range(depth-1, -1, -1)
        x = UpSampling3D(size=(2, 2, 1))(x)
        x = concatenate([x, encoder_convs[i]], axis=4)
        x = ConvBlock(
            filters=[n_filters * 2 ** i, n_filters * 2 ** i],
            bn_momentums=[bn_momentum, bn_momentum]
        )(x)
    
    x = Conv3D(1, (1, 1, 1), activation='sigmoid')(x) # Output Layer

    model = Model(inputs=[inputs], outputs=[x], name="3D_Dense_U-Net")
    log.debug(model.summary())

    return model