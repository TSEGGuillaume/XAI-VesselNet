import logging as log

import tensorflow as tf
from tensorflow.keras.layers import BatchNormalization, concatenate, Conv3D, MaxPooling3D, UpSampling3D

def contracting_layer_3D(input, neurons, ba_norm, ba_norm_momentum, kernel_size=(3, 3, 3)):
    """
    Create a contracting layer

    Parameters
        input               : Input of the layer
        neurons (int)       : Number of filters of the layer
        ba_norm (bool)      : Indicate if batch normalization is required
        ba_norm_momentum    : Momentum value of the BatchNormalization layer
        kernel_size         : Size of the 3D kernel of Convolutional layers composing the contracting layer

    Returns
        pool, conc2 (tuple) : contains the final MaxPooling layer and its raw version tensor (contracting_layer before pooling)
    """
    conv1 = Conv3D(neurons, kernel_size, activation='relu', padding='same')(input)
    if ba_norm : conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)
    conc1 = concatenate([input, conv1], axis=-1)
    conv2 = Conv3D(neurons, kernel_size, activation='relu', padding='same')(conc1)
    if ba_norm : conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    conc2 = concatenate([input, conv2], axis=-1)
    pool = MaxPooling3D(pool_size=(2, 2, 1))(conc2)

    return pool, conc2

def middle_layer_3D(input, neurons, ba_norm, ba_norm_momentum, kernel_size=(3, 3, 3)):
    """
    Create the middle layer between the contracting and expanding layers

    Parameters
        input               : Input of the layer
        neurons (int)       : Number of filters of the layer
        ba_norm (bool)      : Indicate if batch normalization is required
        ba_norm_momentum    : Specified the momentum value of the BatchNormalization layer
        kernel_size         : Size of the 3D kernel of Convolutional layers composing the middle_layer

    Returns
        conc2 (tensor) : The middle layer
    """
    conv_m1 = Conv3D(neurons, kernel_size, activation='relu', padding='same')(input)
    if ba_norm : conv_m1 = BatchNormalization(momentum=ba_norm_momentum)(conv_m1)
    conc1 = concatenate([input, conv_m1], axis=-1)
    conv_m2 = Conv3D(neurons, kernel_size, activation='relu', padding='same')(conc1)
    if ba_norm : conv_m2 = BatchNormalization(momentum=ba_norm_momentum)(conv_m2)
    conc2 = concatenate([input, conv_m2], axis=-1)

    return conc2

def expanding_layer_3D(input, neurons, concatenate_link, ba_norm, ba_norm_momentum, kernel_size=(3, 3, 3)):
    """
    Create an expanding layer

    Parameters
        input               : Input of the layer
        neurons (int)       : Number of filters of the layer
        concatenate_link    : 
        ba_norm (bool)      : Indicate if batch normalization is required
        ba_norm_momentum    : Momentum value of the BatchNormalization layer
        kernel_size         : Size of the 3D kernel of Convolutional layers composing the contracting layer

    Returns
        conc2 (tensor) : The expanding layer
    """

    up = concatenate([UpSampling3D(size=(2, 2, 1))(input), concatenate_link], axis=4)
    conv1 = Conv3D(neurons, kernel_size, activation='relu', padding='same')(up)
    if ba_norm : conv1 = BatchNormalization(momentum=ba_norm_momentum)(conv1)
    conc1 = concatenate([up, conv1], axis=-1)
    conv2 = Conv3D(neurons, kernel_size, activation='relu', padding='same')(conc1)
    if ba_norm : conv2 = BatchNormalization(momentum=ba_norm_momentum)(conv2)
    conc2 = concatenate([up, conv2], axis=-1)

    return conc2

def create_3d_dense_unet():
    """
    Define the neural network
    
    Returns the model
    """
    from tensorflow.keras import Input
    from tensorflow.keras.models import Model

    # Parameters of the model
    n_filters = 4
    depth = 4
    activation = 'sigmoid'
    ## Batch normalization settings
    ba_norm = True
    ba_norm_momentum = 0.99


    inputs = Input((None, None, None, 1)) # Input layer
    
    # Start the CNN model chain with adding the inputs as first tensor
    cnn_chain = inputs

    # Cache contracting normalized conv layersfor later copy & concatenate links
    contracting_convs = []

    # Contracting Layers
    for i in range(depth):
        neurons = n_filters * 2**i
        cnn_chain, last_conv = contracting_layer_3D(cnn_chain, neurons,
                                                    ba_norm,
                                                    ba_norm_momentum)
        contracting_convs.append(last_conv)

    # Middle Layer
    neurons = n_filters * 2**depth
    cnn_chain = middle_layer_3D(cnn_chain, neurons, ba_norm,
                                ba_norm_momentum)

    # Expanding Layers
    for i in reversed(range(0, depth)):
        neurons = n_filters * 2**i
        cnn_chain = expanding_layer_3D(cnn_chain, neurons,
                                        contracting_convs[i], ba_norm,
                                        ba_norm_momentum)

  
    conv_out = Conv3D(1, (1, 1, 1), activation=activation)(cnn_chain) # Output Layer

    # Create Model with associated input and output layers
    model = Model(inputs=[inputs], outputs=[conv_out], name="3d_dense_unet")

    log.debug(model.summary())

    # Return the 3D Dense U-Net model
    return model

def main():
    model = create_3d_dense_unet()

if __name__=="__main__":
    log.basicConfig(format="%(levelname)s: %(message)s", level = log.DEBUG)

    main()