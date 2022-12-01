import os
import argparse
import logging as log
import datetime

import tensorflow as tf

import configuration as appcfg
import utils.metrics as custometrics
import models.dense_3d_unet as dense_3d_unet
import models.datagenerator as datagenerator
import models.losses as customlosses  

def parse_arguments():
    parser = argparse.ArgumentParser()
    #parser.add_argument('data', type=str, metavar=("DATA_NAME"), help="Path to the data to infer") # TODO : access the dataset dynamically
    parser.add_argument('model', choices=['dense_unet', 'multires_unet'], default='dense_unet')
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()
    
    return args

def train(model, training_set, validation_set):
    from tensorflow.keras.optimizers import Adam
    from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard

    adam = Adam(learning_rate = 0.0001) # Optimizer
    
    model.compile(optimizer=adam, loss = customlosses.CustomDiceLoss(), metrics = [custometrics.compute_dice])
    
    weight_path = os.path.join(cfg.result_dir, "weights", model.name + ".h5") # TODO encapusler le type d'image (raw, jerman, ...), la taille du resize, la date, la loss

    checkpoint = ModelCheckpoint(weight_path,
                                 monitor='val_loss',
                                 verbose=1, 
                                 save_best_only=True, 
                                 mode='min', 
                                 save_weights_only = True
                                )

    tensorboard_callback = TensorBoard(
        log_dir=os.path.join(cfg.log_dir, "fit", datetime.datetime.now().strftime("%Y%m%d-%H%M%S")),
        histogram_freq=1
    )

    reduceLROnPlat = ReduceLROnPlateau(monitor='val_loss', 
                                       factor=0.1,
                                       patience=5
                                    )

    monitor_early_stopping = EarlyStopping(monitor='val_loss',
                                           mode='min',
                                           patience=10
                                        )

    callbacks_list = [checkpoint, tensorboard_callback]

    history = model.fit(training_set,
                      batch_size=1,
                      epochs=100,
                      validation_data = validation_set, 
                      callbacks = callbacks_list    # ,use_multiprocessing=True
                    )

def main():
    if args.model == 'dense_unet':
        model = dense_3d_unet.create_3d_dense_unet()
    else:
        raise ValueError("Other model not supported yet")

    # TODO : settings spécifiques aux apprentissages
    # Training : PATIENT 1-16
    # Evaluation : PATIENT 17-19
    # Testing : 20
    train_data_path         = "D:/Workspace/Dataset/dataset/data/jerman/train"
    train_mask_path         = "D:/Workspace/Dataset/dataset/mask/train"
    validation_data_path    = "D:/Workspace/Dataset/dataset/data/jerman/evaluation"
    validation_mask_path    = "D:/Workspace/Dataset/dataset/mask/evaluation"

    train_gen = datagenerator.Custom3dIrcadbGen(train_data_path, train_mask_path) 
    valid_gen = datagenerator.Custom3dIrcadbGen(validation_data_path, validation_mask_path)

    train(model, train_gen, valid_gen)

if __name__=="__main__":
    log.debug("Tensorflow version : {}".format(tf.__version__))
    log.debug("Keras version : {}".format(tf.keras.__version__))
    
    # Disable GPU, data too large for VRAM
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    
    cfg = appcfg.CConfiguration(p_filename="../default.ini")

    args = parse_arguments()

    log.basicConfig(format="%(levelname)s: %(message)s", level = args.verbose * 10 if args.verbose > 0 and args.verbose <= 5 else 4 * 10) # log level correspondant au nombre de 'v' contenu dans l'arg s'il est spécifié (dans la limite de 5) sinon level error 

    main()
    