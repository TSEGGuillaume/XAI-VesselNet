import os
import argparse
import time
import logging as log

import tensorflow as tf
import numpy as np
import nibabel as nib

import configuration as appcfg
import models.dense_3d_unet as dense_3d_unet
import preprocessing.preprocess_image as preprocess_image


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', type=str, metavar=("DATA_NAME"), help="Path to the data to infer")
    parser.add_argument('weights', type=str, metavar=("WEIGHTS_NAME"), help="Name of the 3D Dense U-Net weigths") # Replace this by a model with weights
    #parser.add_argument('model', choices=['dense_unet', 'multires_unet'])
    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()
    
    return args

def main():   
    # Create the model
    model = dense_3d_unet.create_3d_dense_unet()
    log.info("Model {} created and compiled.".format(model.name))
    
    # Manage weights
    weights_path = os.path.join(cfg.weights_dir, args.weights)
    model.load_weights(weights_path)
    log.info("Weights checkpoint ({}) loaded.".format(weights_path))
    
    # Manage data
    inference_data_path = os.path.join(cfg.data_dir, args.data)
    
    inference_data = nib.load(inference_data_path)
    data_affine = inference_data.affine
    log.info("Data {} loaded.".format(inference_data_path))

    inference_data = preprocess_image.resize_image(inference_data.get_fdata(), (256, 256, 80)) # Preprocess the volume by resizing

    # Save the rescale data
    temp_dir = os.path.join(cfg.data_dir, "_temp/")
    if os.path.isdir(temp_dir):
        resized_img_path = temp_dir
    else:
        log.debug("Create a temporary directory at {}".format(temp_dir))
        os.mkdir(temp_dir)
        resized_img_path = temp_dir
    
    nib.save(
        nib.Nifti1Image(inference_data, data_affine),
        os.path.join(resized_img_path, "resized_" + args.data)
        )
    log.debug("Resized data saved at {}.".format(resized_img_path))
    
    # Prediction
    log.info("Start prediction on {}{} volume...".format(type(inference_data), inference_data.shape))
    start_chrono = time.time()
    res = model.predict(np.expand_dims(inference_data,0))[0]
    stop_chrono = time.time() - start_chrono
    log.info("Done. Prediction time {}".format(stop_chrono))
    
    # Save the result
    res_nifti_savepath = os.path.join(cfg.result_dir, "res_" + args.data)
    nib.save(
        nib.Nifti1Image(res, data_affine),
        res_nifti_savepath
        )
    log.info("Inference saved at {}".format(res_nifti_savepath))
    
if __name__=="__main__":
    log.debug("Tensorflow version : {}".format(tf.__version__))
    log.debug("Keras version : {}".format(tf.keras.__version__))
    log.debug("Num GPUs Available: {}".format(len(tf.config.list_physical_devices('GPU'))))
    
    cfg = appcfg.CConfiguration(p_filename="../default.ini")

    args = parse_arguments()

    log.basicConfig(format="%(levelname)s: %(message)s", level = args.verbose * 10 if args.verbose > 0 and args.verbose <= 5 else 4 * 10) # log level correspondant au nombre de 'v' contenu dans l'arg s'il est spécifié (dans la limite de 5) sinon level error 

    main()
    