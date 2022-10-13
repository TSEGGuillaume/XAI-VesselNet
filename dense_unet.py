import os
import copy
import argparse
import time
import logging as log

import tensorflow as tf
import numpy as np
import nibabel as nib

import dense_3d_unet
import preprocessing.preprocess_image as preprocess_image
import loss_metrics
import inout.images as io

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v', action='count', default=0)
    args = parser.parse_args()
    
    return args

def main():
    # Working variable
    workspace = os.getcwd()
    weights_dir = os.path.join(workspace, "resources/weights/")
    data_dir = os.path.join(workspace, "resources/data/")
    result_dir = os.path.join(workspace, "results/")
    
    weights_name = "DenseUNet_Jerman_25625680.h5"
    data_name = "ircad3d_test.nii"
    
    # Manage the model
    model = dense_3d_unet.create_3d_dense_unet()
    model.compile(optimizer='adam', loss = loss_metrics.dice_coefficient_loss, metrics = [loss_metrics.dice_coefficient])
    log.info("Model {} created and compiled.".format(model.name))
    
    # Manage weights
    weights_path = os.path.join(weights_dir, weights_name)
    model.load_weights(weights_path)
    log.info("Weights checkpoint ({}) loaded.".format(weights_path))
    
    # Manage data
    inference_data_path = os.path.join(data_dir, data_name)
    
    inference_volume = nib.load(inference_data_path)
    log.info("Data {} loaded.".format(inference_data_path))
    inference_volume_cpy = copy.deepcopy(inference_volume)
    
    inference_volume = inference_volume.get_fdata()
    
    # npad is a tuple of (n_before, n_after) for each dimension
    x = inference_volume.shape[0]
    y = inference_volume.shape[1]
    z = inference_volume.shape[2]
    
    inference_volume = preprocess_image.resize_image(inference_volume, (256, 256, 80)) # Preprocess the volume

    temp_dir = os.path.join(data_dir, "_temp/")
    if os.path.isdir(temp_dir):
        resized_img_path = temp_dir
    else:
        log.debug("Create a temporary directory at {}".format(temp_dir))
        os.mkdir(temp_dir)
        resized_img_path = temp_dir
    
    nii_resized_img = nib.Nifti1Image(inference_volume, inference_volume_cpy.affine)
    resized_img_path = os.path.join(resized_img_path, "resized_" + data_name)
    nib.save(nii_resized_img, resized_img_path)
    log.debug("Resized data saved at {}.".format(resized_img_path))
    
    test_single_vol = io.load_image(resized_img_path)
    log.info("Start prediction on {}{} volume...".format(type(test_single_vol), test_single_vol.shape))
    start_chrono = time.time()
    pred_seg = model.predict(np.expand_dims(test_single_vol[:],0))[0]
    stop_chrono = time.time() - start_chrono
    log.info("Done. Prediction time {}".format(stop_chrono))
    
    res_nifti_savepath = os.path.join(result_dir, "res_" + data_name)

    # to save this 3D (ndarray) numpy use this
    nii_img = nib.Nifti1Image(pred_seg,  inference_volume_cpy.affine)
    nib.save(nii_img, res_nifti_savepath)
    log.info("Inference saved at {}".format(res_nifti_savepath))
    
if __name__=="__main__":
    args = parse_arguments()
    log.basicConfig(format="%(levelname)s: %(message)s", level = args.verbose * 10 if args.verbose > 0 and args.verbose <= 5 else 4 * 10) # log level correspondant au nombre de 'v' contenu dans l'arg s'il est spécifié (dans la limite de 5) sinon level error 

    log.debug("Tensorflow version : {}".format(tf.__version__))
    log.debug("Keras version : {}".format(tf.keras.__version__))
    log.debug("Num GPUs Available: {}".format(len(tf.config.list_physical_devices('GPU'))))
    
    main()
    