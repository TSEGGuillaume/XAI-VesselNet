import argparse

import os
import logging as log

import numpy as np
import tensorflow as tf
import nibabel as nib

import models.dense_3d_unet as dense_3d_unet
from graph import CNode
import configuration as appcfg
import utils.oob_prevention as oob_prevention
import preprocessing.preprocess_image as preprocess_image
import preprocessing.coordinate_systems as coordinate_systems
import preprocessing.process_voreen_data as process_voreen_data

def parse_arguments():
    parser = argparse.ArgumentParser()

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--node', '-n', type=int, metavar=("ID_NODE"), help="id of the node to inspect", default=None)
    group.add_argument('--centerline', '-c', type=int, metavar=("ID_CENTERLINE"), help="id of the centerline to inspect", default=None)
    group.add_argument('--position', '-p', nargs=3, type=int, metavar=("X", "Y", "Z"), help="image coordinates (x,y,z) of the voxel to inspect", default=None)

    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()

    return args

def interpolate_images(baseline, image, alphas):
    """
    Interpolate the image along a linear path baseline->image so that the information appears gradually.
    This function is used by Integrated Gradient to progressively show the information carried by the pixels.

    Parameters
        baseline    : The baseline image.
        images      : The image to interpolate along the linear path. 
        alphas      : The coefficients of interpolation.

    Returns
        images (list) : The list of interpolated images, order by their "fading" coefficient.
    """
    alphas_x = tf.cast(alphas[:, tf.newaxis, tf.newaxis, tf.newaxis, tf.newaxis], dtype="float64") # Adds axis [:, height, weight, depth, color]
    baseline_x = tf.expand_dims(baseline, axis=0) # Add the batch channel according to the inputs required by TensorFlow.
    input_x = tf.expand_dims(image, axis=0) # Add the batch channel according to the inputs required by TensorFlow.
    delta = input_x - baseline_x
    images = baseline_x + alphas_x * delta

    return images

def compute_gradients(model, images, logit = None, window_radius = (0, 0, 0)):
    """
    Compute gradients of the provided logits with respect to the inputs.

    Parameters
        model                   : The model used for the explanation
        images (ndarray)        : The images used for the explanation. The shape of the array is (nbatch, nrows, ncols, depth, nchannel) 
        logits                  : Indicates which output to explain with respect to the inputs (pixels of images variables)
                                  Possible values :
                                    - None (default) : allows to explain each output with respect to the inputs into a single map
                                    - CNode : allows to explain one specific logit. Coupled with window_radius, it explains a window a specified radius around the logit.
        window_radius (tuple)   : Radii along x, y, z of the observed window. Set to 0 for all axis, the algorithm explains only the specified logit.

    Returns
        gradients (ndarray) : The gradients maps. The shape of this variable is the same than images argument (one map per image of images).
    """
    with tf.GradientTape() as tape:
        tape.watch(images)

        if logit == None:
            logits = model(images)
        else:
            if type(logit) == tuple:
                min_pos, max_pos = oob_prevention.side_management(logit, window_radius, images.shape[1:4])  # Make sure that the observation window does not extends the borders of the image. [1:4] to avoid batch and color channel dimensions checking.

                log.debug("({}, {}, {}), ({}, {}, {}) (excluded)".format(min_pos[0], min_pos[1], min_pos[2], max_pos[0], max_pos[1], max_pos[2]))

                logits = model(images)[:, min_pos[0]:max_pos[0]:1, min_pos[1]:max_pos[1]:1, min_pos[2]:max_pos[2]:1, :]
            else:
                raise NotImplementedError("Not implemented yet")
            
    gradients = tape.gradient(logits, images)

    return gradients

def integral_approximation(gradients):
    """
    Approximate the integral of gradients by trapezoidal sums (Riemann).

    Parameters
        gradients   : The gradients to integrate. 

    Returns
        integrated_gradients : The approximation of the integral.
    """ 
    grads = (gradients[:-1] + gradients[1:]) / tf.constant(2.0, dtype="float64")
    integrated_gradients = tf.math.reduce_mean(grads, axis=0)

    return integrated_gradients

def integrated_gradients(model, data, baseline=None, nsteps=50, observation_point=None, window_shape=None):
    """
    Explain a prediction by the Integrated Gradients method.

    Parameters
        model               : The prediction model.
        data                : The data to predict.
        baseline            : The baseline. Default is `None`.
        nsteps              : The number of step for the image interpolation.
        observation_point   : Default is `None`.
        window_shape        : The radius of the observation window. Default is `None`.

    Notes :
        1. The baseline must represents the absence of information. That means that the model should be insensitive to the baseline.
        2. If no baseline is given, a full black image will be created for this purpose.
        2. If no observation point is given, the IG will explain the whole image. This is not recommended to have a fine topological explanation.
        3. If no window shape is given, the IG will explain only the observation point.
        4. If an observation point is provided, a window shape must also be provided, and vice-versa.

    References :
        This code is an implementation of the method proposed by Sundararajan et al. in the Axiomatic Attribution for Deep Networks paper. See https://arxiv.org/abs/1703.01365
        This code is inspired by the implementation proposed by the Tensorflow team. See https://www.tensorflow.org/tutorials/interpretability/integrated_gradients?hl=en

    Returns
        ig_arr : The map of contributions (by Integrated Gradients).
    """ 
    if (observation_point is None and window_shape is not None) or (observation_point is not None and window_shape is None):
        raise ValueError("If an observation point is provided, a window_shape must also be provided, and vice-versa.")

    if baseline is None:
        # Creation of the baseline
        # The black image is the default baseline. Black means no information
        baseline = tf.zeros(shape=data.shape, dtype="float64")

    data        = tf.expand_dims(data, axis=-1)
    baseline    = tf.expand_dims(baseline, axis=-1)

    # Generate alpha coefficient for linear path.
    alphas = tf.linspace(start=0.0, stop=1.0, num=nsteps+1)
    interpolated_images = interpolate_images(baseline=baseline, image=data, alphas=alphas)

    # Collect gradients.
    path_gradients = []
    for k, image in enumerate(interpolated_images):
        log.info("Compute {}*I".format(k / nsteps))
        image = tf.expand_dims(image, axis=0)
        path_gradients.append(compute_gradients(model, image, observation_point, window_shape))
        
    # Concatenate path gradients together row-wise into single tensor.
    total_gradients = tf.concat(path_gradients, axis=0)

    # Integral approximation through averaging gradients.
    avg_gradients = integral_approximation(gradients=total_gradients)

    # Scale integrated gradients with respect to input.
    integrated_gradients = (data - baseline) * avg_gradients
        
    ig_arr = np.squeeze(np.array(integrated_gradients), axis=-1)

    return ig_arr

def main():    
    # Manage the model
    model = dense_3d_unet.create_3d_dense_unet()
    log.info("Model {} created and compiled.".format(model.name))

    # Manage weights
    weights_path = os.path.join(cfg.weights_dir, weights_name)
    model.load_weights(weights_path)
    log.info("Weights checkpoint ({}) loaded.".format(weights_path))
    
    # Manage data
    inference_data_path = os.path.join(cfg.data_dir, data_name)
    
    inference_volume = nib.load(inference_data_path)
    log.info("Data {} loaded.".format(inference_data_path))
    
    inference_data = inference_volume.get_fdata()

    inference_data = preprocess_image.resize_image(inference_data, (256, 256, 80)) # Preprocess the volume by resizing
    
    temp_dir = os.path.join(cfg.data_dir, "_temp/")
    if os.path.isdir(temp_dir):
        resized_img_path = temp_dir
    else:
        log.debug("Create a temporary directory at {}".format(temp_dir))
        os.mkdir(temp_dir)
        resized_img_path = temp_dir
    
    nii_resized_img = nib.Nifti1Image(inference_data, inference_volume.affine)
    resized_img_path = os.path.join(resized_img_path, "resized_" + data_name)
    nib.save(nii_resized_img, resized_img_path)
    log.debug("Resized data saved at {}.".format(resized_img_path))
    
    log.debug("Data shape : {}".format(inference_data.shape))
    
    ### Graph part
    graph_file = os.path.join(cfg.data_dir, "graph.vvg")
    vessel_graph = process_voreen_data.parse_voreen_VesselGraphSave_file(graph_file)
    
    vessel_graph = coordinate_systems.anatomical_graph_to_image_graph(vessel_graph, inference_volume.affine)
    
    ### Logit selection --> Point to explain (type CNode even for skeleton points)
    window_shape = (6, 6, 4) # z has lower dimension because of the Z-axis downscale. To sum up, we will see (6+1+6)^2*(4+1+4)=1521 vx

    if args.node is not None:
        obs_pt = vessel_graph.nodes[args.node]
        log.info(obs_pt)

        attribution_path = os.path.join(cfg.result_dir, "ig_node_{}_w{}{}{}_{}".format(obs_pt._id, window_shape[0], window_shape[1], window_shape[2], data_name))

    elif args.centerline is not None:
        centerline = vessel_graph.connections[args.centerline]
        obs_pt = centerline.getMidPoint()

        log.info("_{}_ |{}<->{}| - Skeleton voxel : {}".format(centerline._id, centerline.node1._id, centerline.node2._id, obs_pt.pos))

        attribution_path = os.path.join(cfg.result_dir, "ig_skvx_{}_w{}{}{}_{}".format(centerline._id, window_shape[0], window_shape[1], window_shape[2], data_name))

    elif args.position is not None:
        id_raw_position = "{}-{}-{}".format(args.position[0], args.position[1], args.position[2])
        obs_pt = CNode(id_raw_position, tuple(args.position), -1)
        log.debug("Raw position : {}".format(obs_pt.pos))

        attribution_path = os.path.join(cfg.result_dir, "ig_pos_{}_w{}{}{}_{}".format(id_raw_position, window_shape[0], window_shape[1], window_shape[2], data_name))

    else:
        obs_pt = None
    
    ### Integrated Gradients
    if os.path.exists(attribution_path):
        log.info("Attribution has been already computed for this sample. Only loading is required.")
        ig_array = nib.load(attribution_path).get_fdata()
    else:
        log.info("Compute attributions using Integrated Gradients.")
        ig_array = integrated_gradients(model, inference_data, observation_point=obs_pt.pos, window_shape=window_shape)
    
        # Save the result
        ig_volume = nib.Nifti1Image(ig_array, inference_volume.affine)
        nib.save(ig_volume, attribution_path)

        log.info("Integrated Gradients map has been saved at {}".format(attribution_path))

if __name__=="__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
  
    cfg = appcfg.CConfiguration(p_filename="default.ini")

    if bool(cfg.debug) == True:
        weights_name = "DenseUNet_Jerman_25625680.h5"
        data_name = "ircad3d_test.nii"

    args = parse_arguments()
    
    log.basicConfig(format="%(levelname)s: %(message)s", level = args.verbose * 10 if args.verbose > 0 and args.verbose <= 5 else 4 * 10) # log level correspondant au nombre de 'v' contenu dans l'arg s'il est spécifié (dans la limite de 5) sinon level error 

    main()
