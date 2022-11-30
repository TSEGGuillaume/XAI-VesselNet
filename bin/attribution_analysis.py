import argparse
import logging as log

import os

import numpy as np
import nibabel as nib
from scipy.stats import entropy, skew
import json

from graph import CNode
import configuration as appcfg
import utils.oob_prevention as oob_prevention
import utils.metrics as metrics
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

def compute_first_order_moments(data, dist_prob):
    """
    Compute statistical moments.

    Implemented moments are :
        - sum*      : the sum of the distribution.
        - range*    : the range, as tuple (min, max), value of the distribution.
        - nonzero*  : the count of non-zero, as tuple (count_negative, count_positive), values in the distribution.
        - mean      : the mean of the distribution.
        - median    : the median of the distribution.
        - std_dev   : the standard deviation of the distribution.
        - skewness  : the skewness of the distribtion. This is usefull to measure the symmetry of the distribution.
        - entropy   : the entropy (base=2) of the distribution. This is usefull to measure the "disorder" of the distribution.

    Parameters
        data : The distribution.

    Notes : 
        1. "Moments" marked by the asterisk (*) are not really statistical moments per se, but these indicators can be useful.
    
    Returns
        moments (dict) : The dictionnary of statistical moments.
    """

    flat_data = data.flatten() # Be sure that the distribution is 1D

    moments = {
        "sum"       : np.sum(flat_data),
        "range"     : (np.min(flat_data), np.max(flat_data)),
        "nonzero"   : (np.sum(flat_data < 0), np.sum(flat_data > 0)),
        "mean"      : np.mean(flat_data),
        "median"    : np.median(flat_data),
        "std_dev"   : np.std(flat_data),
        "skewness"  : skew(flat_data),
        "entropy"   : entropy(dist_prob, base=2)
    }
    # NB : All values of the distribution should be >=0 to compute entropy. As attribution map could have negative contribution, we perform data translation before compute the entropy. 
    
    log.debug("Compute statistics on window")

    return moments

def compute_performance_on_window(gt_data, pred_data, observation_point:tuple, window_shape:tuple):
    """
    Compute the performance of the model in a specific observation window.

    Parameters
        gt_data                     : The image of the ground truth.
        pred_data                   : The image of the prediction.
        observation_point (tuple)   : The center position of the observation window. The observation window will surround this position with a radius of `window_shape`.
        window_radius (tuple)       : The radii of the observed window along x, y, z axes.

    Notes : 
        1. The performance metric used is the Dice score ;
        2. The data of the ground truth and the prediction are commutable. The name of the parameters are set for "explaination" purpose.

    Returns
        dice (float) : The dice score in the observation window.
    """
    # Prepare data for comparison
    gt_data_binary      = preprocess_image.binarize(gt_data)
    pred_data_binary    = preprocess_image.binarize(pred_data)

    # Prepare the observation window
    pt_min, pt_max    = oob_prevention.side_management(observation_point, window_shape, gt_data.shape)

    # Slice the data around the observation window
    gt_sub     = gt_data_binary[pt_min[0]:pt_max[0]:1, pt_min[1]:pt_max[1]:1, pt_min[2]:pt_max[2]:1]
    pred_sub   = pred_data_binary[pt_min[0]:pt_max[0]:1, pt_min[1]:pt_max[1]:1, pt_min[2]:pt_max[2]:1]
     
    dice = metrics.compute_dice(gt_sub, pred_sub)
    log.debug("Dice : {}".format(dice))

    return dice

def compute_statistics_on_windows(attribution_map, observation_point:tuple, window_shape:tuple, stride:tuple=None):
    """
    Compute statistical indicators in the subsets of the attribution map.

    Implemented statistics :
        - sum*              : the sum of the distribution.
        - range*            : the range, given by (min, max) tuple, value of the distribution.
        - nonzero*          : the number of non-zero values in the distribution.
        - mean*             : the mean of the distribution.
        - median*           : the median of the distribution.
        - std_dev*          : the standard deviation of the distribution.
        - skewness*         : the skewness of the distribtion. This is usefull to measure the symmetry of the distribution.
        - entropy*          : the entropy (base=2) of the distribution. This is usefull to measure the "disorder" of the distribution.
        - distribution      : the distribution = main_window.flatten().
        - norm              : the Euclidean distance from the observed position.
        - histogram (dict)  : the distogram of the distribution. Access the histogram by "x" and "y" index.

    \* see fn compute_first_order_moments

    Parameters
        attribution_map             : The attribution map.
        observation_point (tuple)   : The center position of the observation window. The observation window will surround this position with a radius of `window_shape`.
        window_radius (tuple)       : The radii of the observed window along x, y, z axes.
        stride (tuple)              : The stride for the sliding of the observation window. Default is `None`.

    Notes : 
        1. If stride is set to `None`, the windows will not overlap during the sliding process.
        2. The first column of the returned matrix corresponds to the observation point window. The other columns are produced by the sliding window process and may overlap the observation window.
    
    Returns
        main_statistics (dict), sliding_statistics (list) : The statistics about the main window and the statistics about sliding windows
    """
    # Variables
    histogram_bins = 25

    # Compute statistical moments within the main window
    min_pos, max_pos = oob_prevention.side_management(observation_point, window_shape, attribution_map.shape)
    main_window = attribution_map[min_pos[0]:max_pos[0]:1, min_pos[1]:max_pos[1]:1, min_pos[2]:max_pos[2]:1]

    hist = np.histogram(main_window, bins=histogram_bins)

    indicators = compute_first_order_moments(main_window, hist[0]/np.sum(hist[0]))
    indicators["distribution"] = main_window.flatten()
    indicators["norm"] = 0 # Euclidean distance from the observed position
    indicators["histogram"] = {
            "x": hist[1],
            "y": hist[0]
    }
    main_statistics = indicators

    # Sliding window process
    # Scanning the image by sliding window. Compute indicators within all image subsets.
    sliding_statistics = []

    if stride == None:
        stride = (2 * window_shape[0], 2 * window_shape[1], 2 * window_shape[2])

    for j in range(window_shape[0], attribution_map.shape[0]-window_shape[0], stride[0]):
        for i in range(window_shape[1], attribution_map.shape[1]-window_shape[1], stride[1]):
            for k in range(window_shape[2], attribution_map.shape[2]-window_shape[2], stride[2]):

                sliding_window = attribution_map[j-window_shape[0]:j+window_shape[0], i-window_shape[1]:i+window_shape[1], k-window_shape[2]:k+window_shape[2]]

                hist = np.histogram(sliding_window, bins=histogram_bins)

                indicators = compute_first_order_moments(sliding_window, hist[0]/np.sum(hist[0]))
                indicators["distribution"] = sliding_window.flatten()
                indicators["norm"] = metrics.distance(observation_point, (j, i, k)) # Euclidean distance from the observed position
                indicators["histogram"] = {
                        "x": hist[1],
                        "y": hist[0]
                }
                sliding_statistics.append(indicators)

    return main_statistics, sliding_statistics

def compute_vessel_thickness(gt_data, observation_point:tuple, affine=None):
    """
    Compute the vessel thickness (by Exact Euclidean Distance Transform) at a specified position.

    Parameters
        gt_data                     : The image of the ground truth.
        observation_point (tuple)   : The center position of the observation window. The observation window will surround this position with a radius of `window_shape`.
        affine                      : The orientation of the medical image. Optional (default `None`)

    Returns
        vessel_thickness (float) : The thickness of the vessel at the specified position.
    """
    dist_map = preprocess_image.distance_map(gt_data, method="edt")

    vessel_thickness = dist_map[observation_point]
    vessel_thickness = vessel_thickness * 2 # EEDT gives the minimal radius to the background

    log.debug("Vessel thickness : {} (vx)".format(vessel_thickness))

    if affine is not None:
        simulated_pos =  tuple([vessel_thickness for n in range(len(observation_point))])  # TODO : change this part if the thickness changes according to the direction
        fake_affine = affine * np.eye(4) # We set the "translation part" of the matrix to 0 because this is not suitable for thickness computation.

        vessel_thickness = tuple([abs(elem) for elem in coordinate_systems.image_to_anatomical(simulated_pos, fake_affine)]) # Returns a vector of thickness along x,y,z axis. Absolute values because speaking about thickness. Direction doesn't matter

    return vessel_thickness

def serialize_results(main_wnd_stats:dict, sliding_wnds_stats:list, dice:float, vessel_thickness:float, obs_pt):
    """
    Save computed results into files.

    Parameters
        main_wnd_stats (dict)               : The statistics for the main window
        sliding_wnds_stats (list(dict))     : The list of statistics computed on each sliding window.
        dice (float)                        : The dice score inside the main window
        vessel_thickness (float)            : The vessel thickness computed at the observation point
    """
    # Process results of every windows (main and sliding windows)
    dump_stats = []
    for stat in ([main_wnd_stats] + sliding_wnds_stats): # Insert stats of the main window in front of the list of sliding windows stats

        dump_stats.append([
            stat["sum"],
            stat["range"][0], stat["range"][1],
            stat["nonzero"][0], stat["nonzero"][1],
            stat["mean"],
            stat["median"],
            stat["std_dev"],
            stat["skewness"],
            stat["entropy"],
            stat["norm"]
        ])

    # Saving data on the disk
    save_path = os.path.join(cfg.result_dir, contribution_map_name.split(".")[0] + '.csv')
    np.savetxt(save_path, np.array(dump_stats), fmt='%.18e', delimiter=';') # Save distributions of all statistical indicators
    log.info("Global results saved : {}".format(save_path))

    # Convert data about the main windows into JSON
    # Warning : json serialisation does not support all data type - TODO: https://bobbyhadz.com/blog/python-typeerror-object-of-type-int64-is-not-json-serializable
    main_wnd_stats["nonzero"]            = [int(val) for val in main_wnd_stats["nonzero"]] # Cast for JSON seralisation
    main_wnd_stats["dice"]               = dice
    main_wnd_stats["vessel_thickness"]   = vessel_thickness
    main_wnd_stats["degree"]             = obs_pt.degree
    main_wnd_stats["histogram"]["x"]     = main_wnd_stats["histogram"]["x"].tolist() # Numpy array to list for JSON seralisation
    main_wnd_stats["histogram"]["y"]     = main_wnd_stats["histogram"]["y"].tolist() # Numpy array to list for JSON seralisation
    main_wnd_stats["distribution"]       = main_wnd_stats["distribution"].tolist() # Numpy array to list for JSON seralisation

    json_stats_main_wnd = json.dumps(main_wnd_stats)

    save_path = os.path.join(cfg.result_dir, contribution_map_name.split(".")[0] + '_stat_mw.json')
    with open(save_path, "w") as jsonFile:
        jsonFile.write(json_stats_main_wnd)
    log.info("Main window results saved : {}".format(save_path))

def main():
    ### Loading the data
    contribution_map = nib.load(os.path.join(cfg.result_dir, contribution_map_name))

    gt = nib.load(os.path.join(cfg.data_dir, gt_name)).get_fdata() #  Ground-truth
    pred = np.squeeze(nib.load(os.path.join(cfg.result_dir, prediction_name)).get_fdata(), -1)

    ### Graph part
    graph_file = os.path.join(cfg.data_dir, "graph.vvg")
    vessel_graph = process_voreen_data.parse_voreen_VesselGraphSave_file(graph_file)
    
    vessel_graph = coordinate_systems.anatomical_graph_to_image_graph(vessel_graph, contribution_map.affine)

    ### Logit selection --> Point to explain (type CNode even for skeleton points)
    if args.node is not None:
        obs_pt = vessel_graph.nodes[int(args.node)]
        log.debug(obs_pt)

    elif args.centerline is not None:
        centerline = vessel_graph.connections[int(args.centerline)]
        obs_pt = centerline.getMidPoint()
        log.debug("_{}_ |{}<->{}| - Skeleton voxel : {}".format(centerline._id, centerline.node1._id, centerline.node2._id, obs_pt.pos))

    elif args.position is not None:
        obs_pt = CNode("{}-{}-{}".format(args.position[0], args.position[1], args.position[2]), tuple(args.position), -1)
        log.debug("Raw position : {}".format(obs_pt.pos))

    window_shape = (6, 6, 4) # z has lower dimension because of the Z-axis downscale. To sum up, we will see (6+1+6)^2*(4+1+4)=1521 vx
    analysis_stride = (window_shape[0],  window_shape[1], window_shape[2])

    dice = compute_performance_on_window(gt, pred, obs_pt.pos, window_shape)
    stat_main_wnd, stats_sliding_wnd = compute_statistics_on_windows(contribution_map.get_fdata(), obs_pt.pos, window_shape, stride=analysis_stride)
    vessel_thickness = compute_vessel_thickness(gt, obs_pt.pos)

    # Saving results into files
    serialize_results(stat_main_wnd, stats_sliding_wnd, dice, vessel_thickness, obs_pt)

if __name__=="__main__":  
    cfg = appcfg.CConfiguration(p_filename="default.ini")

    args = parse_arguments()

    log.basicConfig(format="%(levelname)s: %(message)s", level = args.verbose * 10 if args.verbose > 0 and args.verbose <= 5 else 4 * 10) # log level correspondant au nombre de 'v' contenu dans l'arg s'il est spécifié (dans la limite de 5) sinon level error 

    if bool(cfg.debug) == True:
        log.basicConfig(format="%(levelname)s: %(message)s", level = log.DEBUG)
        data_name = "ircad3d_test.nii" # TODO : add to cfg
        gt_name = "liver_vessel_mask_iso_resized.nii" # TODO : add to cfg

    if args.node is not None:
        logit_type = "node"
        logit_id = args.node

    elif args.centerline is not None:
        logit_type = "skvx"
        logit_id = args.centerline

    elif args.position is not None:
        logit_type = "pos"
        logit_id = "{}-{}-{}".format(args.position[0], args.position[1], args.position[2])

    prediction_name = "res_{}".format(data_name)
    contribution_map_name = "ig_{}_{}_w664_{}".format(logit_type, logit_id, data_name)

    main()