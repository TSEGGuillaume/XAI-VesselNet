import numpy as np

def distance(p1:tuple, p2:tuple, norm:str = "L2"):
    """
    Compute the distance between two 3D points.
    Implemented distance methods :
        - "L2" : Euclidean distance 

    Parameters
        p1 (tuple)  : The first point
        p2 (tuple)  : The second point
        norm (str)  : The key of the distance method. See Implemented distance methods.

    Returns
        The distance according to the specified method (float).
    """
    import math

    if norm == "L2":
        return math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2 + (p1[2]-p2[2])**2)

def compute_iou(ground_truth, prediction, smooth=0.00001):
    """
    Compute the Intersection over Union coefficient.

    Parameters
        ground_truth    : The image of the ground truth.
        prediction      : The image of the prediction.
        smooth          : The smoothing coefficient.

    Returns
        The IoU (float).
    """
    ground_truth = ground_truth.flatten()
    prediction   = prediction.flatten()
    
    intersection = np.sum(ground_truth * prediction)
    return (intersection + smooth) / (np.sum(ground_truth) + np.sum(prediction) - intersection + smooth)

def compute_dice(ground_truth, prediction, smooth=0.00001):
    """
    Compute the Dice coefficient.

    Parameters
        ground_truth    : The image of the ground truth.
        prediction      : The image of the prediction.
        smooth          : The smoothing coefficient.

    Returns
        The Dice coefficient (float).
    """
    ground_truth = ground_truth.flatten()
    prediction   = prediction.flatten()
    
    intersection = np.sum(ground_truth * prediction)
    return (2. * intersection + smooth) / (np.sum(ground_truth) + np.sum(prediction) + smooth)