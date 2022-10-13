import math

import numpy as np

def histogram_correlation(hist1, hist2):
    """
    Compute the correlation between two histograms.
    Reference : 
        https://en.wikipedia.org/wiki/Pearson_correlation_coefficient

    Parameters
        hist1  : the first normalized histogram.
        hist2  : the second normalized histogram.

    Returns
        r (float) : The correlation between the given histograms.
    """
    assert len(hist1) == len(hist2)

    # Normalize histograms by the size of the population
    hist1 = hist1 / np.sum(hist1)
    hist2 = hist2 / np.sum(hist2)

    n = len(hist1)

    denom_dist1 = math.sqrt(n * np.sum(hist1 * hist1) - np.sum(hist1) * np.sum(hist1))
    denom_dist2 = math.sqrt(n * np.sum(hist2 * hist2) - np.sum(hist2) * np.sum(hist2))
    
    r = (n * np.sum(hist1 * hist2) - np.sum(hist1) * np.sum(hist2)) / (denom_dist1 * denom_dist2)

    return r

def histogram_intersection(hist1, hist2):
    """
    Compute the intersection between two histograms.

    Parameters
        hist1  : the first normalized histogram.
        hist2  : the second normalized histogram.

    Returns
        intersec (int) : The intersection between the given histograms.
    """
    assert len(hist1) == len(hist2)

    intersec = 0
    for i in range(len(hist1)):
        intersec = intersec + min(hist1[i], hist2[i])

    return intersec

def histogram_hellinger(hist1, hist2):
    """
    Compute the Hellinger distance between two histograms.
    The Hellinger distance is closely related to the Bhattacharyya distance.
    References :
        https://en.wikipedia.org/wiki/Hellinger_distance
        https://en.wikipedia.org/wiki/Bhattacharyya_distance

    Parameters
        hist1  : the first normalized histogram.
        hist2  : the second normalized histogram.

    Returns
        dist (float) : The Hellinger distance between the given histograms.
    """
    assert len(hist1) == len(hist2)

    # Normalize histograms
    norm_hist1 = hist1 / np.sum(hist1)
    norm_hist2 = hist2 / np.sum(hist2)

    BC = np.sum(np.sqrt(norm_hist1 * norm_hist2))
    dist = math.sqrt(1-BC)

    return dist