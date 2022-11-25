import numpy as np

def check_points_validity(pt_min:tuple, pt_max:tuple, img_shape:tuple):
    """
    Check that `pt_min` is closer to the origin than `pt_max` and that each point does not go beyond image boundaries.

    Parameters
        pt_min (tuple)      : The minimal point (i.e. top-left point (2D) / top-left-front point (3D)).
        pt_max (tuple)      : The maximal point (i.e. bottom-right point (2D) / bottom-right-back point (3D)).
        img_shape (tuple)   : The image shape.
    """
    # Transform 3D coordinates to 1D index
    img_max_idx = img_shape[0] * img_shape[1] * img_shape[2]
    idx_pt_min = np.ravel_multi_index(pt_min, img_shape)
    idx_pt_max = np.ravel_multi_index(pt_max, img_shape)

    assert idx_pt_min >= 0 and idx_pt_max < img_max_idx and idx_pt_min < idx_pt_max

def side_management(pos:tuple, wnd_shape:tuple, img_shape:tuple):
    """
    Create the boundary points of a window from a given position and size. This function is used to facilitiate slicing.

    Parameters
        pos (tuple)         : The position of the window.
        wnd_shape (tuple)   : The radius of the window.
        img_shape (tuple)   : The image shape.

    Returns
        pt_min, pt_max (tuple, tuple) : The boundary points of a window.
    """
    i_min = int(max(0, pos[0] - wnd_shape[0]))
    j_min = int(max(0, pos[1] - wnd_shape[1]))
    d_min = int(max(0, pos[2] - wnd_shape[2]))
    # Keep in mind that for slicing, the max boundary is excluded (see +1)
    i_max = int(min(img_shape[0], pos[0] + wnd_shape[0] + 1))
    j_max = int(min(img_shape[1], pos[1] + wnd_shape[1] + 1))
    d_max = int(min(img_shape[2], pos[2] + wnd_shape[2] + 1))

    check_points_validity((i_min, j_min, d_min), (i_max, j_max, d_max), img_shape)

    return (i_min, j_min, d_min), (i_max, j_max, d_max)