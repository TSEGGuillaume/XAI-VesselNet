import numpy as np
import nibabel as nib

def load_image(img_path: str):
    """
    Loads a medical image and expands its dimension to match Tensorflow's expectations. The newly added dimension represents image channels.

    Parameters
        img_path (str) : Input of the layer

    Returns
        as_tf_shape : The expanded image
    """
    np_image = nib.load(img_path).get_fdata()
    print(np_image.shape)
    as_tf_shape = np.expand_dims(np_image, -1)
    
    return as_tf_shape


def read_all_slices(in_paths, rescale = False):
    cur_vol = np.expand_dims(np.concatenate([nib.load(os.path.join(in_paths, c_path)).get_fdata() for c_path in os.listdir(in_paths)], 2), -1)
    
    if rescale:
        return (cur_vol.astype(np.float32) + 500) / 2000.0 # Explication ?
    else:
        return cur_vol

    
def read_both(in_paths):
    return read_all_slices(in_paths)