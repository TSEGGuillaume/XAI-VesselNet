import logging as log
import argparse
import os

from skimage.transform import resize
import numpy as np

import nibabel as nib

def resize_image(data, static_size: tuple):
    """
    Resizes the data to fit the specified size.

    Parameters
        data                : The data to resize
        static_size (tuple) : The desired size. Using `None` on a k-th dimension of the tuple allows not to resize the k-th dimension.

    Returns
        data_resized (numpy.array) : the resized data
    """
    new_size = list(data.shape)
        
    for i in range(len(static_size)):
        new_size[i] = new_size[i] if static_size[i] == None else static_size[i]

    log.info("Resizing data {} from {} to {}".format(type(data), data.shape, new_size))

    data_resized = resize(data, new_size, anti_aliasing=False)
    
    return np.array(data_resized)

def resize_image_min_side(data, min_size: int):
    """
    Resizes the data to fit the minimal side to the argument value. The aspect ratio of the data will be preserved.

    Parameters
        data           : The data to resize
        min_size (int) : The desired size for the minimal side of the data.

    Returns
        data_resized (numpy.array) : the resized data
    """
    new_size = tuple()

    arr_shape = np.asarray(data.shape)
    
    ratio = min_size / np.min(arr_shape)
    arr_shape = ratio * arr_shape
    
    new_size = tuple(arr_shape.astype(int))

    log.info("Resizing data {} from {} to {}".format(type(data), data.shape, new_size))
    
    data_resized = resize(data, new_size, anti_aliasing=False)
    
    return np.array(data_resized)
    
def resize_image_max_side(data, max_size: int):
    """
    Resizes the data to fit the maximal side to the argument value. The aspect ratio of the data will be preserved.

    Parameters
        data           : The data to resize
        max_side (int) : The desired size for the maximal side of the data.

    Returns
        data_resized (numpy.array) : the resized data
    """
    new_size = tuple()

    arr_shape = np.asarray(data.shape)
    
    ratio = max_size / np.max(arr_shape)
    arr_shape = ratio * arr_shape
    
    new_size = tuple(arr_shape.astype(int))

    log.info("Resizing data {} from {} to {}".format(type(data), data.shape, new_size))
    
    data_resized = resize(data, new_size, anti_aliasing=False)
    
    return np.array(data_resized)    
        
def mask_image(raw_data, mask):
    """
    Processes the image to extract only regions of interest.

    Parameters
        raw_data : The data to process.
        mask     : The mask. This parameter should be of the same type and dimensions than raw_data.

    Returns
        masked_data : the masked data. Returned data has the same dimension as raw_data. The pixels/voxels of interest keeps their raw values and non-relevant pixels/voxels are set to 0.
    """
    assert raw_data.shape == mask.shape
    # In theory, if the processed image is a medical image, we should also assert thant cells dimension of the sensor (voxel size) are equal.
    
    return raw_data * mask

def binarize(data, threshold=0.5):
    """
    Binarise an image.
    Pixel value <= threshold    --> 0 
    Pixel value > threshold     --> 1 

    Parameters
        data                : The image to binarize.
        threshold (float)   : The threshold. Default is 0.5.

    Returns
        The binarized data
    """
    return (data > threshold).astype(int)

def distance_map(data, method="edt"):
    """
    Distance map transformation.
    Implemented transformations :
        - "edt" : Exact Euclidean Distance Transform 

    Parameters
        data            : The image to transform.
        method (str)    : The key of the transformation to compute ; see implemented transformations.

    Returns
        The distance map
    """

    data = binarize(data) # Binarize the data

    # Exact Euclidean Distance Transform
    if method == "edt": 
        from scipy import ndimage as ndi

        distance_map = ndi.distance_transform_edt(data)

    return distance_map

def main(args):
    # Manage args
    input_path = args.input
    output_path = args.output if args.output is not None else "./resized_" + os.path.basename(input_path)
    new_size = (args.x, args.y, args.z)
    
    # Resize part
    raw_volume = nib.load(input_path)
    resized_volume = resize_image(raw_volume.get_fdata(), new_size)
    nii_resized_volume = nib.Nifti1Image(resized_volume, raw_volume.affine)
    nib.save(nii_resized_volume, output_path)
    
    log.info("Resized data has been saved at {}".format(output_path))


if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--verbose', '-v',  action='count', default=0)
    parser.add_argument('input',            type=str, help="path of the data to resize.")
    parser.add_argument('x',                type=int, help="new data size on the x axis.")
    parser.add_argument('y',                type=int, help="new data size on the y axis." )
    parser.add_argument('z',                type=int, help="new data size on the z axis.")
    parser.add_argument('--output', '-out', type=str, help="output file (<path>/<filename>.nii)")
    args = parser.parse_args()
    
    log.basicConfig(format="%(levelname)s: %(message)s", level = args.verbose * 10 if args.verbose > 0 and args.verbose <= 5 else 4 * 10) # log level correspondant au nombre de 'v' contenu dans l'arg s'il est spécifié (dans la limite de 5) sinon level error 
    
    main(args)