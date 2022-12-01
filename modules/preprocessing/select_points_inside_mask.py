import os

import numpy as np

def create_mask_inner_not_outer(inner_mask, outer_mask, savepath:str=None):
    # "booleanize" the masks
    inner_mask_bin = inner_mask > 0
    outer_mask_bin = outer_mask > 0

    not_outer_mask_bin = np.invert(outer_mask_bin)

    inner_not_outer_mask = np.logical_and(inner_mask_bin, not_outer_mask_bin)

    if savepath is not None:
        import nibabel as nib

        # Of course the save volume will not fit with the orientation of the image from which the mask is computed from
        # The saved mask is (at the moment) only for information 
        mask_volume = nib.Nifti1Image(np.array(inner_not_outer_mask, dtype=np.float32), np.eye(4))
        nib.save(mask_volume, os.path.join(savepath, "inner_outer_mask.nii"))

    return inner_not_outer_mask

def get_inner_not_outer_points(inner_mask, outer_mask, sample_size:int=None, mask_savepath:str=None) -> list:

    inner_not_outer_mask = create_mask_inner_not_outer(inner_mask, outer_mask, savepath=mask_savepath)

    positions_inside_mask = np.stack(np.where(inner_not_outer_mask != 0), axis=-1)

    if sample_size == None:
        selected_positions = positions_inside_mask
    else:
        np.random.shuffle(positions_inside_mask) # Shuffle positions vector to select random points inside the mask

        selected_positions = positions_inside_mask[0:sample_size]

    return selected_positions