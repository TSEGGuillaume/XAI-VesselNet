import argparse

import os
import logging as log

from datetime import datetime
import nibabel as nib
import numpy as np

import configuration as appcfg
import preprocessing.select_points_inside_mask as select_not_vessel_nodes

def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('N', type=int, help="the number of points to output")

    parser.add_argument('--verbose', '-v', action='count', default=0)

    args = parser.parse_args()

    return args


def main():
    liver_mask_volume = nib.load(os.path.join(cfg.data_dir, "liver_mask_iso_resized.nii"))
    vessel_mask_volume = nib.load(os.path.join(cfg.data_dir, "liver_vessel_mask_iso_resized.nii"))

    dump_positions = select_not_vessel_nodes.get_inner_not_outer_points(liver_mask_volume.get_fdata(), vessel_mask_volume.get_fdata(), args.N, mask_savepath=os.path.join(cfg.data_dir, "_temp"))

    # Saving data on the disk
    filename = "liver_not_vessel_points_{}.csv".format(datetime.now().strftime('%Y%m%d-%H%M%S')) # datetime as ID to not overwrite previously generated file(s)
    save_path = os.path.join(cfg.result_dir, filename)
    np.savetxt(save_path, np.array(dump_positions), fmt='%.18e', delimiter=';') # Save positions

    log.info("The points list has been saved at {}".format(save_path))

if __name__=="__main__":
    cfg = appcfg.CConfiguration(p_filename="default.ini")

    args = parse_arguments()

    log.basicConfig(format="%(levelname)s: %(message)s", level = args.verbose * 10 if args.verbose > 0 and args.verbose <= 5 else 4 * 10)

    main()