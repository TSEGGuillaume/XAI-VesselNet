import os
import logging as log

import tensorflow as tf
import nibabel as nib
import numpy as np

from tensorflow.keras.preprocessing.image import ImageDataGenerator

import preprocessing.preprocess_image as preprocess_image


class Custom3dIrcadbGen(tf.keras.utils.Sequence):
    """
    Custom 3Dircadb1 dataset.
    Generate a sequence walking through nifti files 3Dircadb1.

    Parameters
        data_dir (str)      : Path to the data directory
        gt_dir (str)        : Path to the ground-truth directory
        batch_size (int)    : Size of the batch. Default 1
        input_size (tuple)  : Shape of the input tensor. Default (256, 256, 80)
        suffle (boolean)    : Indicate if the data should be suffled
    """
    def __init__(self, data_dir:str,
                 gt_dir:str, 
                 batch_size = 1,
                 input_size = (256, 256, 80, 1),
                 shuffle    = False):
        self.data_dir   = data_dir
        self.gt_dir     = gt_dir
        self.batch_size = batch_size
        self.input_size = input_size
        self.shuffle    = shuffle

        # List the training data and ground-truth files
        # TODO : Maybe look for a cleaner way to associate (data,gt)
        self.xy_filename = []
        ext = '.nii'
        
        for f in [f for f in os.listdir(data_dir) if f.endswith(ext)]:
            try:
                x = os.path.join(self.data_dir, f)
                y = os.path.join(self.gt_dir, f[:-len(ext)] + "_vesselmask" + ext)

                # Verify that the data has an assosciated ground-truth
                # The ground-truth files must have the same name as the data + the suffix `_vesselmask`
                if os.path.isfile(y):
                    self.xy_filename.append((x, y))
                else:
                    raise FileNotFoundError(y)
            except FileNotFoundError as e:
                log.warning("Ground-truth file not found. \nSkipped file :", e.args[0])
                continue    
        
        self.n = len(self.xy_filename) # Size of the dataset

        log.debug("The dataset contains {} samples.".format(self.n), self.xy_filename)

    def on_epoch_end(self):
        pass
    
    def __get_volume_data(self, path):
        """
        Access a nifti data.
        Read the data from the given path and preprocess it :
         - Change the precision to float16 ;
         - Rescale to the shape of the input tensor ;
         - Normaliza pixels/voxels values.

        Parameters
            path (str) : Path to the data to access

        Returns
            <expression> (ndarray) : The data
        """
        image = nib.load(path)
        image_arr = np.array(image.get_fdata(), dtype=np.float16)

        resized_image = preprocess_image.resize_image(image_arr, self.input_size[0:-1]) # Preprocess the volume by resizing ; We do not take channels into account

        return (resized_image - np.min(resized_image)) / (np.max(resized_image) - np.min(resized_image)) # Data normalization

    def __get_data(self, xy_subset_filename):        
        """
        Generate a batch of data.

        Parameters
            xy_subset_filename (str) : Pathes to the data to access

        Returns
            <expression> (tuple) : The batch X,Y as ([],[])
        """
        X_batch = np.asarray([self.__get_volume_data(x[0]) for x in xy_subset_filename])
        Y_batch = np.asarray([self.__get_volume_data(x[1]) for x in xy_subset_filename])
        
        return X_batch, Y_batch

    def __getitem__(self, index):
        """
        Walking through the Sequence of data.

        Parameters
            index (int) : Index for the iteration

        Returns
            <expression> (tuple) : The tuple data, ground-truth ([],[]). Each sample has a shape (batch, `dims`, channel) with `dims` the dimension of the data.
        """
        xy_subset_filename = self.xy_filename[index * self.batch_size:(index + 1) * self.batch_size]

        log.debug(xy_subset_filename)

        X, y = self.__get_data(xy_subset_filename)        
        return np.expand_dims(X, axis=-1), np.expand_dims(y, axis=-1)
    
    def __len__(self):
        return self.n // self.batch_size