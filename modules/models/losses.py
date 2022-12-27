from tensorflow.keras.losses import Loss

import utils.metrics as metrics

class CustomDiceLoss(Loss):
    """
    Dice loss function.
    
    Parameters
        smooth (float)    : Smooth value to avoid division by zero. Optional (default to 0.0001)
    """
    def __init__(self, smooth=0.0001):
        super().__init__()

        self.smooth = smooth
        
    def __call__(self, y_truth, y_pred, sample_weight=None):
        """
        Loss computation callback.

        Parameters 
            y_true  : The ground truth image.
            y_pred  : The prediction image.

        Returns
            <expr> : loss value
        """
        return 1.0 - metrics.compute_dice(y_truth, y_pred, self.smooth)

class CustomDiceClDiceLoss(Loss):
    """
    Compound loss function, combining Dice loss and centerline-Dice loss.
    
    References
        - clDice : 10.1109/CVPR46437.2021.01629

    Parameters
        iters_skel (int)  : Number of iterations for skeletonization. Optional (default to 10).
        alpha (float)     : Weight for the cldice component (between [0, .5]). Optional (default to 0.5).
        smooth (float)    : Smooth value to avoid division by zero. Optional (default to 0.0001)
    
    Notes
        - The number of skeletonization iteration depends on the dataset. It should match with the radius of the larger tubular structure of the dataset. Choosing a larger k does not reduce performance but increases computation time. On the other hand, a too low k leads to incomplete skeletonization.  
    """
    def __init__(self, iters_skel = 10, alpha=0.5, smooth=0.0001):
        super().__init__()

        self.iters = iters_skel
        self.alpha = alpha
        self.smooth = smooth

    def __call__(self, y_true, y_pred, sample_weight=None):
        """
        Loss computation callback.

        Parameters 
            y_true  : The ground truth image.
            y_pred  : The prediction image.

        Returns
            <expr> : loss value
        """
        import models.cldice as cldice
        import tensorflow as tf
        from tensorflow.keras import backend as K

        # Compute skeletons
        skel_pred = cldice.soft_skel(y_pred, self.iters)
        skel_true = cldice.soft_skel(y_true, self.iters)

        # Compute precision and recall
        pres    = (K.sum(tf.math.multiply(skel_pred, y_true)) + self.smooth) / (K.sum(skel_pred) + self.smooth)    
        rec     = (K.sum(tf.math.multiply(skel_true, y_pred)) + self.smooth) / (K.sum(skel_true) + self.smooth)

        cldice_loss = 1.0 - 2.0 * (pres * rec) / (pres + rec)
        dice_loss   = 1.0 - metrics.compute_dice(y_true, y_pred)
        return (1.0 - self.alpha) * dice_loss + self.alpha * cldice_loss