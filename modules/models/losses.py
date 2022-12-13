from tensorflow.keras.losses import Loss

import utils.metrics as metrics

class CustomDiceLoss(Loss):
    def __init__(self, smooth=0.0001):
        super().__init__()

        self.smooth = smooth
        
    def __call__(self, y_truth, y_pred, sample_weight=None):
        return 1.0-metrics.compute_dice(y_truth, y_pred, self.smooth)