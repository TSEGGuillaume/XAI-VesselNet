from tensorflow.keras.losses import Loss

import utils.metrics as metrics

class CustomDiceLoss(Loss):
    def __init__(self, smooth=0.0001):
        super().__init__()

        self.smooth = smooth
        
    def __call__(self, y_truth, y_pred, sample_weight=None):
        return 1.0-metrics.compute_dice(y_truth, y_pred, self.smooth)
    
class CustomFocalLoss(Loss):
    """Compute focal loss.
    
    # Arguments
        y_true: Ground truth targets,
            tensor of shape (?, num_boxes, num_classes).
        y_pred: Predicted logits,
            tensor of shape (?, num_boxes, num_classes).
    
    # Returns
        focal_loss: Focal loss, tensor of shape (?, num_boxes).

    # References
        https://arxiv.org/abs/1708.02002
    """
    def __init__(self, p_gamma=2, p_alpha=0.25):
        super().__init__()

        self.gamma = p_gamma
        self.alpha = p_alpha

    def __call__(self, y_truth, y_pred, sample_weight=None):
        import tensorflow as tf
        from tensorflow.keras import backend as K
    
        eps = K.epsilon()
        y_pred = K.clip(y_pred, eps, 1. - eps)
    
        pt = tf.where(tf.equal(y_truth, 1), y_pred, 1 - y_pred)
        focal_loss = -tf.reduce_sum(self.alpha * K.pow(1. - pt, self.gamma) * K.log(pt), axis=-1)
        return focal_loss 
