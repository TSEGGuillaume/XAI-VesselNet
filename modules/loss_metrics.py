import numpy as np

#-----------------------------------------------------#
#              Standard Dice coefficient              #
#-----------------------------------------------------#
def dice_coefficient(arr_1, arr_2, smooth=0.00001):
    """
    Compute Dice coefficient between two matrices.
    This implementation does not allow the multiclass Dice assessment.

    Parameters
        arr_1           : First matrix
        arr_2           : Second matrix
        smooth (float)  : Smoothing value

    Returns
        <expression> (Tensor) : the one-element tensor containing the computed dice cefficient.
    """
    from tensorflow.keras import backend as K

    arr_1_f = K.flatten(arr_1)
    arr_2_f = K.flatten(arr_2)
    intersection = K.sum(arr_1_f * arr_2_f)

    return (2. * intersection + smooth) / \
           (K.sum(arr_1_f) + K.sum(arr_2_f) + smooth)

def dice_coefficient_loss(y_true, y_pred):
    """
    Compute the Dice loss function.

    Parameters
        y_true  : The matrix reprensentation of the ground-truth
        y_pred  : The matrix reprensentation of the prediction

    Returns
        <expression> (Tensor) : the one-element tensor containing the Dice loss function value
    """
    return 1 - dice_coefficient(y_true, y_pred)

def main():
    # TODO : write unit tests for the module

    arr_1 = np.ones((4, 4))
    arr_2 = np.ones_like(arr_1)

    arr_2 = arr_2 * 0.5

    print(arr_1)
    print(arr_2)

    res = dice_coefficient_loss(arr_1, arr_2)
    print(type(res))
    print(res)

if __name__=="__main__":    
    main()