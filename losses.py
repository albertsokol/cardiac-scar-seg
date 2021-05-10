import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss


class __Loss(Loss):
    """ Generic methods for all loss functions. """
    def __init__(self, batch_size, image_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_vox = image_size[0] * image_size[1] * image_size[2]

    @staticmethod
    def clip(z):
        """ Clip all values in a tensor to prevent divide by 0 errors. """
        z = K.clip(z, 1e-7, 1)
        return z


class SoftmaxLoss(__Loss):
    """ Basic non-weighted softmax loss. """
    def __init__(self, batch_size, image_size):
        super().__init__(batch_size, image_size)

    def call(self, *args, **kwargs):
        y_true, y_pred = args[0], args[1]

        loss = - K.sum(y_true * K.log(self.clip(y_pred)))

        # Return loss normalized by batch size and image size for stable LRs across different input sizes
        return loss / self.num_vox / self.batch_size


class WeightedSoftmaxLoss(__Loss):
    """ Categorical cross-entropy loss, which is used in the original UNet-3D paper. """
    def __init__(self, batch_size, image_size, label_weights):
        super().__init__(batch_size, image_size)
        self.label_weights = label_weights

    def call(self, *args, **kwargs):
        y_true, y_pred = args[0], args[1]

        # label_weights is be a len(labels) length vector of voxel weights for each label
        loss = - K.sum(self.label_weights * y_true * K.log(self.clip(y_pred)))

        # Return loss normalized by batch size and image size for stable LRs across different input sizes
        return loss / self.num_vox / self.batch_size


def weighted_pixel_bce_loss(beta, batch_size):
    """
    Weighted pixel-wise binary cross entropy loss function, averaged over the batch.
    Parameters:
    beta: weighting factor. Set this to be the average proportion of class = 1 pixels in a training image
    batch_size: number of training examples in the batch
    """

    def compute_loss(y_true, y_pred):
        # Assign a greater loss to false negative predictions to prevent model always predicting y = 0 for all pixels
        px_wt = 1. / beta
        # Find the number of total pixels in an image
        num_pix = K.int_shape(y_pred)[1] * K.int_shape(y_pred)[2]
        # Calculate the loss
        bce = - ((px_wt * y_true * K.log(clip(y_pred))) + (1 - y_true) * K.log(clip(1 - y_pred)))

        # Sum and average the loss by the number of pixels and by the batch size
        loss = (K.sum(bce) / num_pix) / batch_size

        return loss

    return compute_loss


def dice_loss():
    """
    Computes the dice loss for a predicted segmentation.
    """

    def compute_loss(y_true, y_pred):
        # Compute dice coefficient and return loss
        numerator = 2 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true ** 2) + K.sum(y_pred ** 2)
        dice_coefficient = numerator / denominator
        loss = 1 - dice_coefficient

        return loss

    return compute_loss


def combined_dice_wpce_loss(beta, batch_size):
    """
    A combination of 2x dice loss and 1x weighted pixel-wise binary cross entropy loss for a prediction.
    Tends to result in better performance than wpce or dice alone.
    Parameters:
    beta: weighting factor. Set this to be the average proportion of class = 1 pixels in a training image
    batch_size: number of training examples in the batch
    """

    def compute_loss(y_true, y_pred):
        # Get weighted pixel cross entropy loss
        # Assign a greater loss to false negative predictions to prevent model always predicting y = 0 for all pixels
        px_wt = 1. / beta
        # Find the number of total pixels in an image
        num_pix = K.int_shape(y_pred)[1] * K.int_shape(y_pred)[2]
        bce = - ((px_wt * y_true * K.log(clip(y_pred))) + (1 - y_true) * K.log(clip(1 - y_pred)))

        # Sum and average the loss by the number of pixels and by the batch size
        wpce_loss = (K.sum(bce) / num_pix) / batch_size

        # Compute dice coefficient and return loss
        numerator = 2 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true ** 2) + K.sum(y_pred ** 2)
        dice_coefficient = numerator / denominator
        _dice_loss = 1 - dice_coefficient

        # The final loss value is a mix of both loss functions
        loss = 2 * _dice_loss + wpce_loss

        return loss

    return compute_loss
