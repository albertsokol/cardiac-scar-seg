from functools import reduce

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss


class __Loss(Loss):
    """ Generic methods for all loss functions. """
    def __init__(self, batch_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size

    @staticmethod
    def get_num_vox(t):
        """Get the number of voxels in a tensor."""
        shape = tf.shape(t)[:-1]
        num_voxels = tf.math.reduce_prod(shape)
        return tf.cast(num_voxels, tf.float32)

    def get_config(self):
        return super().get_config()

    def call(self, *args, **kwargs):
        raise NotImplementedError

    @staticmethod
    def clip(z):
        """ Clip all values in a tensor to prevent divide by 0 errors. """
        z = K.clip(z, 1e-7, 1)
        return z


class SoftmaxLoss(__Loss):
    """ Basic non-weighted softmax loss. """
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def call(self, *args, **kwargs):
        y_true, y_pred = args[0], args[1]

        loss = - K.sum(y_true * K.log(self.clip(y_pred)))

        # Return loss normalized by batch size and image size for stable LRs across different input sizes
        return loss / self.get_num_vox(y_pred)


class WeightedSoftmaxLoss(__Loss):
    """ Categorical cross-entropy loss, which is used in the original UNet-3D paper. """
    def __init__(self, batch_size, label_weights):
        super().__init__(batch_size)
        self.label_weights = label_weights

    def call(self, *args, **kwargs):
        y_true, y_pred = args[0], args[1]

        # label_weights is be a len(labels) length vector of voxel weights for each label
        loss = - K.sum(self.label_weights * y_true * K.log(self.clip(y_pred)))

        # Return loss normalized by batch size and image size for stable LRs across different input sizes
        return loss / self.get_num_vox(y_pred)


class DiceLoss(__Loss):
    """ Dice loss function. """
    def __init__(self, batch_size):
        super().__init__(batch_size)

    def call(self, *args, **kwargs):
        y_true, y_pred = K.flatten(args[0]), K.flatten(args[1])

        numerator = 2 * K.sum(y_true * y_pred) + 1e-7
        denominator = K.sum(y_true) + K.sum(y_pred) + 1e-7

        dice_coefficient = numerator / denominator

        return 1 - dice_coefficient


class WeightedSoftmaxDiceLoss(__Loss):
    """ A combination of weighted cross-entropy loss and Dice loss with adjustable weighting. """
    def __init__(self, batch_size, label_weights, dice_weight=0.5):
        """
        Initialiser for WeightedSoftmaxDiceLoss.

        Params
        ======
        :param batch_size: int: number of examples per training batch
        :param label_weights:
        :param dice_weight: float: e.g., if 2., then Dice loss is doubled to weight it more strongly
        """
        super().__init__(batch_size)
        self.label_weights = label_weights
        self.dice_weight = dice_weight

    def call(self, *args, **kwargs):
        y_true, y_pred = args[0], args[1]

        # First, get the softmax loss
        softmax_loss = - K.sum(self.label_weights * y_true * K.log(self.clip(y_pred)))
        softmax_loss /= self.get_num_vox(y_pred)

        # Then, get the Dice loss
        numerator = 2 * K.sum(y_true * y_pred) + 1e-7
        denominator = K.sum(y_true) + K.sum(y_pred) + 1e-7
        dice_coefficient = numerator / denominator
        dice_loss = 1 - dice_coefficient

        # Finally, combine the two
        return self.dice_weight * dice_loss + softmax_loss


class WeightedSoftmaxDiceLossPlusQuality(__Loss):
    """
    A combination of weighted cross-entropy loss and Dice loss with adjustable weighting and quality score weighting.
    """
    def __init__(self, batch_size, label_weights, full_3d_mode, dice_weight=0.5):
        """
        Initialiser for WeightedSoftmaxDiceLossPlusQuality.

        Params
        ======
        :param batch_size: int: number of examples per training batch
        :param label_weights:
        :param dice_weight: float: e.g., if 2., then Dice loss is doubled to weight it more strongly
        """
        super().__init__(batch_size)
        self.label_weights = label_weights
        self.dice_weight = dice_weight
        self.curr_slice_qualities = None
        self.full_3d_mode = full_3d_mode

        if self.full_3d_mode:
            self.qw_out_name = ["IteratorGetNext:1", "Identity_1:0"]
            self.sum_axes = [1, 2, 4]
        else:
            self.qw_out_name = ["ExpandDims:0"]
            self.sum_axes = [1, 2, 3]

    def call(self, *args, **kwargs):
        y_true, y_pred = args[0], args[1]

        # Update the quality for the current slice if the current call is for the 'qw_out' tensor
        if any([x in y_pred.name for x in self.qw_out_name]):
            if self.full_3d_mode:
                self.curr_slice_qualities = tf.identity(y_pred)
            else:
                self.curr_slice_qualities = tf.squeeze(y_pred)
            # Return a loss of 0 for this case
            return 0.

        # First, get the softmax loss
        softmax_loss = - K.sum(self.label_weights * y_true * K.log(self.clip(y_pred)), axis=self.sum_axes)
        softmax_loss /= self.get_num_vox(y_pred)

        # Then, get the Dice loss
        numerator = 2 * K.sum(y_true * y_pred, axis=self.sum_axes) + 1e-7
        denominator = K.sum(y_true, axis=self.sum_axes) + K.sum(y_pred, axis=self.sum_axes) + 1e-7
        dice_coefficient = numerator / denominator
        dice_loss = 1 - dice_coefficient

        # Combine the two and apply the image-wise quality weightings
        ele_wise_loss = self.dice_weight * dice_loss + softmax_loss
        ele_wise_loss *= self.curr_slice_qualities

        return K.sum(ele_wise_loss)
