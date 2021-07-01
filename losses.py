from functools import reduce

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.losses import Loss


class __Loss(Loss):
    """ Generic methods for all loss functions. """
    def __init__(self, batch_size, image_size, **kwargs):
        super().__init__(**kwargs)
        self.batch_size = batch_size
        self.num_vox = reduce(lambda x, y: x * y, image_size, 1)

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


class DiceLoss(__Loss):
    """ Dice loss function. """
    def __init__(self, batch_size, image_size):
        super().__init__(batch_size, image_size)

    def call(self, *args, **kwargs):
        y_true, y_pred = K.flatten(args[0]), K.flatten(args[1])

        numerator = 2 * K.sum(y_true * y_pred) + 1e-7
        denominator = K.sum(y_true) + K.sum(y_pred) + 1e-7

        dice_coefficient = numerator / denominator

        return 1 - dice_coefficient


class WeightedDiceLoss(__Loss):
    """ Dice loss, but each class is taken separately and weighted to give the final loss values. """
    def __init__(self, batch_size, image_size, label_weights):
        super().__init__(batch_size, image_size)
        self.label_weights = label_weights

    def call(self, *args, **kwargs):
        y_true, y_pred = args[0], args[1]

        # Should reduce to 5-dim vector somehow - use axis or keepdims ?
        numerator = 2 * K.sum(y_true * y_pred) + 1e-7
        denominator = K.sum(y_true) + K.sum(y_pred) + 1e-7


class WeightedSoftmaxDiceLoss(__Loss):
    """ A combination of weighted cross-entropy loss and Dice loss with adjustable weighting. """
    def __init__(self, batch_size, image_size, label_weights, dice_weight=0.5):
        """
        Initialiser for WeightedSoftmaxDiceLoss.

        Params
        ======
        :param batch_size: int: number of examples per training batch
        :param image_size:
        :param label_weights:
        :param dice_weight: float: e.g., if 2., then Dice loss is doubled to weight it more strongly
        """
        super().__init__(batch_size, image_size)
        self.label_weights = label_weights
        self.dice_weight = dice_weight

    def call(self, *args, **kwargs):
        y_true, y_pred = args[0], args[1]

        # First, get the softmax loss
        softmax_loss = - K.sum(self.label_weights * y_true * K.log(self.clip(y_pred)))
        softmax_loss = softmax_loss / self.num_vox / self.batch_size

        # Then, get the Dice loss
        numerator = 2 * K.sum(y_true * y_pred) + 1e-7
        denominator = K.sum(y_true) + K.sum(y_pred) + 1e-7
        dice_coefficient = numerator / denominator
        dice_loss = 1 - dice_coefficient

        # Finally, combine the two
        return self.dice_weight * dice_loss + softmax_loss


class CascadedWeightedSoftmaxDiceLoss(__Loss):
    def __init__(self, batch_size, image_size, label_weights, dice_weight=0.5):
        super().__init__(batch_size, image_size)
        self.label_weights = label_weights
        self.dice_weight = dice_weight

    def call(self, *args, **kwargs):
        y_true, y_pred = args[0], args[1]

        # First, get the softmax loss
        if "general" in args[1].name:
            curr_weights = self.label_weights["general"]
            softmax_loss = - K.sum(curr_weights * y_true * K.log(self.clip(y_pred)))
        else:
            if "scar" in args[1].name:
                curr_weights = self.label_weights["scar"]
            else:
                curr_weights = self.label_weights["pap"]
            softmax_loss = curr_weights * tf.keras.metrics.binary_crossentropy(y_true, y_pred)

        softmax_loss = softmax_loss / self.num_vox / self.batch_size

        # Then, get the Dice loss
        numerator = 2 * K.sum(y_true * y_pred) + 1e-7
        denominator = K.sum(y_true) + K.sum(y_pred) + 1e-7
        dice_coefficient = numerator / denominator
        dice_loss = 1 - dice_coefficient

        # Finally, combine the two
        return self.dice_weight * dice_loss + softmax_loss
