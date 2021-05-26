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


class DiceLoss(__Loss):
    """ Dice loss function. """
    def __init__(self, batch_size, image_size):
        super().__init__(batch_size, image_size)

    def call(self, *args, **kwargs):
        y_true, y_pred = args[0], args[1]

        numerator = 2 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true ** 2) + K.sum(y_pred ** 2)

        dice_coefficient = numerator / denominator

        loss = 1 - dice_coefficient

        # Return loss normalized by batch size and image size for stable LRs across different input sizes
        return loss / self.batch_size
