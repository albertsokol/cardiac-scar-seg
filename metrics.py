import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric
import tensorflow as tf


class DiceMetric(Metric):
    def __init__(self, batch_size, name='dice', **kwargs):
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.batch_size = batch_size
        self.total_dice = self.add_weight(name='dice_value', initializer='zeros')
        self.num_examples = self.add_weight(name='num_examples', initializer='zeros')

    def update_state(self, y_true, y_pred):
        self.num_examples.assign_add(self.batch_size)
        numerator = 2 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true ** 2) + K.sum(y_pred ** 2)
        self.total_dice.assign_add(numerator / denominator)

    def result(self):
        return self.total_dice / self.num_examples


class ClassWiseDiceMetric(Metric):
    def __init__(self, batch_size, i, name='c_dice', **kwargs):
        super(ClassWiseDiceMetric, self).__init__(name=name, **kwargs)
        self.batch_size = batch_size
        self.i = i
        self.num_examples = self.add_weight(name='num_examples', initializer='zeros')
        self.total_dice = self.add_weight(name='dice_value', initializer='zeros')

    def update_state(self, y_true, y_pred):
        self.num_examples.assign_add(self.batch_size)
        numerator = 2 * K.sum(y_true[..., self.i] * y_pred[..., self.i])
        denominator = K.sum(y_true[..., self.i] ** 2) + K.sum(y_pred[..., self.i] ** 2)
        self.total_dice.assign_add(numerator / denominator)

    def result(self):
        return self.total_dice / self.num_examples
