import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric


class __Metric(Metric):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def get_config(self):
        return super().get_config()

    def update_state(self, *args, **kwargs):
        raise NotImplementedError

    def result(self):
        raise NotImplementedError


class DiceMetric(__Metric):
    def __init__(self, batch_size=1, name='dice', **kwargs):
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.batch_size = batch_size
        self.total_dice = self.add_weight(name='dice_value', initializer='zeros')
        self.num_batches = self.add_weight(name='num_batches', initializer='zeros')

    def update_state(self, y_true, y_pred):
        self.num_batches.assign_add(1)
        y_true, y_pred = K.flatten(y_true), K.flatten(y_pred)
        numerator = 2 * K.sum(y_true * y_pred) + 1e-7
        denominator = K.sum(y_true) + K.sum(y_pred) + 1e-7
        self.total_dice.assign_add(numerator / denominator)

    def result(self):
        return self.total_dice / self.num_batches


class ClassWiseDiceMetric(__Metric):
    def __init__(self, batch_size=1, i=0, name='c_dice', **kwargs):
        super(ClassWiseDiceMetric, self).__init__(name=name, **kwargs)
        self.batch_size = batch_size
        self.i = i
        self.total_dice = self.add_weight(name='dice_value', initializer='zeros')
        self.num_batches = self.add_weight(name='num_batches', initializer='zeros')

    def update_state(self, y_true, y_pred):
        self.num_batches.assign_add(1)
        # Trim to only the current i and then flatten
        y_true = K.flatten(y_true[..., self.i])
        y_pred = K.flatten(y_pred[..., self.i])
        numerator = 2 * K.sum(y_true * y_pred) + 1e-7
        denominator = K.sum(y_true) + K.sum(y_pred) + 1e-7
        self.total_dice.assign_add(numerator / denominator)

    def result(self):
        return self.total_dice / self.num_batches
