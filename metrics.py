import tensorflow.keras.backend as K
from tensorflow.keras.metrics import Metric


class DiceMetric(Metric):
    def __init__(self, name='dice', **kwargs):
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.dice = self.add_weight(name='dice_value', initializer='zeros')

    def update_state(self, y_true, y_pred):
        numerator = 2 * K.sum(y_true * y_pred)
        denominator = K.sum(y_true ** 2) + K.sum(y_pred ** 2)
        self.dice.assign_add(numerator / denominator)

    def result(self):
        return self.dice
