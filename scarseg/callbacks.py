import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.callbacks import Callback


class LearningRatePrinter(Callback):
    """Pretty simple: just prints the learning rate."""

    def __init__(self, frequency="epoch"):
        """
        Initializer for LearningRatePrinter.

        :param frequency: str:
        """
        super().__init__()
        self.frequency = frequency

    def on_batch_end(self, batch, logs=None):
        if self.frequency == "batch":
            print(
                f"Current learning rate: {K.eval(self.model.optimizer._decayed_lr(tf.float32))}"
            )

    def on_epoch_end(self, epoch, logs=None):
        print(
            f"Current learning rate: {K.eval(self.model.optimizer._decayed_lr(tf.float32))}"
        )
