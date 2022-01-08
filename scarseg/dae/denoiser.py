import numpy as np
import tensorflow as tf

from scarseg.metrics import DiceMetric, ClassWiseDiceMetric


class Denoiser:
    """Class for denoising postprocessing using a trainer DenoisingUnet model."""
    def __init__(self, model_path, label_len):
        self.model = self.load_model(model_path)
        self.label_len = label_len

    @staticmethod
    def load_model(model_path):
        """ Loads the pretrained model. """
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={
                'DiceMetric': DiceMetric,
                'ClassWiseDiceMetric': ClassWiseDiceMetric,
            },
        )

    def get_one_hot(self, f):
        """Get one-hot encoding representations of labels and predictions."""
        return tf.one_hot(f, self.label_len, dtype=np.int8).numpy()

    def denoise(self, f):
        """Denoise the given image using the denoiser model and return the output. Expects one-hot encoded input."""
        denoised = np.empty(f.shape, dtype=np.int8)

        for i in range(f.shape[-1]):
            one_hot_slice = self.get_one_hot(f[..., i])
            pred = self.model.predict(one_hot_slice[np.newaxis, ...])
            denoised[..., i] = np.argmax(pred, axis=-1)

        return denoised
