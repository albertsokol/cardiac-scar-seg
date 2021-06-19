import json
import os
import tensorflow as tf
import numpy as np

from losses import SoftmaxLoss, DiceLoss, WeightedSoftmaxDiceLoss, WeightedSoftmaxLoss
from metrics import DiceMetric, ClassWiseDiceMetric
from readers import NIIReader, NPYReader


def np_dice_coefficient(seg_gt, seg_pred):
    """ Computes the dice coefficient between gt and predicted segmentations using numpy rather than Keras. """
    numerator = 2 * np.sum(seg_gt * seg_pred)
    denominator = np.sum(seg_gt ** 2) + np.sum(seg_pred ** 2)

    _dice = numerator / denominator

    return _dice


class Predictor:
    def __init__(self, pipeline, model_paths, data_path, dataset):
        self.rng = np.random.default_rng()

        self.loss_dict = {
            'softmax': SoftmaxLoss,
            'weighted softmax': WeightedSoftmaxLoss,
            'dice': DiceLoss,
            'weighted softmax dice': WeightedSoftmaxDiceLoss
        }

        assert pipeline in ["single", "cascaded", "multi2D"], f"pipeline type {pipeline} not recognised"
        self.pipeline = pipeline

        self.image_size = None
        self.model_name = None
        self.dims = None
        self.reader = None

        if len(model_paths) > 1:
            # TODO: combine all models in a pipeline type thing ? Could be a lil complicated
            self.model = 0
        else:
            self.model = self.load_model(model_paths[0])

        self.data_path = data_path
        assert dataset in ["train", "val", "test"], f"dataset must be one of: 'train', 'val', 'test'; but got {dataset}"
        self.dataset = dataset
        self.data_path = os.path.join(data_path, self.dims, self.dataset)
        self.image_fnames = [os.path.join(self.data_path, x, f'{x}_SAX.nii.gz') for x in sorted(os.listdir(self.data_path))]
        self.label_fnames = [os.path.join(self.data_path, x, f'{x}_SAX_mask2.nii.gz') for x in sorted(os.listdir(self.data_path))]

    def load_model(self, model_path):
        """ Loads the pretrained model and config. """
        with open(os.path.join(model_path, 'train_config.json'), 'r') as train_config_file:
            train_config = json.load(train_config_file)

        self.model_name = train_config['model']
        self.image_size = train_config['image_size']

        # Keep an eye on this and make sure the right 3D models can all be found in this list
        # Make sure that the correct readers and data are loaded for the given config / dimensionality
        if self.model_name in ['UNet3D']:
            self.dims = '3D'
            self.reader = NIIReader()
        else:
            plane = train_config['plane']
            self.dims = '2D'
            self.dataset = f'{self.dataset}/{plane}'
            self.reader = NPYReader()

        # Might get an error about custom objects - will need to include the losses and metrics if so
        return tf.keras.models.load_model(
            model_path,
            compile=False,
            custom_objects={
                'DiceMetric': DiceMetric,
                'ClassWiseDiceMetric': ClassWiseDiceMetric
            }
        )

    def load_image_label(self, fname):
        """ Loads the image and label files. """
        # Define paths
        if fname is None:
            idx = self.rng.integers(0, len(self.image_fnames))
            image_path = self.image_fnames[idx]
            label_path = self.label_fnames[idx]

        # Load image and label
        image = self.reader.read(image_path)
        label = self.reader.read(label_path)

        # Set to the correct dimensions
        if image.shape != self.image_size:
            image = self.reader.resize(image, self.image_size)
        if label.shape != self.image_size:
            label = self.reader.resize(label, self.image_size, interpolation_order=0)

        image = self.reader.normalize(image)

        # Set to the correct rank
        image = image[np.newaxis, ..., np.newaxis]

        print(f'Loaded image at {image_path}')
        print(f'Loaded label at {label_path}')
        return image, label

    def display(self, image, label, pred_label, plane='transverse'):
        """
        Should show the label vs. prediction, label vs. image and prediction vs. image in a single scrollable view.
        """
        # TODO: also look @ the pred label on image, pred label on label, label on image in same view
        # Set all arrays to the same shape and rank
        image = np.squeeze(image)
        pred_label = np.squeeze(np.argmax(pred_label, axis=-1))
        print(pred_label.shape)
        print(np_dice_coefficient(label, pred_label))

        self.reader.scroll_view(np.concatenate((label, pred_label)), plane=plane)

    def predict(self, fname=None):
        image, label = self.load_image_label(fname)
        # This might be too much customisation for nothing tbh
        if self.pipeline == "single":
            pred_label = self.model.predict(image)
            self.display(image, label, pred_label)
        elif self.pipeline == "cascaded":
            pass
        elif self.pipeline == "multi2d":
            pass


if __name__ == '__main__':
    with open('predict_config.json', 'r') as predict_config_file:
        config = json.load(predict_config_file)

    p = Predictor(**config)
    p.predict()
