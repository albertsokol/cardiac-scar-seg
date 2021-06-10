import json
import os
import tensorflow as tf

from losses import SoftmaxLoss, WeightedSoftmaxLoss, DiceLoss, WeightedSoftmaxDiceLoss
from readers import NIIReader, NPYReader


class Predictor:
    def __init__(self, model_path, data_path, dataset):
        self.loss_dict = {
            'softmax': SoftmaxLoss,
            'weighted softmax': WeightedSoftmaxLoss,
            'dice': DiceLoss,
            'weighted softmax dice': WeightedSoftmaxDiceLoss
        }

        self.image_size = None
        self.model_name = None
        self.dims = None
        self.reader = None

        self.model = self.load_model(model_path)

        self.data_path = data_path
        assert dataset in ["train", "val", "test"], f"dataset must be one of: 'train', 'val', 'test'; but got {dataset}"
        self.dataset = dataset

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
            self.reader = NIIReader(slice_20=train_config['slice_20'])
        else:
            plane = train_config['plane']
            self.dims = '2D'
            self.dataset = f'{self.dataset}/{plane}'
            self.reader = NPYReader()

        # Might get an error about custom objects - will need to include the losses and metrics if so
        return tf.keras.models.load_model(model_path)

    def load_image_label(self, fname):
        """ Loads the image and label files. """
        image_path = os.path.join(self.data_path, self.dims, self.dataset, 'image', fname)
        label_path = os.path.join(self.data_path, self.dims, self.dataset, 'label', fname)

    def display(self, plane='transverse'):
        """
        Should show the label vs. prediction, label vs. image and prediction vs. image in a single scrollable view.
        """

    def predict(self, fname=None):
        # self.seg_pred = self.seg.predict(self.img_grayscale)
        pass


if __name__ == '__main__':
    with open('predict_config.json', 'r') as predict_config_file:
        config = json.load(predict_config_file)

    p = Predictor(**config)
