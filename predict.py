import json
import os
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

from metrics import DiceMetric, ClassWiseDiceMetric
from readers import NIIReader, NPYReader
from cropper import Cropper


def load_predictor(predict_config, train_config):
    """Return the correct type of Predictor class for the given model type."""
    if train_config['model'] in ['UNet3D']:
        p = Predictor3D(**predict_config, train_config=train_config)
    elif train_config['model'] in ['UNet3DShallow']:
        p = Predictor3DShallow(**predict_config, train_config=train_config)
    elif train_config['model'] in ['CascadedUNet3D']:
        p = Predictor3DCascaded(**predict_config, train_config=train_config)
    else:
        p = Predictor2D(**predict_config, train_config=train_config)

    return p


class __Predictor:
    def __init__(self, data_path, dataset, train_config):
        self.rng = np.random.default_rng()

        self.model_name = train_config['model']
        self.image_size = train_config['image_size']
        self.labels_dict = train_config['labels']
        self.combine_labels = train_config['combine_labels']

        self.data_path = data_path
        assert dataset in ["train", "val", "test"], f"dataset must be one of: 'train', 'val', 'test'; but got {dataset}"
        self.dataset = dataset

        self.use_cropper = train_config['use_cropper']
        if self.use_cropper:
            self.cropper = Cropper(data_path, dataset, train_config['use_cropper'])

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

    def get_one_hot(self, y_true, y_pred):
        """Get one-hot encoding representations of labels and predictions."""
        if self.combine_labels:
            y_true = tf.one_hot(y_true, len(self.combine_labels), dtype=np.int8).numpy()
            y_pred = tf.one_hot(y_pred, len(self.combine_labels), dtype=np.int8).numpy()
        else:
            y_true = tf.one_hot(y_true, len(self.labels_dict), dtype=np.int8).numpy()
            y_pred = tf.one_hot(y_pred, len(self.labels_dict), dtype=np.int8).numpy()

        return y_true, y_pred

    def calculate_dice(self, y_true, y_pred):
        """Computes dice coefficient between gt and predicted segmentations using numpy."""
        y_true, y_pred = self.get_one_hot(y_true, y_pred)

        numerator = 2 * np.sum(y_true * y_pred) + 1e-7
        denominator = np.sum(y_true) + np.sum(y_pred) + 1e-7

        return numerator / denominator

    def calculate_class_wise_dice(self, y_true, y_pred):
        """Computes dice coefficient for each class using numpy."""
        y_true, y_pred = self.get_one_hot(y_true, y_pred)
        length = len(self.combine_labels) if self.combine_labels else len(self.labels_dict)
        dices = np.zeros([length])

        # Iterate over all the classes, getting the dice score
        for i in range(length):
            numerator = 2 * np.sum(y_true[..., i] * y_pred[..., i]) + 1e-7
            denominator = np.sum(y_true[..., i]) + np.sum(y_pred[..., i]) + 1e-7
            dices[i] = numerator / denominator

        return dices

    def predict(self, fname=None, display=False):
        raise NotImplementedError

    def display(self, image, label, pred_label, plane='transverse'):
        """
        Should show the label vs. prediction, label vs. image and prediction vs. image in a single scrollable view.
        """
        self.reader.scroll_view(np.concatenate((label, pred_label)), plane=plane)


class Predictor3D(__Predictor):
    def __init__(self, data_path, dataset, model_path, train_config):
        super().__init__(data_path, dataset, train_config)
        self.data_path = os.path.join(data_path, '3D', self.dataset)

        self.reader = NIIReader()
        self.image_fnames = [os.path.join(self.data_path, x, f'{x}_SAX.nii.gz') for x in
                             sorted(os.listdir(self.data_path))]
        self.label_fnames = [os.path.join(self.data_path, x, f'{x}_SAX_mask2.nii.gz') for x in
                             sorted(os.listdir(self.data_path))]

        self.model = self.load_model(model_path)
        self.dimensionality = '3D'

    def load_image_label(self, fname):
        """ Loads the image and label files. """
        # Define paths
        if fname is None:
            idx = self.rng.integers(0, len(self.image_fnames))
            image_path = self.image_fnames[idx]
            label_path = self.label_fnames[idx]
            suffix = image_path.split('/')[-1].split('.')[0][:12]
            print(image_path)
        else:
            suffix = fname.split("/")[-1]
            image_path = os.path.join(fname, f'{suffix}_SAX.nii.gz')
            label_path = os.path.join(fname, f'{suffix}_SAX_mask2.nii.gz')

        # Load image and label
        image = self.reader.read(image_path)
        label = self.reader.read(label_path)

        if self.use_cropper:
            image = self.cropper.crop(image, suffix)
            label = self.cropper.crop(label, suffix)

        # Set to the correct dimensions
        if image.shape != self.image_size:
            image = self.reader.resize(image, self.image_size)
        if label.shape != self.image_size:
            label = self.reader.resize(label, self.image_size, interpolation_order=0)

        image = self.reader.normalize(image)

        # Set to the correct rank
        image = image[np.newaxis, ..., np.newaxis]

        return image, label

    def predict(self, fname=None, display=False):
        image, label = self.load_image_label(fname)
        pred_label = self.model.predict(image)
        pred_label = np.squeeze(np.argmax(pred_label, axis=-1))

        if display:
            print(self.calculate_dice(label, pred_label))
            self.display(image, label, pred_label)

        return image, label, pred_label


class Predictor2D(__Predictor):
    def __init__(self, data_path, dataset, model_path, train_config):
        super().__init__(data_path, dataset, train_config)
        self.data_path = os.path.join(data_path, '2D', self.dataset, train_config["plane"])

        self.reader = NPYReader()
        self.image_fnames = [os.path.join(self.data_path, x) for x in sorted(os.listdir(self.data_path))]
        self.label_fnames = [os.path.join(self.data_path, x) for x in sorted(os.listdir(self.data_path))]

        self.model = self.load_model(model_path)
        self.dimensionality = '2D'

    def load_image_label(self, fname):
        """ Loads the image and label files. """
        # Define paths
        if fname is None:
            idx = self.rng.integers(0, len(self.image_fnames))
            image_folder = self.image_fnames[idx]
            label_folder = self.label_fnames[idx]
        else:
            image_folder = fname
            label_folder = fname
        suffix = image_folder.split("/")[-1]

        # Load image and label
        image_paths = [os.path.join(self.data_path, image_folder, x)
                       for x in sorted(os.listdir(image_folder)) if 'image' in x]
        label_paths = [os.path.join(self.data_path, label_folder, x)
                       for x in sorted(os.listdir(label_folder)) if 'label' in x]

        if not image_paths:
            raise AttributeError(f"No images found at {image_folder}")
        if not label_paths:
            raise AttributeError(f"No labels found at {label_folder}")

        images = np.empty([len(image_paths), *self.image_size, 1], dtype=np.float32)
        labels = np.empty([len(label_paths), *self.image_size], dtype=np.int8)

        for i, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
            image = self.reader.read(image_path)
            label = self.reader.read(label_path)

            if self.use_cropper:
                image = self.cropper.crop(image, suffix)
                label = self.cropper.crop(label, suffix)

            # Set to the correct dimensions
            if image.shape != self.image_size:
                image = self.reader.resize(image, self.image_size)
            if label.shape != self.image_size:
                label = self.reader.resize(label, self.image_size, interpolation_order=0)

            image = self.reader.normalize(image)

            # Set to the correct rank
            image = image[..., np.newaxis]
            images[i, ...] = image
            labels[i, ...] = label

        return images, labels

    def predict(self, fname=None, display=False):
        images, labels = self.load_image_label(fname)
        pred_label = np.empty(images.shape[:-1], dtype=np.int8)

        for i in range(images.shape[0]):
            curr_pred = self.model.predict(images[i, np.newaxis, ...])
            pred_label[i, ...] = np.squeeze(np.argmax(curr_pred, axis=-1))

        image = np.moveaxis(np.squeeze(images), [0, 1, 2], [2, 0, 1])
        label = np.moveaxis(labels, [0, 1, 2], [2, 0, 1])
        pred_label = np.moveaxis(pred_label, [0, 1, 2], [2, 0, 1])

        if display:
            print(self.calculate_dice(label, pred_label))
            self.display(image, label, pred_label)

        return image, label, pred_label

    def predict_logits(self, fname=None):
        """Return logits only rather than the softmax output."""
        images, labels = self.load_image_label(fname)
        if self.combine_labels:
            pred_logits = np.empty(list(images.shape[:-1]) + [len(self.combine_labels)], dtype=np.int8)
        else:
            pred_logits = np.empty(list(images.shape[:-1]) + [len(self.labels_dict)], dtype=np.int8)

        # Create a new model which pulls out the logits before the softmax activation at the end
        pred_model = tf.keras.models.Model(inputs=self.model.inputs, outputs=self.model.layers[-2].output)

        for i in range(images.shape[0]):
            curr_pred = pred_model.predict(images[i, np.newaxis, ...])
            pred_logits[i, ...] = np.squeeze(curr_pred)

        image = np.moveaxis(np.squeeze(images), [0, 1, 2], [2, 0, 1])
        label = np.moveaxis(labels, [0, 1, 2], [2, 0, 1])
        pred_logits = np.moveaxis(pred_logits, [0, 1, 2], [2, 0, 1])

        return image, label, pred_logits


class Predictor3DShallow(__Predictor):
    def __init__(self, data_path, dataset, model_path, train_config):
        super().__init__(data_path, dataset, train_config)
        self.data_path = os.path.join(data_path, '3DShallow', self.dataset, train_config["plane"])

        self.reader = NPYReader()
        self.image_fnames = [os.path.join(self.data_path, x) for x in sorted(os.listdir(self.data_path))]
        self.label_fnames = [os.path.join(self.data_path, x) for x in sorted(os.listdir(self.data_path))]

        self.model = self.load_model(model_path)
        self.dimensionality = '3DShallow'

    def load_image_label(self, fname):
        """ Loads the image and label files. """
        # Define paths
        if fname is None:
            idx = self.rng.integers(0, len(self.image_fnames))
            image_folder = self.image_fnames[idx]
            label_folder = self.label_fnames[idx]
        else:
            image_folder = fname
            label_folder = fname
        suffix = image_folder.split("/")[-1]

        # Load image and label
        image_paths = [os.path.join(self.data_path, image_folder, x)
                       for x in sorted(os.listdir(image_folder)) if 'image' in x]
        label_paths = [os.path.join(self.data_path, label_folder, x)
                       for x in sorted(os.listdir(label_folder)) if 'label' in x]

        images = []
        labels = []

        for i, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
            image = self.reader.read(image_path)
            label = self.reader.read(label_path)

            if self.use_cropper:
                image = self.cropper.crop(image, suffix)
                label = self.cropper.crop(label, suffix)

            # Set to the correct dimensions
            if image.shape != self.image_size:
                image = self.reader.resize(image, self.image_size)
            if label.shape != self.image_size:
                label = self.reader.resize(label, self.image_size, interpolation_order=0)

            image = self.reader.normalize(image)

            # Set to the correct rank
            image = image[..., np.newaxis]

            images += [image]
            labels += [label]

        return images, labels

    def construct_slice_wise(self, f_list, image_depth, slice_depth):
        """Given the images or labels list, re-construct the image/label slice by slice to the full tensor."""
        f = np.empty([*self.image_size[:2], image_depth + slice_depth - 1], dtype=np.float32)
        j = 1
        for i in range(f.shape[-1]):
            if i < image_depth:
                f[..., i] = np.squeeze(f_list[i])[..., 0]
            else:
                f[..., i] = np.squeeze(f_list[image_depth - 1])[..., j]
                j += 1

        return f

    def aggregate_slice_logits(self, pred_logits, image_depth, slice_depth=3):
        """Combine logits by averaging and re-construct a full 3D predicted label from slices."""
        if slice_depth % 2 != 1:
            raise AttributeError(f'slice_depth must be an odd int currently - but received {slice_depth}')

        # Initialise with empty array
        pred_label = np.empty([*self.image_size[:2], image_depth + slice_depth - 1], dtype=np.int8)

        # Construct the predicted label slice-wise
        all_ims = list(range(image_depth))
        end_img_idx = 0

        for i in range(image_depth + slice_depth - 1):
            start_img_idx = max(i - slice_depth + 1, 0)
            if i < image_depth:
                end_img_idx += 1

            # Get the pred label outputs to be referenced for the current slice
            curr_ims = all_ims[start_img_idx:end_img_idx]

            # Get the slice index of each pred label output for the current slice
            pairs = []
            if i < image_depth:
                for j, k in enumerate(reversed(range(len(curr_ims)))):
                    pairs += [[curr_ims[j], k]]
            else:
                for j, k in enumerate(reversed(range(slice_depth - len(curr_ims), slice_depth))):
                    pairs += [[curr_ims[j], k]]

            # Pull the logits from the correct predicted logits image and slice
            curr_slices = [pred_logits[j][..., k, :] for (j, k) in pairs]

            # Get the mean of the logits
            curr_logits = np.zeros([*curr_slices[0].shape])
            for s in curr_slices:
                curr_logits += s
            curr_logits /= float(len(curr_slices))

            # Apply softmax and argmax to the logits to get the predicted labels for the current slice
            pred_label[..., i] = np.argmax(softmax(curr_logits, axis=-1), axis=-1)

        return pred_label

    def predict(self, fname=None, display=False):
        """Return logits only rather than the softmax output."""
        images, labels = self.load_image_label(fname)
        pred_logits = []

        # Create a new model which pulls out the logits before the softmax activation at the end
        pred_model = tf.keras.models.Model(inputs=self.model.inputs, outputs=self.model.layers[-2].output)

        for image in images:
            curr_pred = pred_model.predict(image[np.newaxis, ...])
            pred_logits += [np.squeeze(curr_pred)]

        # Re-generate the image and labels slice-wise
        image = self.construct_slice_wise(images, len(images), self.image_size[-1])
        label = self.construct_slice_wise(labels, len(images), self.image_size[-1])
        pred_label = self.aggregate_slice_logits(pred_logits, len(images), self.image_size[-1])

        if display:
            print(self.calculate_dice(label, pred_label))
            self.display(image, label, pred_label)

        return image, label, pred_label


class Predictor3DCascaded(Predictor3D):
    def __init__(self, data_path, dataset, model_path, train_config):
        super().__init__(data_path, dataset, model_path, train_config)

    def predict(self, fname=None, display=False):
        image, label = self.load_image_label(fname)
        # Get the multiple outputs from the model
        predictions = self.model.predict(image)

        # Convert to the correct dimensionality and shift along to take all 8 values
        general = np.squeeze(np.argmax(predictions[0], axis=-1))
        general = np.where(np.greater_equal(general, 3), general + 1, general)
        pred_label = np.where(np.equal(general, 6), 7, general)

        # Add in scar predictions
        pred_label = np.where(np.greater_equal(np.squeeze(predictions[1]), 0.5), 3, pred_label)

        # And add in papillary muscle predictions
        pred_label = np.where(np.greater_equal(np.squeeze(predictions[2]), 0.5), 6, pred_label)

        if display:
            print(self.calculate_dice(label, pred_label))
            self.display(image, label, pred_label)

        return image, label, pred_label

    def display(self, image, label, pred_label, plane='transverse'):
        """
        Should show the label vs. prediction, label vs. image and prediction vs. image in a single scrollable view.
        """
        self.reader.scroll_view(np.concatenate((label, pred_label)), plane=plane)


if __name__ == '__main__':
    with open('predict_config.json', 'r') as predict_config_file:
        predict_config = json.load(predict_config_file)

    with open(os.path.join(predict_config['model_path'], 'train_config.json'), 'r') as train_config_file:
        train_config = json.load(train_config_file)

    p = load_predictor(predict_config, train_config)
    p.predict(display=True)
