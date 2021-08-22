import json
import os
import random

import cv2
import numpy as np
import tensorflow as tf
from scipy.special import softmax

from cropper import Cropper
from masker import Masker
from metrics import DiceMetric, ClassWiseDiceMetric
from readers import NIIReader, NPYReader


def load_predictor(predict_config):
    """Return the correct type of Predictor class for the given model type."""
    if isinstance(predict_config['model_path'], str):
        # Load the training config from the predict config
        with open(os.path.join(predict_config['model_path'], 'train_config.json'), 'r') as train_config_file:
            train_config = json.load(train_config_file)
        # Then load the model
        if train_config['model'] in ['UNet3D', 'VNet']:
            p = Predictor3D(**predict_config, train_config=train_config)
        elif train_config['model'] in ['UNet3DShallow', 'VNetShallow']:
            p = Predictor3DShallow(**predict_config, train_config=train_config)
        elif train_config['model'] in ['CascadedUNet3DShallowB']:
            p = Predictor3DCascadedShallowB(**predict_config, train_config=train_config)
        elif train_config['model'] in ['CascadedUNet3DShallowC']:
            p = Predictor3DCascadedShallowC(**predict_config, train_config=train_config)
        else:
            p = Predictor2D(**predict_config, train_config=train_config)
    elif isinstance(predict_config['model_path'], list):
        # Otherwise, if a list, then build a graph of Predictors to run sequentially for staggered cascading
        p = PredictorStaggeredCascaded(**predict_config)
    else:
        raise AttributeError(f"predict_config.json model_path must be str or list, but got {type(predict_config['model_path'])}")

    return p


class __Predictor:
    def __init__(self, data_path, dataset, train_config, post_process):
        self.rng = np.random.default_rng()

        if post_process:
            assert post_process in ["erosion dilation", "dae"], "post_process must either be false of one of: " \
                                                                "'erosion dilation', 'dae'"
        else:
            assert isinstance(post_process, bool), "post_process must be the false bool if not in use"
        self.post_process = post_process

        self.model_path = train_config['model_save_path']
        self.model_name = train_config['model']
        self.image_size = train_config['image_size']
        self.labels_dict = train_config['labels']
        self.label_indices = {self.labels_dict[k]: int(k) for k in self.labels_dict}
        self.combine_labels = train_config['combine_labels']
        self.cascade = {} if not train_config['cascade'] else train_config['cascade']

        self.quality_weighted_mode = True if 'quality' in train_config['loss_fn'] else False

        self.dimensionality = None
        self.plane = train_config["plane"]

        self.data_path = data_path
        assert dataset in ["train", "val", "test"], f"dataset must be one of: 'train', 'val', 'test'; but got {dataset}"
        self.dataset = dataset

        self.use_cropper = train_config['use_cropper']
        if self.use_cropper:
            self.cropper = Cropper(data_path, dataset, train_config['use_cropper'])

        # If masked images are required and do not exist, then create them here
        if self.cascade:
            Masker(
                **self.cascade,
                data_path=self.data_path,
                dataset=self.dataset,
                folder=self.model_path,
                plane=self.plane,
            ).create_masks()

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

    def apply_label_combine(self, label):
        """Combine each list of labels in self.combine_labels into a single label."""
        out = np.zeros([*label.shape, len(self.combine_labels)], dtype=np.int8)

        # Iterate over each list in combine_labels and update the zeros vector to 1 where that label is present
        for i, combo in enumerate(self.combine_labels):
            idxs = np.array([self.label_indices[x] for x in combo])
            out[..., i] = np.where(np.isin(label, idxs), 1, 0)

        return np.argmax(out, axis=-1)

    def get_one_hot(self, y_true, y_pred=None):
        """Get one-hot encoding representations of labels and predictions."""
        if self.combine_labels:
            y_true = tf.one_hot(y_true, len(self.combine_labels), dtype=np.int8).numpy()
            if y_pred is not None:
                y_pred = tf.one_hot(y_pred, len(self.combine_labels), dtype=np.int8).numpy()
        else:
            y_true = tf.one_hot(y_true, len(self.labels_dict), dtype=np.int8).numpy()
            if y_pred is not None:
                y_pred = tf.one_hot(y_pred, len(self.labels_dict), dtype=np.int8).numpy()

        if y_pred is None:
            return y_true
        else:
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

    def predict(self, fname=None, display=False, apply_combine=True, return_fname=False):
        raise NotImplementedError

    def post_process_erosion_dilation(self, pred_label, kernel_size=2):
        """Perform slice-wise erosion-dilation based smoothing and noise reduction of the given predicted label."""
        kernel = np.ones([kernel_size, kernel_size], dtype=np.int8)
        out = np.empty(pred_label.shape, dtype=np.int8)

        for i in range(pred_label.shape[-1]):
            curr = pred_label[..., i].astype(np.uint8)

            # Convert to one-hot encoding to allow label-wise erosion dilation
            curr = self.get_one_hot(curr).astype(np.uint8)

            for j in range(curr.shape[-1]):
                # Skip the first label as this will simply be background
                if j == 0:
                    continue

                # Erode and dilate the predicted areas using the kernel to get rid of noise smaller than kernel_size
                curr[..., j] = cv2.erode(curr[..., j], kernel, iterations=1)
                curr[..., j] = cv2.dilate(curr[..., j], kernel, iterations=1)

            # argmax will automatically set to 0 if all values are 0
            curr = np.argmax(curr, axis=-1)
            out[..., i] = curr

        return out

    def post_process_label(self, pred_label):
        """Route to the correct post-processing function and return the updated predicted label."""
        if self.post_process == "erosion dilation":
            pred_label = self.post_process_erosion_dilation(pred_label)
        elif self.post_process == "dae":
            pred_label = self.post_process_dae(pred_label)

        return pred_label

    def display(self, image, label, pred_label, plane='transverse'):
        """
        Should show the label vs. prediction, label vs. image and prediction vs. image in a single scrollable view.
        """
        self.reader.scroll_view(np.concatenate((label, pred_label)), plane=plane)

    @staticmethod
    def __check_image_label_paths(image_paths, image_folder, label_paths, label_folder):
        """Check that the image and label paths for a given scan are correct."""
        if not image_paths:
            raise AttributeError(f"No images found at {image_folder}")
        if not label_paths:
            raise AttributeError(f"No labels found at {label_folder}")
        assert len(image_paths) == len(label_paths), \
            f"image and label size mismatch: image length {len(image_paths)}, label length {len(label_paths)}"

    def _get_folder_paths(self, fname):
        """Get the correct image folder and label folder name plus the correct suffix for loading."""
        # Define paths - should be the path to the folder containing the images
        if fname is None:
            idx = self.rng.integers(0, len(self.image_fnames))
            image_folder = self.image_fnames[idx]
            label_folder = self.label_fnames[idx]
        else:
            image_folder = fname
            label_folder = os.path.join(self.data_path, self.dimensionality, self.dataset, self.plane, fname.split("/")[-1])

        suffix = image_folder.split("/")[-1]

        if fname is None:
            return image_folder, label_folder, suffix, self.image_fnames[idx]
        else:
            return image_folder, label_folder, suffix, fname

    def _get_image_label_paths(self, image_folder, label_folder, suffix):
        """Get the paths to all components of a scan and label (the image and label sub-parts)."""
        if not self.cascade:
            image_paths = [os.path.join(image_folder, x)
                           for x in sorted(os.listdir(image_folder)) if '_image.' in x]
        else:
            image_paths = [os.path.join(self.model_path, 'mask', self.dataset, suffix, x)
                           for x in sorted(os.listdir(os.path.join(
                                self.model_path, 'mask', self.dataset, suffix
                           ))) if '_image.' in x]

        label_paths = [os.path.join(label_folder, x)
                       for x in sorted(os.listdir(label_folder)) if '_label.' in x]

        self.__check_image_label_paths(image_paths, image_folder, label_paths, label_folder)

        return image_paths, label_paths

    def _prepare_image_label(self, image, label, suffix):
        """Get image and label ready for down-stream predictions."""
        if self.use_cropper:
            if not self.cascade:
                image = self.cropper.crop(image, suffix)
            label = self.cropper.crop(label, suffix)

        # Set to the correct dimensions
        if image.shape != self.image_size:
            image = self.reader.resize(image, self.image_size)
        if label.shape != self.image_size:
            label = self.reader.resize(label, self.image_size, interpolation_order=0)

        image = self.reader.normalize(image)

        return image, label


class Predictor3D(__Predictor):
    def __init__(self, data_path, dataset, model_path, train_config, post_process):
        super().__init__(data_path, dataset, train_config, post_process)
        self.full_data_path = os.path.join(data_path, '3D', self.dataset)

        self.reader = NIIReader() if not self.cascade else NPYReader()

        # TODO: test this all works ok
        self.image_fnames = [os.path.join(self.full_data_path, x) for x in
                             sorted(os.listdir(self.full_data_path))]
        self.label_fnames = [os.path.join(self.full_data_path, x) for x in
                             sorted(os.listdir(self.full_data_path))]

        self.model = self.load_model(model_path)
        self.dimensionality = '3D'

    def load_image_label(self, fname):
        """ Loads the image and label files. """
        image_folder, label_folder, suffix, fname = self._get_folder_paths(fname)

        # Load image and label
        image = self.reader.read(os.path.join(image_folder, f'{suffix}_SAX.nii.gz'))
        label = self.reader.read(os.path.join(label_folder, f'{suffix}_SAX_mask2.nii.gz'))

        image, label = self._prepare_image_label(image, label, suffix)

        # Set to the correct rank
        image = image[np.newaxis, ..., np.newaxis]

        return image, label, fname

    def predict(self, fname=None, display=False, apply_combine=True, return_fname=False):
        image, label, fname = self.load_image_label(fname)
        if self.quality_weighted_mode:
            pred_label = self.model.predict((image, np.array([1.], dtype=np.float32)))[1]
        else:
            pred_label = self.model.predict(image)

        pred_label = np.squeeze(np.argmax(pred_label, axis=-1))

        if self.combine_labels and apply_combine:
            label = self.apply_label_combine(label)

        if self.post_process:
            pred_label = self.post_process_label(pred_label)

        if display:
            print(self.calculate_dice(label, pred_label))
            self.display(image, label, pred_label)

        if return_fname:
            return image, label, pred_label, fname
        else:
            return image, label, pred_label


class Predictor2D(__Predictor):
    def __init__(self, data_path, dataset, model_path, train_config, post_process):
        super().__init__(data_path, dataset, train_config, post_process)
        self.full_data_path = os.path.join(data_path, '2D', self.dataset, train_config["plane"])

        self.reader = NPYReader()
        if not self.cascade:
            self.image_fnames = [os.path.join(self.full_data_path, x) for x in sorted(os.listdir(self.full_data_path))]
        else:
            self.image_fnames = [os.path.join(self.model_path, 'mask', self.dataset, x)
                                 for x in sorted(os.listdir(os.path.join(self.model_path, 'mask', self.dataset)))]
        self.label_fnames = [os.path.join(self.full_data_path, x) for x in sorted(os.listdir(self.full_data_path))]

        self.model = self.load_model(model_path)
        self.dimensionality = '2D'
        self.plane = train_config['plane']

    def load_image_label(self, fname):
        """ Loads the image and label files. """
        image_folder, label_folder, suffix, fname = self._get_folder_paths(fname)
        image_paths, label_paths = self._get_image_label_paths(image_folder, label_folder, suffix)

        images = np.empty([len(image_paths), *self.image_size, 1], dtype=np.float32)
        labels = np.empty([len(label_paths), *self.image_size], dtype=np.int8)

        for i, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
            image = self.reader.read(image_path)
            label = self.reader.read(label_path)

            image, label = self._prepare_image_label(image, label, suffix)

            # Set to the correct rank
            image = image[..., np.newaxis]
            images[i, ...] = image
            labels[i, ...] = label

        return images, labels, fname

    def predict(self, fname=None, display=False, apply_combine=True, return_fname=False):
        images, labels, fname = self.load_image_label(fname)
        pred_label = np.empty(images.shape[:-1], dtype=np.int8)

        for i in range(images.shape[0]):
            if self.quality_weighted_mode:
                # Get the prediction on the current image with dummy quality weighting - pred is index 1
                curr_pred = self.model.predict((images[i, np.newaxis, ...], np.array([1.], dtype=np.float32)))[1]
            else:
                curr_pred = self.model.predict(images[i, np.newaxis, ...])
            pred_label[i, ...] = np.squeeze(np.argmax(curr_pred, axis=-1))

        image = np.moveaxis(np.squeeze(images), [0, 1, 2], [2, 0, 1])
        label = np.moveaxis(labels, [0, 1, 2], [2, 0, 1])
        pred_label = np.moveaxis(pred_label, [0, 1, 2], [2, 0, 1])

        if self.combine_labels and apply_combine:
            label = self.apply_label_combine(label)

        if self.post_process:
            pred_label = self.post_process_label(pred_label)

        if display:
            print(self.calculate_dice(label, pred_label))
            self.display(image, label, pred_label)

        if return_fname:
            return image, label, pred_label, fname
        else:
            return image, label, pred_label

    def predict_logits(self, fname=None):
        """Return logits only rather than the softmax output."""
        images, labels, fname = self.load_image_label(fname)
        if self.combine_labels:
            pred_logits = np.empty(list(images.shape[:-1]) + [len(self.combine_labels)], dtype=np.int8)
        else:
            pred_logits = np.empty(list(images.shape[:-1]) + [len(self.labels_dict)], dtype=np.int8)

        # Create a new model which pulls out the logits before the softmax activation at the end
        if self.quality_weighted_mode:
            pred_model = tf.keras.models.Model(inputs=self.model.inputs, outputs=self.model.layers[-1].output)
        else:
            pred_model = tf.keras.models.Model(inputs=self.model.inputs, outputs=self.model.layers[-2].output)

        for i in range(images.shape[0]):
            if self.quality_weighted_mode:
                curr_pred = self.model.predict((images[i, np.newaxis, ...], np.array([1.], dtype=np.float32)))[1]
            else:
                curr_pred = pred_model.predict(images[i, np.newaxis, ...])
            pred_logits[i, ...] = np.squeeze(curr_pred)

        image = np.moveaxis(np.squeeze(images), [0, 1, 2], [2, 0, 1])
        label = np.moveaxis(labels, [0, 1, 2], [2, 0, 1])
        pred_logits = np.moveaxis(pred_logits, [0, 1, 2], [2, 0, 1])

        return image, label, pred_logits


class Predictor3DShallow(__Predictor):
    def __init__(self, data_path, dataset, model_path, train_config, post_process):
        super().__init__(data_path, dataset, train_config, post_process)
        self.full_data_path = os.path.join(data_path, '3DShallow', self.dataset, train_config["plane"])

        self.reader = NPYReader()
        if not self.cascade:
            self.image_fnames = [os.path.join(self.full_data_path, x) for x in sorted(os.listdir(self.full_data_path))]
        else:
            self.image_fnames = [os.path.join(self.model_path, 'mask', self.dataset, x)
                                 for x in sorted(os.listdir(os.path.join(self.model_path, 'mask', self.dataset)))]
        self.label_fnames = [os.path.join(self.full_data_path, x) for x in sorted(os.listdir(self.full_data_path))]

        self.model = self.load_model(model_path)
        self.dimensionality = '3DShallow'
        self.plane = train_config['plane']

    def load_image_label(self, fname):
        """ Loads the image and label files. """
        image_folder, label_folder, suffix, fname = self._get_folder_paths(fname)
        image_paths, label_paths = self._get_image_label_paths(image_folder, label_folder, suffix)
        images, labels = [], []

        for i, (image_path, label_path) in enumerate(zip(image_paths, label_paths)):
            image = self.reader.read(image_path)
            label = self.reader.read(label_path)

            image, label = self._prepare_image_label(image, label, suffix)

            # Set to the correct rank
            image = image[..., np.newaxis]

            images += [image]
            labels += [label]

        return images, labels, fname

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
            if curr_logits.shape[-1] == 1:
                # Sigmoid (binary case) - just use z > 0 since don't need sigmoid function for prediction
                pred_label[..., i] = np.where(np.greater_equal(curr_logits[..., 0], 0), 1, 0)
            else:
                # Softmax
                pred_label[..., i] = np.argmax(softmax(curr_logits, axis=-1), axis=-1)

        return pred_label

    def predict(self, fname=None, display=False, apply_combine=True, return_fname=False):
        """Return logits only rather than the softmax output."""
        images, labels, fname = self.load_image_label(fname)
        pred_logits = []

        # Create a new model which pulls out the logits before the softmax activation at the end
        if self.quality_weighted_mode:
            pred_model = tf.keras.models.Model(inputs=self.model.inputs, outputs=self.model.layers[-1].output)
        else:
            pred_model = tf.keras.models.Model(inputs=self.model.inputs, outputs=self.model.layers[-2].output)

        for image in images:
            if self.quality_weighted_mode:
                curr_pred = pred_model.predict((image[np.newaxis, ...], np.array([self.image_size[-1]], dtype=np.float32)))
            else:
                curr_pred = pred_model.predict(image[np.newaxis, ...])
            pred_logits += [np.squeeze(curr_pred)]

        # Re-generate the image and labels slice-wise
        image = self.construct_slice_wise(images, len(images), self.image_size[-1])
        label = self.construct_slice_wise(labels, len(images), self.image_size[-1])
        pred_label = self.aggregate_slice_logits(pred_logits, len(images), self.image_size[-1])

        if self.combine_labels and apply_combine:
            label = self.apply_label_combine(label)

        if self.post_process:
            pred_label = self.post_process_label(pred_label)

        if display:
            print(self.calculate_dice(label, pred_label))
            self.display(image, label, pred_label)

        if return_fname:
            return image, label, pred_label, fname
        else:
            return image, label, pred_label


class PredictorNode:
    """Node that forms part of the PredictorStaggeredCascaded tree."""
    def __init__(self, model_path, combine_labels, keep_labels=None):
        self.model_path = model_path
        self.combine_labels = combine_labels
        self.keep_labels = keep_labels
        self.outputs = []
        self.output_idx = []
        self.children = []

    def __repr__(self, level=0):
        s = f"\t" * level + f"path: {self.model_path}, combine_labels: {self.combine_labels}, keep_labels: " \
                            f"{self.keep_labels} outputs: {self.outputs}, output_idx: {self.output_idx}\n"
        for child in self.children:
            s += child.__repr__(level + 1)
        return s

    def add_node(self, node):
        self.children += [node]


class PredictorStaggeredCascaded:
    """
    Predictor class for handling staggered cascaded models, combining all nested Predictors to form a single output.
    Builds a generic tree structure to define execution order of models and determine which models are responsible
    for each output label automatically.
    """
    def __init__(self, data_path, dataset, model_path, post_process):
        self.data_path = data_path
        self.dataset = dataset
        self.model_paths = model_path
        self.post_process = post_process
        self.tree = None
        self.node_map = {}
        self.label_map = {}
        self.model_cache = {}

        with open(os.path.join(self.model_paths[0], 'train_config.json'), 'r') as f:
            init_config = json.load(f)

        # Attributes to enable compatability with performance.py
        self.plane = init_config["plane"]
        self.model_name = init_config["model"]
        self.combine_labels = False
        self.labels_dict = init_config["labels"]

    def get_one_hot(self, y_true, y_pred):
        """Get one-hot encoding representations of labels and predictions."""
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
        length = len(self.labels_dict)
        dices = np.zeros([length])

        # Iterate over all the classes, getting the dice score
        for i in range(length):
            numerator = 2 * np.sum(y_true[..., i] * y_pred[..., i]) + 1e-7
            denominator = np.sum(y_true[..., i]) + np.sum(y_pred[..., i]) + 1e-7
            dices[i] = numerator / denominator

        return dices

    def dfs_add(self, model_path, combine_labels, keep_labels, parent, node):
        """Search the tree for a node with the named parent, and add the curr as its child."""
        if parent == node.model_path:
            node.add_node(PredictorNode(model_path, combine_labels, keep_labels))
            return True
        if not node.children:
            return
        else:
            for child in node.children:
                added = self.dfs_add(model_path, combine_labels, keep_labels, parent, child)
                if added is not None:
                    return True

    def dfs_update(self, k, node):
        """
        Traverse all nodes in the tree, updating the node map with the model for the shortest list for the current k.
        """
        curr_len = [len(x) for x in node.combine_labels if k in x][0]
        if curr_len < self.node_map[k]['len']:
            self.node_map[k]['node'] = node
            self.node_map[k]['len'] = curr_len
        if not node.children:
            return
        else:
            for child in node.children:
                self.dfs_update(k, child)

    def bfs_run(self, tree, fname, display):
        """Run all models in tree in breadth-first manner to ensure masks are available for downstream models."""
        queue = [tree]
        pred_label = None

        while len(queue) != 0:
            node = queue.pop(0)

            # Run predictions with node, which will be root tree on first iter
            if node not in self.model_cache:
                # Load models into the cache to prevent re-loading on every predicted image
                self.model_cache[node] = load_predictor({
                    'model_path': node.model_path,
                    'data_path': self.data_path,
                    'dataset': self.dataset,
                    'post_process': self.post_process
                })

            if fname is None:
                image, label, curr_pred, fname = self.model_cache[node].predict(fname, display=False, apply_combine=False, return_fname=True)
            else:
                image, label, curr_pred = self.model_cache[node].predict(fname, display=False, apply_combine=False, return_fname=False)

            if pred_label is None:
                pred_label = np.zeros([*curr_pred.shape], dtype=np.int8)

            for k, i in zip(node.outputs, node.output_idx):
                pred_label = np.where(np.equal(curr_pred, i), self.label_map[k], pred_label)

            if node.children:
                queue += node.children

        if display:
            self.model_cache[node].calculate_dice(label, pred_label)
            self.model_cache[node].display(image, label, pred_label)

        return image, label, pred_label

    def construct_tree(self):
        """Build the tree for dynamically combining Predictor objects to eventually form a single output label."""
        # First find the base of the tree; the model which does not have any nested cascading
        tree = None
        for model_path in self.model_paths:
            with open(os.path.join(model_path, 'train_config.json'), 'r') as f:
                curr_train_config = json.load(f)
            if not curr_train_config['cascade']:
                # Identified the root
                tree = PredictorNode(model_path, curr_train_config['combine_labels'])
                self.model_paths.remove(model_path)
                # Also update the self.node_map with all labels in label dict here
                self.node_map = {k: {"node": tree, "len": len(curr_train_config['labels'])} for k in list(curr_train_config['labels'].values())}
                self.label_map = {curr_train_config['labels'][k]: int(k) for k in curr_train_config['labels']}
            break
        assert tree is not None, "No base model found - at least one model in the list should not be cascaded."

        # Now, build the remainder of the tree based on the paths
        i, threshold = 0, 2048
        while self.model_paths:
            i += 1
            if i > threshold:
                raise ValueError("Unable to build tree, exceeded maximum number of tries (2048).")
            model_path = random.choice(self.model_paths)
            with open(os.path.join(model_path, 'train_config.json'), 'r') as f:
                curr_train_config = json.load(f)
            parent = curr_train_config['cascade']['model_path']
            combine_labels = curr_train_config['combine_labels']
            keep_labels = curr_train_config['cascade']['keep_labels']
            added = self.dfs_add(model_path, combine_labels, keep_labels, parent, tree)
            if added is not None:
                self.model_paths.remove(model_path)

        return tree

    def update_node_map(self, tree):
        """Find out which model should predict each label by finding the shortest list in which that label occurs."""
        for k in self.node_map:
            self.dfs_update(k, tree)
            self.node_map[k]['node'].outputs += [k]
            self.node_map[k]['node'].output_idx += [i for i, x in enumerate(self.node_map[k]['node'].combine_labels) if k in x]

    def predict(self, fname=None, display=False):
        if self.tree is None:
            self.tree = self.construct_tree()
            self.update_node_map(self.tree)

        image, label, pred_label = self.bfs_run(self.tree, fname, display)

        if self.post_process:
            pred_label = self.post_process_label(pred_label)

        return image, label, pred_label


class Predictor3DCascadedShallowB(Predictor3DShallow):
    def __init__(self, data_path, dataset, model_path, train_config, post_process):
        super().__init__(data_path, dataset, model_path, train_config, post_process)

    def predict(self, fname=None, display=False, apply_combine=True, return_fname=False):
        images, labels, fname = self.load_image_label(fname)
        image = self.construct_slice_wise(images, len(images), self.image_size[-1])
        label = self.construct_slice_wise(labels, len(images), self.image_size[-1])

        # Get the multiple outputs from the model
        pred_model = tf.keras.models.Model(inputs=self.model.inputs, outputs=[
            self.model.get_layer("conv3d_14").output,
            self.model.get_layer("conv3d_29").output,
            self.model.get_layer("conv3d_44").output,
        ])

        pred_logits_root = []
        pred_logits_scar = []
        pred_logits_pap = []

        for x in images:
            predictions = pred_model.predict(x[np.newaxis, ...])

            pred_logits_root += [np.squeeze(predictions[0], axis=0)]
            pred_logits_scar += [np.squeeze(predictions[1], axis=0)]
            pred_logits_pap += [np.squeeze(predictions[2], axis=0)]

        pred_label_root = self.aggregate_slice_logits(pred_logits_root, len(images), self.image_size[-1])
        pred_label_scar = self.aggregate_slice_logits(pred_logits_scar, len(images), self.image_size[-1])
        pred_label_pap = self.aggregate_slice_logits(pred_logits_pap, len(images), self.image_size[-1])

        pred_label = np.zeros(pred_label_root.shape)
        pred_label = np.where(np.equal(pred_label_root, 1), 1, pred_label)  # l lumen
        pred_label = np.where(np.equal(pred_label_root, 2), 2, pred_label)  # l myo
        pred_label = np.where(np.equal(pred_label_root, 3), 4, pred_label)  # r lumen
        pred_label = np.where(np.equal(pred_label_root, 4), 5, pred_label)  # r lumen
        pred_label = np.where(np.equal(pred_label_root, 5), 7, pred_label)  # aorta
        pred_label = np.where(np.equal(pred_label_scar, 1), 3, pred_label)  # scar
        pred_label = np.where(np.equal(pred_label_pap, 1), 6, pred_label)  # pap

        if self.post_process:
            pred_label = self.post_process_label(pred_label)

        if display:
            print(self.calculate_dice(label, pred_label))
            self.display(image, label, pred_label)

        return image, label, pred_label


class Predictor3DCascadedShallowC(Predictor3DShallow):
    def __init__(self, data_path, dataset, model_path, train_config, post_process):
        super().__init__(data_path, dataset, model_path, train_config, post_process)

    def predict(self, fname=None, display=False, apply_combine=True, return_fname=False):
        images, labels, fname = self.load_image_label(fname)
        image = self.construct_slice_wise(images, len(images), self.image_size[-1])
        label = self.construct_slice_wise(labels, len(images), self.image_size[-1])

        # Get the multiple outputs from the model
        pred_model = tf.keras.models.Model(inputs=self.model.inputs, outputs=[
            self.model.get_layer("conv3d_14").output,  # root out
            self.model.get_layer("conv3d_29").output,  # l myo out
            self.model.get_layer("conv3d_44").output,  # l lumen out
            self.model.get_layer("conv3d_59").output,  # r lumen myo out
            self.model.get_layer("conv3d_74").output,  # scar out
            self.model.get_layer("conv3d_89").output,  # pap out
        ])

        pred_logits_root = []
        pred_logits_l_myo = []
        pred_logits_l_lumen = []
        pred_logits_r_lumen_myo = []
        pred_logits_scar = []
        pred_logits_pap = []

        for x in images:
            predictions = pred_model.predict(x[np.newaxis, ...])

            pred_logits_root += [np.squeeze(predictions[0], axis=0)]
            pred_logits_l_myo += [np.squeeze(predictions[1], axis=0)]
            pred_logits_l_lumen += [np.squeeze(predictions[2], axis=0)]
            pred_logits_r_lumen_myo += [np.squeeze(predictions[3], axis=0)]
            pred_logits_scar += [np.squeeze(predictions[4], axis=0)]
            pred_logits_pap += [np.squeeze(predictions[5], axis=0)]

        pred_label_root = self.aggregate_slice_logits(pred_logits_root, len(images), self.image_size[-1])
        pred_label_l_myo = self.aggregate_slice_logits(pred_logits_l_myo, len(images), self.image_size[-1])
        pred_label_l_lumen = self.aggregate_slice_logits(pred_logits_l_lumen, len(images), self.image_size[-1])
        pred_label_r_lumen_myo = self.aggregate_slice_logits(pred_logits_r_lumen_myo, len(images), self.image_size[-1])
        pred_label_scar = self.aggregate_slice_logits(pred_logits_scar, len(images), self.image_size[-1])
        pred_label_pap = self.aggregate_slice_logits(pred_logits_pap, len(images), self.image_size[-1])

        pred_label = np.zeros(pred_label_root.shape)
        pred_label = np.where(np.equal(pred_label_root, 3), 7, pred_label)  # aorta
        pred_label = np.where(np.equal(pred_label_l_myo, 1), 2, pred_label)  # l myo
        pred_label = np.where(np.equal(pred_label_l_lumen, 1), 1, pred_label)  # l myo
        pred_label = np.where(np.equal(pred_label_r_lumen_myo, 1), 4, pred_label)  # r lumen
        pred_label = np.where(np.equal(pred_label_r_lumen_myo, 2), 5, pred_label)  # r myo
        pred_label = np.where(np.equal(pred_label_scar, 1), 3, pred_label)  # scar
        pred_label = np.where(np.equal(pred_label_pap, 1), 6, pred_label)  # pap

        if self.post_process:
            pred_label = self.post_process_label(pred_label)

        if display:
            print(self.calculate_dice(label, pred_label))
            self.display(image, label, pred_label)

        return image, label, pred_label


if __name__ == '__main__':
    with open('predict_config.json', 'r') as predict_config_file:
        predict_config = json.load(predict_config_file)

    p = load_predictor(predict_config)
    p.predict(display=True)
