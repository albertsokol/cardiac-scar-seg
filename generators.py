import os

import numpy as np
from tensorflow import one_hot
from tensorflow.keras.utils import Sequence

from readers import NIIReader, NPYReader


class __Generator(Sequence):
    def __init__(self, image_path, label_path, batch_size, image_size, labels, shuffle, augmenter):
        # Set up image filenames and indexing
        self.augmenter = augmenter
        self.image_path = image_path
        self.label_path = label_path
        self.image_fnames = [os.path.join(image_path, x) for x in sorted(os.listdir(image_path))]
        self.label_fnames = [os.path.join(label_path, x) for x in sorted(os.listdir(label_path))]
        assert len(self.image_fnames) == len(self.label_fnames), "Number of image files and label files did not match!"
        self.index = np.arange(len(self.image_fnames))

        # Model parameters
        self.batch_size = batch_size
        self.image_size = image_size
        self.labels = labels

        # Shuffle the data before starting if shuffling has been turned on
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        # Optionally shuffle the data at the end of each epoch
        if self.shuffle:
            np.random.shuffle(self.index)

    def __len__(self):
        # Number of gradient descent steps that will be taken per epoch
        return len(self.image_fnames) // self.batch_size

    def __getitem__(self, index, weight_mode=False):
        # Create a list of batch_size numerical indices
        indices = self.index[self.batch_size * index:self.batch_size * (index + 1)]
        if indices.size == 0:
            raise IndexError('Index not within possible range (0 to number of training steps)')
        # Generate the data
        x, y = self.get_data(indices, weight_mode)
        return x, y

    def get_data(self, batch_indices, weight_mode):
        # Initialise empty arrays for the training data and labels
        x = np.empty([self.batch_size, *self.image_size, 1], dtype=np.float32)
        y = np.empty([self.batch_size, *self.image_size, len(self.labels)], dtype=np.int8)

        # Get the training data
        for i, index in enumerate(batch_indices):
            img = self.read_file(index, self.image_fnames)
            label = self.read_file(index, self.label_fnames)
            img = self.reader.normalize(img)

            # Apply data augmentation if option is turned on
            if self.augmenter and not weight_mode:
                img, label = self.augmenter.augment(img, label)

            x[i, ...] = self.prepare_img(img)
            y[i, ...] = self.prepare_label(label)

        return x, y

    def read_file(self, index, fname_list):
        """ Read the file at the given index of the given list. """
        return self.reader.read(fname_list[index])

    def prepare_img(self, img):
        """ Set the image to the correct size and dimensions for placement into the input tensor. """
        # Resize to the input shape of the model
        if img.shape != self.image_size:
            img = self.reader.resize(img, self.image_size)

        img = img[..., np.newaxis]

        return self.reader.normalize(img)

    def prepare_label(self, label):
        """ Set the label to the correct size and dimensions for placement into the ground truth tensor. """
        # Resize to the input shape of the model without interpolation
        if label.shape != self.image_size:
            label = self.reader.resize(label, self.image_size, interpolation_order=0)

        # Any labels which should not be included will be set as background here - just hack for now until RV sorted
        if len(self.labels) != 5:
            key_ints = [int(x) for x in list(self.labels.keys())]
            label = np.where(np.isin(label, key_ints), label, 0.)
            label = np.where(label == 2., 1., label)
            label = np.where(label == 3., 2., label)
            label = np.where(label == 4., 3., label)

        # One hot encode the labels to create a new channel for each label and save as int8 to save space
        return one_hot(label, len(self.labels), dtype=np.int8).numpy()


class Generator3D(__Generator):
    """ Class for the 3D image and label generator. """
    def __init__(self, image_path, label_path, batch_size, image_size, labels, slice_20, shuffle=True, augmenter=None):
        super().__init__(image_path, label_path, batch_size, image_size, labels, shuffle, augmenter)
        self.reader = NIIReader(slice_20=slice_20)


class Generator2D(__Generator):
    """ Class for the 2D image and label generator. """
    def __init__(self, image_path, label_path, batch_size, image_size, labels, slice_20, shuffle=True, augmenter=None):
        super().__init__(image_path, label_path, batch_size, image_size, labels, shuffle, augmenter)
        self.reader = NPYReader()

