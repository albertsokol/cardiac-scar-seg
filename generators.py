import json
import os
from abc import ABC

import numpy as np
from matplotlib import pyplot as plt
from tensorflow import one_hot
from tensorflow.keras.utils import Sequence

from readers import NIIReader, NPYReader
from cropper import Cropper
from masker import Masker


class __Generator(Sequence, ABC):
    def __init__(
        self,
        model_save_path,
        generic_data_path,
        data_path,
        plane,
        batch_size,
        image_size,
        labels,
        dataset,
        shuffle,
        augmenter,
        use_cropper,
        combine_labels,
        cascade,
        quality_weighting_scores,
    ):
        # Set up image filenames and indexing
        self.generic_data_path = generic_data_path  # root of all data folders
        self.data_path = data_path  # model and plane specific path
        self.dataset = dataset
        self.image_fnames = [os.path.join(data_path, x, f'{x}_SAX.nii.gz') for x in sorted(os.listdir(data_path))]
        self.label_fnames = [os.path.join(data_path, x, f'{x}_SAX_mask2.nii.gz') for x in sorted(os.listdir(data_path))]
        assert len(self.image_fnames) == len(self.label_fnames), "Number of image files and label files did not match!"
        self.index = np.arange(len(self.image_fnames))

        # Get the quality weightings dictionary
        if quality_weighting_scores:
            with open('quality_scores.json', 'r') as f:
                self.quality_weightings_dict = json.load(f)
        self.quality_weighting_scores = quality_weighting_scores

        # Image handling: augmentation, cropping and masking
        self.augmenter = augmenter

        self.cascade = cascade
        if cascade:
            Masker(
                **cascade,
                data_path=generic_data_path,
                dataset=dataset,
                folder=model_save_path,
                plane=plane,
            ).create_masks()

        self.cropper = Cropper(generic_data_path, dataset, mode=use_cropper) if use_cropper else False

        # Model parameters
        self.batch_size = batch_size
        self.image_size = image_size
        self.labels = labels
        self.label_indices = {self.labels[k]: int(k) for k in self.labels}
        self.use_cropper = use_cropper
        self.combine_labels = combine_labels

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
        if self.combine_labels:
            y = np.empty([self.batch_size, *self.image_size, len(self.combine_labels)], dtype=np.int8)
        else:
            y = np.empty([self.batch_size, *self.image_size, len(self.labels)], dtype=np.int8)

        if len(self.image_size) == 2:
            quality_weightings = np.zeros([self.batch_size], dtype=np.float32)
        else:
            quality_weightings = np.zeros([self.batch_size, self.image_size[-1]], dtype=np.float32)

        # Get the training data
        for i, index in enumerate(batch_indices):
            img, fname = self.read_file(index, self.image_fnames)
            label, fname = self.read_file(index, self.label_fnames)
            img = self.reader.normalize(img)

            # Apply data augmentation if option is turned on
            if self.augmenter and not weight_mode:
                img, label = self.augmenter.augment(img, label)

            x[i, ...] = self.prepare_img(img, fname)
            y[i, ...] = self.prepare_label(label, fname)
            if self.quality_weighting_scores:
                quality_weightings[i, ...] = self.get_quality_weightings(fname)

        if self.quality_weighting_scores:
            return {'model_in': x, 'qw_in': quality_weightings}, {'m': y, 'qw_out': quality_weightings}
        else:
            return x, y

    def read_file(self, index, fname_list):
        """ Read the file at the given index of the given list. """
        return self.reader.read(fname_list[index]), fname_list[index]

    def prepare_img(self, img, fname):
        """ Set the image to the correct size and dimensions for placement into the input tensor. """
        # If cropping, then crop the image here
        if self.use_cropper and not self.cascade:
            img = self.cropper.crop(img, fname)

        # Resize to the input shape of the model
        if img.shape != self.image_size:
            img = self.reader.resize(img, self.image_size)

        img = img[..., np.newaxis]

        return self.reader.normalize(img)

    def prepare_label(self, label, fname):
        """ Set the label to the correct size and dimensions for placement into the ground truth tensor. """
        # If cropping, then crop the label here
        if self.use_cropper:
            label = self.cropper.crop(label, fname)

        # Resize to the input shape of the model without interpolation
        if label.shape != self.image_size:
            label = self.reader.resize(label, self.image_size, interpolation_order=0)

        # If labels are to be combined, do that here and return a one hot encoded tensor
        if self.combine_labels:
            return self.apply_label_combine(label)

        # One hot encode the labels to create a new channel for each label and save as int8 to save space
        return one_hot(label, len(self.labels), dtype=np.int8).numpy()

    def get_quality_weightings(self, fname):
        """Get the quality weights for the current image slice."""
        raise NotImplementedError

    def apply_label_combine(self, label):
        """Combine each list of labels in self.combine_labels into a single label."""
        out = np.zeros([*label.shape, len(self.combine_labels)], dtype=np.int8)

        # Iterate over each list in combine_labels and update the zeros vector to 1 where that label is present
        for i, combo in enumerate(self.combine_labels):
            idxs = np.array([self.label_indices[x] for x in combo])
            out[..., i] = np.where(np.isin(label, idxs), 1, 0)

        return out


class Generator3D(__Generator):
    """ Class for the 3D image and label generator. """
    def __init__(
        self,
        model_save_path,
        generic_data_path,
        data_path,
        plane,
        batch_size,
        image_size,
        labels,
        dataset,
        shuffle=True,
        augmenter=None,
        use_cropper=False,
        combine_labels=None,
        cascade=None,
        quality_weighting_scores=None,
    ):
        super().__init__(
            model_save_path,
            generic_data_path,
            data_path,
            plane,
            batch_size,
            image_size,
            labels,
            dataset,
            shuffle,
            augmenter,
            use_cropper,
            combine_labels,
            cascade,
            quality_weighting_scores,
        )
        # TODO: if cascade -> need to use fnames from the mask folder instead - test this
        if cascade:
            self.reader = NPYReader()
            self.image_fnames = [
                os.path.join(model_save_path, 'mask', dataset, f'{x}_SAX.nii.gz')
                for x in sorted(os.listdir(os.path.join(model_save_path, 'mask', dataset)))
            ]
        else:
            self.reader = NIIReader()
        if self.quality_weighting_scores:
            self.resizer = NPYReader()

    def get_quality_weightings(self, fname):
        """Get the quality weights for the current image - full or sliced."""
        curr_label = fname.split('/')[-1]
        curr_root = curr_label.split('_')[1]

        # Get the quality weighting scores
        qw = np.array([self.quality_weighting_scores[str(x)] for x in self.quality_weightings_dict[curr_root]], dtype=np.float32)

        # If the shape is not the same as the model depth, need to resize without interpolating
        if qw.shape[0] != self.image_size[-1]:
            qw = self.resizer.resize(qw, [self.image_size[-1]], interpolation_order=0)

        return qw


class Generator2D(__Generator):
    """ Class for the 2D image and label generator. """
    def __init__(
        self,
        model_save_path,
        generic_data_path,
        data_path,
        plane,
        batch_size,
        image_size,
        labels,
        dataset,
        shuffle=True,
        augmenter=None,
        use_cropper=False,
        combine_labels=None,
        cascade=None,
        quality_weighting_scores=None,
    ):
        super().__init__(
            model_save_path,
            generic_data_path,
            data_path,
            plane,
            batch_size,
            image_size,
            labels,
            dataset,
            shuffle,
            augmenter,
            use_cropper,
            combine_labels,
            cascade,
            quality_weighting_scores,
        )
        self.label_fnames = [
            [os.path.join(data_path, x, f'{x}_{i:03}_label.npy')
             for i in range(len(sorted(os.listdir(os.path.join(data_path, x)))) // 2)]
            for x in sorted(os.listdir(data_path))
        ]
        self.label_fnames = [x for y in self.label_fnames for x in y]

        if cascade:
            self.image_fnames = [
                [os.path.join(model_save_path, 'mask', dataset, x, f'{x}_{i:03}_image.npy')
                 for i in range(len(sorted(os.listdir(os.path.join(model_save_path, 'mask', dataset, x)))))]
                for x in sorted(os.listdir(os.path.join(model_save_path, 'mask', dataset)))
            ]
            self.image_fnames = [x for y in self.image_fnames for x in y]
        else:
            self.image_fnames = [
                [os.path.join(data_path, x, f'{x}_{i:03}_image.npy')
                 for i in range(len(sorted(os.listdir(os.path.join(data_path, x)))) // 2)]
                for x in sorted(os.listdir(data_path))
            ]
            self.image_fnames = [x for y in self.image_fnames for x in y]

        assert len(self.image_fnames) == len(self.label_fnames), \
            f"Number of image files and label files did not match! {len(self.image_fnames)} images vs. {len(self.label_fnames)} labels ... "
        self.index = np.arange(len(self.image_fnames))

        self.reader = NPYReader()

    def get_quality_weightings(self, fname):
        """Get the quality weights for the current image slice."""
        curr_label = fname.split('/')[-1]
        curr_root = curr_label.split('_')[1]
        curr_slice = int(curr_label.split('_')[2])

        return self.quality_weighting_scores[str(self.quality_weightings_dict[curr_root][curr_slice])]


class Generator3DShallow(Generator2D):
    """Class for 3D shallow quality weighting scores."""
    def __init__(
        self,
        model_save_path,
        generic_data_path,
        data_path,
        plane,
        batch_size,
        image_size,
        labels,
        dataset,
        shuffle=True,
        augmenter=None,
        use_cropper=False,
        combine_labels=None,
        cascade=None,
        quality_weighting_scores=None,
    ):
        super().__init__(
            model_save_path,
            generic_data_path,
            data_path,
            plane,
            batch_size,
            image_size,
            labels,
            dataset,
            shuffle,
            augmenter,
            use_cropper,
            combine_labels,
            cascade,
            quality_weighting_scores,
        )
        self.slice_depth = image_size[-1]

    def get_quality_weightings(self, fname):
        """Get the quality weights for the current image slice."""
        curr_label = fname.split('/')[-1]
        curr_root = curr_label.split('_')[1]
        curr_slice = int(curr_label.split('_')[2])

        qw = []
        for i in range(curr_slice, curr_slice + self.slice_depth):
            qw += [self.quality_weighting_scores[str(self.quality_weightings_dict[curr_root][i])]]

        return np.array(qw, dtype=np.float32)


class __CascadedGeneratorB(__Generator, ABC):
    def __init__(
        self,
        model_save_path,
        generic_data_path,
        data_path,
        plane,
        batch_size,
        image_size,
        labels,
        dataset,
        shuffle,
        augmenter,
        use_cropper,
        combine_labels,
        cascade,
        quality_weighting_scores,
    ):
        super().__init__(
            model_save_path,
            generic_data_path,
            data_path,
            plane,
            batch_size,
            image_size,
            labels,
            dataset,
            shuffle,
            augmenter,
            use_cropper,
            combine_labels,
            cascade,
            quality_weighting_scores,
        )

    def get_data(self, batch_indices, weight_mode):
        # Initialise empty arrays for the training data and labels
        x = np.empty([self.batch_size, *self.image_size, 1], dtype=np.float32)
        y_general = np.empty([self.batch_size, *self.image_size, 6], dtype=np.int8)
        y_scar = np.empty([self.batch_size, *self.image_size, 1], dtype=np.int8)
        y_pap = np.empty([self.batch_size, *self.image_size, 1], dtype=np.int8)

        if len(self.image_size) == 2:
            quality_weightings = np.zeros([self.batch_size], dtype=np.float32)
        else:
            quality_weightings = np.zeros([self.batch_size, self.image_size[-1]], dtype=np.float32)

        # Get the training data
        for i, index in enumerate(batch_indices):
            img, fname = self.read_file(index, self.image_fnames)
            label, fname = self.read_file(index, self.label_fnames)
            img = self.reader.normalize(img)

            # Apply data augmentation if option is turned on
            if self.augmenter and not weight_mode:
                img, label = self.augmenter.augment(img, label)

            x[i, ...] = self.prepare_img(img, fname)
            y_general[i, ...], y_scar[i, ...], y_pap[i, ...] = self.prepare_label(label, fname)
            if self.quality_weighting_scores:
                quality_weightings[i, ...] = self.get_quality_weightings(fname)

        if self.quality_weighting_scores:
            return {'model_in': x, 'qw_in': quality_weightings}, {'general_out': y_general, 'scar_out': y_scar, 'pap_out': y_pap, 'qw_out': quality_weightings}
        else:
            return x, {'general_out': y_general, 'scar_out': y_scar, 'pap_out': y_pap}

    def prepare_label(self, label, fname):
        """ Set the label to the correct size and dimensions for placement into the ground truth tensor. """
        # If cropping, then crop the label here
        if self.use_cropper:
            label = self.cropper.crop(label, fname)

        # Resize to the input shape of the model without interpolation
        if label.shape != self.image_size:
            label = self.reader.resize(label, self.image_size, interpolation_order=0)

        # Get the one hot label to start with
        one_hot_label = one_hot(label, len(self.labels), dtype=np.int8).numpy()

        # Get the scar-only and papillary muscle-only labels
        l_scar = one_hot_label[..., 3]
        l_pap = one_hot_label[..., 6]

        # The general label should combine the scar into LV myo and papillary muscles into LV lumen
        l_general = np.empty([*self.image_size, 6])
        l_general[..., :3] = one_hot_label[..., :3]
        l_general[..., 3:5] = one_hot_label[..., 4:6]
        l_general[..., 5] = one_hot_label[..., 7]
        l_general[..., 2] += l_scar  # Adding scar to myo
        l_general[..., 1] += l_pap  # Adding pap to lumen

        return l_general, l_scar[..., np.newaxis], l_pap[..., np.newaxis]


class CascadedGenerator2DB(__CascadedGeneratorB):
    def __init__(
        self,
        model_save_path,
        generic_data_path,
        data_path,
        plane,
        batch_size,
        image_size,
        labels,
        dataset,
        shuffle=True,
        augmenter=None,
        use_cropper=False,
        combine_labels=None,
        cascade=None,
        quality_weighting_scores=None,
    ):
        super().__init__(
            model_save_path,
            generic_data_path,
            data_path,
            plane,
            batch_size,
            image_size,
            labels,
            dataset,
            shuffle,
            augmenter,
            use_cropper,
            combine_labels,
            cascade,
            quality_weighting_scores,
        )
        self.label_fnames = [
            [os.path.join(data_path, x, f'{x}_{i:03}_label.npy')
             for i in range(len(sorted(os.listdir(os.path.join(data_path, x)))) // 2)]
            for x in sorted(os.listdir(data_path))
        ]
        self.label_fnames = [x for y in self.label_fnames for x in y]

        if cascade:
            self.image_fnames = [
                [os.path.join(model_save_path, 'mask', dataset, x, f'{x}_{i:03}_image.npy')
                 for i in range(len(sorted(os.listdir(os.path.join(model_save_path, 'mask', dataset, x)))))]
                for x in sorted(os.listdir(os.path.join(model_save_path, 'mask', dataset)))
            ]
            self.image_fnames = [x for y in self.image_fnames for x in y]
        else:
            self.image_fnames = [
                [os.path.join(data_path, x, f'{x}_{i:03}_image.npy')
                 for i in range(len(sorted(os.listdir(os.path.join(data_path, x)))) // 2)]
                for x in sorted(os.listdir(data_path))
            ]
            self.image_fnames = [x for y in self.image_fnames for x in y]

        assert len(self.image_fnames) == len(self.label_fnames), \
            f"Number of image files and label files did not match! {len(self.image_fnames)} images vs. {len(self.label_fnames)} labels ... "
        self.index = np.arange(len(self.image_fnames))

        self.reader = NPYReader()

    def get_quality_weightings(self, fname):
        """Get the quality weights for the current image slice."""
        curr_label = fname.split('/')[-1]
        curr_root = curr_label.split('_')[1]
        curr_slice = int(curr_label.split('_')[2])

        return self.quality_weighting_scores[str(self.quality_weightings_dict[curr_root][curr_slice])]


class CascadedGenerator3DB(__CascadedGeneratorB):
    def __init__(
        self,
        model_save_path,
        generic_data_path,
        data_path,
        plane,
        batch_size,
        image_size,
        labels,
        dataset,
        shuffle=True,
        augmenter=None,
        use_cropper=False,
        combine_labels=None,
        cascade=None,
        quality_weighting_scores=None,
    ):
        super().__init__(
            model_save_path,
            generic_data_path,
            data_path,
            plane,
            batch_size,
            image_size,
            labels,
            dataset,
            shuffle,
            augmenter,
            use_cropper,
            combine_labels,
            cascade,
            quality_weighting_scores,
        )
        # TODO: if cascade -> need to use fnames from the mask folder instead - test this
        if cascade:
            self.reader = NPYReader()
            self.image_fnames = [
                os.path.join(model_save_path, 'mask', dataset, f'{x}_SAX.nii.gz')
                for x in sorted(os.listdir(os.path.join(model_save_path, 'mask', dataset)))
            ]
        else:
            self.reader = NIIReader()
        if self.quality_weighting_scores:
            self.resizer = NPYReader()

    def get_quality_weightings(self, fname):
        """Get the quality weights for the current image - full or sliced."""
        curr_label = fname.split('/')[-1]
        curr_root = curr_label.split('_')[1]

        # Get the quality weighting scores
        qw = np.array([self.quality_weighting_scores[str(x)] for x in self.quality_weightings_dict[curr_root]], dtype=np.float32)

        # If the shape is not the same as the model depth, need to resize without interpolating
        if qw.shape[0] != self.image_size[-1]:
            qw = self.resizer.resize(qw, [self.image_size[-1]], interpolation_order=0)

        return qw
