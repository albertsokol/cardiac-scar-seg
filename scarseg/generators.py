import json
import os
from abc import ABC

import numpy as np
from tensorflow import one_hot
from tensorflow.keras.utils import Sequence

from scarseg.augmenter import Augmenter2D
from scarseg.cropper import Cropper
from scarseg.masker import Masker
from scarseg.readers import NIIReader, NPYReader


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
        self.image_fnames = [
            os.path.join(data_path, x, f"{x}_SAX.nii.gz")
            for x in sorted(os.listdir(data_path))
        ]
        self.label_fnames = [
            os.path.join(data_path, x, f"{x}_SAX_mask2.nii.gz")
            for x in sorted(os.listdir(data_path))
        ]
        assert len(self.image_fnames) == len(
            self.label_fnames
        ), "Number of image files and label files did not match!"
        self.index = np.arange(len(self.image_fnames))

        # Get the quality weightings dictionary
        if quality_weighting_scores:
            with open("quality_scores.json", "r") as f:
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

        self.cropper = (
            Cropper(generic_data_path, dataset, mode=use_cropper)
            if use_cropper
            else False
        )

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
        indices = self.index[self.batch_size * index : self.batch_size * (index + 1)]
        if indices.size == 0:
            raise IndexError(
                "Index not within possible range (0 to number of training steps)"
            )
        # Generate the data
        x, y = self.get_data(indices, weight_mode)
        return x, y.astype(np.float32)

    def get_data(self, batch_indices, weight_mode):
        # Initialise empty arrays for the training data and labels
        x = np.empty([self.batch_size, *self.image_size, 1], dtype=np.float32)
        if self.combine_labels:
            y = np.empty(
                [self.batch_size, *self.image_size, len(self.combine_labels)],
                dtype=np.int8,
            )
        else:
            y = np.empty(
                [self.batch_size, *self.image_size, len(self.labels)], dtype=np.int8
            )

        if len(self.image_size) == 2:
            quality_weightings = np.zeros([self.batch_size], dtype=np.float32)
        else:
            quality_weightings = np.zeros(
                [self.batch_size, self.image_size[-1]], dtype=np.float32
            )

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
            return {"model_in": x, "qw_in": quality_weightings}, {
                "qw_out": quality_weightings,
                "m": y,
            }
        else:
            return x, y

    def read_file(self, index, fname_list):
        """Read the file at the given index of the given list."""
        return self.reader.read(fname_list[index]), fname_list[index]

    def prepare_img(self, img, fname):
        """Set the image to the correct size and dimensions for placement into the input tensor."""
        # If cropping, then crop the image here
        if self.use_cropper and not self.cascade:
            img = self.cropper.crop(img, fname)

        # Resize to the input shape of the model
        if img.shape != self.image_size:
            img = self.reader.resize(img, self.image_size)

        img = img[..., np.newaxis]

        return self.reader.normalize(img)

    def prepare_label(self, label, fname):
        """Set the label to the correct size and dimensions for placement into the ground truth tensor."""
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
    """Class for the 3D image and label generator."""

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
        if cascade:
            self.reader = NPYReader()
            self.image_fnames = [
                os.path.join(model_save_path, "mask", dataset, f"{x}_SAX.nii.gz")
                for x in sorted(
                    os.listdir(os.path.join(model_save_path, "mask", dataset))
                )
            ]
        else:
            self.reader = NIIReader()
        if self.quality_weighting_scores:
            self.resizer = NPYReader()

    def get_quality_weightings(self, fname):
        """Get the quality weights for the current image - full or sliced."""
        curr_label = fname.split("/")[-1]
        curr_root = curr_label.split("_")[1]

        # Get the quality weighting scores
        qw = np.array(
            [
                self.quality_weighting_scores[str(x)]
                for x in self.quality_weightings_dict[curr_root]
            ],
            dtype=np.float32,
        )

        # If the shape is not the same as the model depth, need to resize without interpolating
        if qw.shape[0] != self.image_size[-1]:
            qw = self.resizer.resize(qw, [self.image_size[-1]], interpolation_order=0)

        return qw


class Generator3DFrozen(Generator3D):
    """Generate full-size 3D images, but keep the dimensionality of the depth axis the same (no interpolation)."""

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
        assert (
            batch_size == 1
        ), "Only batch size 1 will work with 3DFrozen as differing input depths are non-batchable"

    def get_data(self, batch_index, weight_mode):
        # Get the training data
        img, fname = self.read_file(batch_index[0], self.image_fnames)
        label, fname = self.read_file(batch_index[0], self.label_fnames)
        img = self.reader.normalize(img)
        depth = img.shape[-1]

        # print(f"get_data: {img.shape=}")
        # print(f"get_data: {depth=}")

        assert (
            img.shape == label.shape
        ), f"Shape incompatibility in file {self.image_fnames[batch_index]}: {img.shape=}, {label.shape=}"

        # TODO: update quality weightings if needed, idk if it will work as-is
        quality_weightings = np.zeros([1, depth], dtype=np.float32)

        x = np.empty(
            [1, self.image_size[0], self.image_size[1], depth, 1], dtype=np.float32
        )
        # print(f"get_data: {x.shape=}")

        if self.combine_labels:
            y = np.empty(
                [
                    1,
                    self.image_size[0],
                    self.image_size[1],
                    depth,
                    len(self.combine_labels),
                ],
                dtype=np.int8,
            )
        else:
            y = np.empty(
                [1, self.image_size[0], self.image_size[1], depth, len(self.labels)],
                dtype=np.int8,
            )

        # Apply data augmentation if option is turned on
        if self.augmenter and not weight_mode:
            img, label = self.augmenter.augment(img, label)

        # print(f"get_data post augmenter: {img.shape=}")

        x[0, ...] = self.prepare_img(img, fname)
        y[0, ...] = self.prepare_label(label, fname)

        if self.quality_weighting_scores:
            quality_weightings[0, ...] = self.get_quality_weightings(fname)

        if self.quality_weighting_scores:
            return {"model_in": x, "qw_in": quality_weightings}, {
                "qw_out": quality_weightings,
                "m": y,
            }
        else:
            return x, y

    def prepare_img(self, img, fname):
        """Set the image to the correct size and dimensions for placement into the input tensor."""
        # If cropping, then crop the image here
        if self.use_cropper and not self.cascade:
            img = self.cropper.crop(img, fname)

        # print(f"prepare_img: {img.shape=}")

        # Resize to the input shape of the model
        if img.shape[:2] != self.image_size[:2]:
            img = self.reader.resize(img, self.image_size[:2] + [img.shape[-1]])

        # print(f"prepare_img: {img.shape=}")

        img = img[..., np.newaxis]

        return self.reader.normalize(img)

    def prepare_label(self, label, fname):
        """Set the label to the correct size and dimensions for placement into the ground truth tensor."""
        # If cropping, then crop the label here
        if self.use_cropper:
            label = self.cropper.crop(label, fname)

        # Resize to the input shape of the model without interpolation
        if label.shape[:2] != self.image_size[:2]:
            label = self.reader.resize(
                label, self.image_size[:2] + [label.shape[-1]], interpolation_order=0
            )

        # If labels are to be combined, do that here and return a one hot encoded tensor
        if self.combine_labels:
            return self.apply_label_combine(label)

        # One hot encode the labels to create a new channel for each label and save as int8 to save space
        return one_hot(label, len(self.labels), dtype=np.int8).numpy()

    def get_quality_weightings(self, fname):
        """Get the quality weights for the current image - full or sliced."""
        curr_label = fname.split("/")[-1]
        curr_root = curr_label.split("_")[1]

        # Get the quality weighting scores
        qw = np.array(
            [
                self.quality_weighting_scores[str(x)]
                for x in self.quality_weightings_dict[curr_root]
            ],
            dtype=np.float32,
        )

        # No need to interpolate - the depth of the quality scores will be the same as the number of slices

        return qw


class Generator2D(__Generator):
    """Class for the 2D image and label generator."""

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
            [
                os.path.join(data_path, x, f"{x}_{i:03}_label.npy")
                for i in range(len(sorted(os.listdir(os.path.join(data_path, x)))) // 2)
            ]
            for x in sorted(os.listdir(data_path))
        ]
        self.label_fnames = [x for y in self.label_fnames for x in y]

        if cascade:
            self.image_fnames = [
                [
                    os.path.join(
                        model_save_path, "mask", dataset, x, f"{x}_{i:03}_image.npy"
                    )
                    for i in range(
                        len(
                            sorted(
                                os.listdir(
                                    os.path.join(model_save_path, "mask", dataset, x)
                                )
                            )
                        )
                    )
                ]
                for x in sorted(
                    os.listdir(os.path.join(model_save_path, "mask", dataset))
                )
            ]
            self.image_fnames = [x for y in self.image_fnames for x in y]
        else:
            self.image_fnames = [
                [
                    os.path.join(data_path, x, f"{x}_{i:03}_image.npy")
                    for i in range(
                        len(sorted(os.listdir(os.path.join(data_path, x)))) // 2
                    )
                ]
                for x in sorted(os.listdir(data_path))
            ]
            self.image_fnames = [x for y in self.image_fnames for x in y]

        assert len(self.image_fnames) == len(
            self.label_fnames
        ), f"Number of image files and label files did not match! {len(self.image_fnames)} images vs. {len(self.label_fnames)} labels ... "
        self.index = np.arange(len(self.image_fnames))

        self.reader = NPYReader()

    def get_quality_weightings(self, fname):
        """Get the quality weights for the current image slice."""
        curr_label = fname.split("/")[-1]
        curr_root = curr_label.split("_")[1]
        curr_slice = int(curr_label.split("_")[2])

        return self.quality_weighting_scores[
            str(self.quality_weightings_dict[curr_root][curr_slice])
        ]


class Generator2DPositional(Generator2D):
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

    @staticmethod
    def embed_position(index: int) -> float:
        """Given the index of a slice in a scan, get a positional embedding which can be input to the model."""
        return (index - 7.5) / 7.5

    def get_data(self, batch_indices, weight_mode):
        # Initialise empty arrays for the training data and labels
        x = np.empty([self.batch_size, *self.image_size, 1], dtype=np.float32)
        pos_embeddings = np.empty([self.batch_size], dtype=np.float32)
        if self.combine_labels:
            y = np.empty(
                [self.batch_size, *self.image_size, len(self.combine_labels)],
                dtype=np.int8,
            )
        else:
            y = np.empty(
                [self.batch_size, *self.image_size, len(self.labels)], dtype=np.int8
            )

        # Get the training data
        for i, index in enumerate(batch_indices):
            img, fname = self.read_file(index, self.image_fnames)
            label, fname = self.read_file(index, self.label_fnames)
            img = self.reader.normalize(img)
            slice_index = int(fname.split("_")[3])
            pos_embeddings[i] = self.embed_position(slice_index)

            # Apply data augmentation if option is turned on
            if self.augmenter and not weight_mode:
                img, label = self.augmenter.augment(img, label)

            x[i, ...] = self.prepare_img(img, fname)
            y[i, ...] = self.prepare_label(label, fname)

        return {"model_in": x, "position_in": pos_embeddings}, y


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
        curr_label = fname.split("/")[-1]
        curr_root = curr_label.split("_")[1]
        curr_slice = int(curr_label.split("_")[2])

        qw = []
        for i in range(curr_slice, curr_slice + self.slice_depth):
            qw += [
                self.quality_weighting_scores[
                    str(self.quality_weightings_dict[curr_root][i])
                ]
            ]

        return np.array(qw, dtype=np.float32)


class DAEGenerator(Sequence):
    """Generator for the denoising auto-encoder."""

    def __init__(
        self,
        generic_data_path,
        data_path,
        batch_size,
        image_size,
        labels,
        dataset,
        shuffle,
        use_cropper,
        vox_permute_p=0.05,
        zoom_aug=(0.96, 1.04),
    ):
        # Set up image filenames and indexing
        self.generic_data_path = generic_data_path  # root of all data folders
        self.data_path = data_path  # model and plane specific path
        self.dataset = dataset
        self.in_fnames = [
            [
                os.path.join(data_path, x, f"{x}_{i:03}_label.npy")
                for i in range(len(sorted(os.listdir(os.path.join(data_path, x)))) // 2)
            ]
            for x in sorted(os.listdir(data_path))
        ]
        self.in_fnames = [x for y in self.in_fnames for x in y]
        self.index = np.arange(len(self.in_fnames))
        self.reader = NPYReader()

        # Image handling: augmentation, cropping
        self.cropper = (
            Cropper(generic_data_path, dataset, mode=use_cropper)
            if use_cropper
            else False
        )
        self.use_zoom_aug = True if zoom_aug else False
        self.augmenter = Augmenter2D(zoom=zoom_aug)
        self.vox_permute_p = vox_permute_p

        # Model parameters
        self.batch_size = batch_size
        self.image_size = image_size
        self.labels = labels
        self.label_indices = {self.labels[k]: int(k) for k in self.labels}
        self.use_cropper = use_cropper

        # Shuffle the data before starting if shuffling has been turned on
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        # Optionally shuffle the data at the end of each epoch
        if self.shuffle:
            np.random.shuffle(self.index)

    def __len__(self):
        # Number of gradient descent steps that will be taken per epoch
        return len(self.in_fnames) // self.batch_size

    def __getitem__(self, index, weight_mode=False):
        # Create a list of batch_size numerical indices
        indices = self.index[self.batch_size * index : self.batch_size * (index + 1)]
        if indices.size == 0:
            raise IndexError(
                "Index not within possible range (0 to number of training steps)"
            )
        # Generate the data
        x, y = self.get_data(indices, weight_mode)
        return x, y

    def get_data(self, batch_indices, weight_mode):
        # Initialise empty arrays for the training data and labels
        x = np.empty(
            [self.batch_size, *self.image_size, len(self.labels)], dtype=np.int8
        )
        y = np.empty(
            [self.batch_size, *self.image_size, len(self.labels)], dtype=np.int8
        )

        # Get the training data
        for i, index in enumerate(batch_indices):
            img, fname = self.read_file(index, self.in_fnames)

            # Apply simple zoom augmentation if option is turned on
            if self.use_zoom_aug and not weight_mode:
                img = self.zoom_aug(img)

            img = self.prepare(img, fname)

            # Add permutation after preparing the label
            img, label = self.permute_input(img)

            x[i, ...] = img
            y[i, ...] = label

        return x, y

    def read_file(self, index, fname_list):
        """Read the file at the given index of the given list."""
        return self.reader.read(fname_list[index]), fname_list[index]

    def prepare(self, label, fname):
        """Set the label to the correct size and dimensions for placement into the ground truth tensor."""
        # If cropping, then crop the image here
        if self.use_cropper:
            label = self.cropper.crop(label, fname)

        # Resize to the input shape of the model without interpolation
        if label.shape != self.image_size:
            label = self.reader.resize(label, self.image_size, interpolation_order=0)

        # One hot encode the labels to create a new channel for each label and save as int8 to save space
        return one_hot(label, len(self.labels), dtype=np.int8).numpy()

    def zoom_aug(self, f):
        """Apply simple zoom augmentation to the input."""
        zoom_factor = self.augmenter.rng.uniform(
            self.augmenter.zoom[0], self.augmenter.zoom[1]
        )
        return self.augmenter.apply_zoom(f, zoom_factor, interpolation_order=0)

    def permute_input(self, f):
        """Apply permutation to the input label to create the new input and the target."""
        permute_field = np.random.uniform(0.0, 1.0, f.shape[:-1])
        new_vox_labels = np.random.randint(0, len(self.labels), f.shape[:-1])
        f_permute = np.where(
            np.less_equal(permute_field, self.vox_permute_p),
            new_vox_labels,
            np.argmax(f, axis=-1),
        )
        f_permute = one_hot(f_permute, len(self.labels), dtype=np.int8).numpy()
        return f_permute, f
