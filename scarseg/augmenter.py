from abc import ABC

import elasticdeform
import numpy as np
import scipy.ndimage as snd


class __Augmenter(ABC):
    """ABC for augmenters."""

    def __init__(self, zoom, rotate, translate, deform, brightness):
        self.rng = np.random.default_rng()
        self.zoom = zoom
        self.rotate = rotate
        self.translate = translate
        self.brightness = brightness
        self.deform = deform

    def augment(self, img, label):
        raise NotImplementedError

    @staticmethod
    def normalize(img):
        """Normalize voxel values to be within 0 to 1 range."""
        normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
        if np.isnan(np.max(normalized)):
            return np.zeros(img.shape)
        return normalized

    def apply_brightness(self, img):
        """Apply brightness augmentations. Note that this only needs to be done for the image, and not the label."""
        # Apply brightness change disproportionately; changing only dark values to prevent changing the blacks to make
        # it harder for the model to understand the bounds of the image
        polarity = -1 if self.rng.uniform() < 0.5 else 1
        brightness_factor = 1 - polarity * self.rng.uniform(0, self.brightness)
        return img ** brightness_factor

    @staticmethod
    def apply_deform(img, label):
        """Apply B-spline deformation to an image and label simultaneously - the same transform applies to both."""
        # Note that the image and label must be the same size in order to use this
        assert (
            img.shape == label.shape
        ), f"Image and label must be the same shape but got image: {img.shape}, label: {label.shape}"
        img, label = elasticdeform.deform_random_grid(
            [img, label], sigma=img.shape[0] / 500.0, points=10, order=[3, 0]
        )
        return img, label


class Augmenter2D(__Augmenter):
    """Class for 2D data augmentation."""

    def __init__(
        self, zoom=None, rotate=None, translate=None, deform=False, brightness=None
    ):
        super(Augmenter2D, self).__init__(zoom, rotate, translate, deform, brightness)

    def augment(self, img, label):
        """Simultaneously augment the image and label together. Both must be numpy arrays."""
        assert (
            type(img) == np.ndarray
        ), f"Image must be a numpy array but type {type(img)} was given"
        assert (
            type(label) == np.ndarray
        ), f"Label must be a numpy array but type {type(label)} was given"
        assert np.max(img) == 1.0 or np.max(img) == 0.0, (
            f"Image must be normalized before augmentation, "
            f"but max value was not 1. - got {np.max(img)}"
        )
        assert (
            np.min(img) == 0.0
        ), f"Image must be normalized before augmentation, but min value was not 0. - got {np.min(img)}"

        # Apply translation augmentations
        if self.translate:
            h_factor, w_factor = self.rng.uniform(0.0, self.translate, 2)
            img = self.apply_translate(img, h_factor, w_factor)
            label = self.apply_translate(label, h_factor, w_factor)

        if self.rotate:
            angle = self.rng.uniform(-self.rotate, self.rotate)
            img = self.apply_rotation(img, angle)
            label = self.apply_rotation(label, angle, interpolation_order=0)

            # Interpolation can affect pixel values, so need to normalize the image here
            img = self.normalize(img)

        # Apply zoom augmentations
        if self.zoom:
            zoom_factor = self.rng.uniform(self.zoom[0], self.zoom[1])
            img = self.apply_zoom(img, zoom_factor)
            label = self.apply_zoom(label, zoom_factor, interpolation_order=0)
            # Interpolation can affect pixel values, so need to normalize the image here

            img = self.normalize(img)

        if self.deform:
            img, label = self.apply_deform(img, label)
            # Interpolation can affect pixel values, so need to normalize here
            img = self.normalize(img)

        # Apply brightness augmentations
        if self.brightness:
            img = self.apply_brightness(img)

        return img, label

    def apply_translate(self, img, h_factor, w_factor):
        """Translate the image and label simultaneously by a random amount up to self.translate"""
        h, w = img.shape

        # Randomly decide whether to shift forwards or backwards in each direction
        plr = [-1 if x < 0.5 else 1 for x in self.rng.uniform(size=2)]
        # Amount to translate each dimension by, based on self.translate
        th = int(h * h_factor)
        tw = int(w * w_factor)

        out = np.zeros_like(img)

        # Shift the square representing the original image inside the square representing the 0-padded container
        out[
            max(th * plr[0], 0) : min(h, h + th * plr[0]),
            max(tw * plr[1], 0) : min(w, w + tw * plr[1]),
        ] = img[
            max(th * -plr[0], 0) : min(h, h + th * -plr[0]),
            max(tw * -plr[1], 0) : min(w, w + tw * -plr[1]),
        ]

        return out

    @staticmethod
    def apply_zoom(img, zoom_factor, interpolation_order=3):
        """
        Apply zoom to 2 dimensional images. Adapted from
        https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
        """
        h, w = img.shape

        # Get a random zoom factor between between the lower and upper bounds of self.zoom
        zoom_tuple = (zoom_factor,) * 2

        # Zooming out
        if zoom_factor < 1:
            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            # Zero-padding
            out = np.zeros_like(img)
            out[top : top + zh, left : left + zw] = snd.zoom(
                img, zoom_tuple, order=interpolation_order
            )

        # Zooming in
        else:
            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2

            # print(f'image shape: {img.shape}, zoom_factor: {zoom_factor}, zh: {zh}, zw: {zw}, top: {top}, left: {left}')

            out = snd.zoom(
                img[top : top + zh, left : left + zw],
                zoom_tuple,
                order=interpolation_order,
            )

            # print(out)

            # `out` might still be slightly larger than `_img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = max((out.shape[0] - h) // 2, 0)
            trim_left = max((out.shape[1] - w) // 2, 0)

            # print(f'trim_top: {trim_top}, trim_left: {trim_left}')
            out = out[trim_top : trim_top + h, trim_left : trim_left + w]
            # print(out)

        return out

    @staticmethod
    def apply_rotation(img, angle, interpolation_order=3):
        """Apply rotation augmentation to the image."""
        out = snd.rotate(img, angle=angle, reshape=False, order=interpolation_order)
        return out


class Augmenter3D(__Augmenter):
    """Class for 3D data augmentation."""

    def __init__(
        self, zoom=None, rotate=None, translate=None, deform=False, brightness=None
    ):
        """

        :param zoom: False, or list(float) length 2, idx0 = lower bound e.g., 0.95 for zoom out by 5% and idx1 = upper
                     bound e.g., 1.05 for zoom in by 5%
        :param rotate: False, or float: maximal number of degrees to rotate by, recommend 8
        :param translate: False, or float: maximal % of given dimension size to translate by, recommend 0.06
        :param brightness: False, or float: exponential factor to modify brightness by., recommend 0.15
        :param deform: bool: set to True to turn B-spline deformations on. Note, this is slow
        """
        super(Augmenter3D, self).__init__(zoom, rotate, translate, deform, brightness)
        self.test_params()

    def test_params(self):
        """Test that the parameters given to the constructor are in the correct format."""
        if self.zoom:
            if (
                type(self.zoom) == list
                and len(self.zoom) == 2
                and type(self.zoom[0] == float)
                and type(self.zoom[1] == float)
            ):
                pass
                if self.zoom[0] > self.zoom[1]:
                    raise AttributeError(
                        f"First value in zoom must be less than second value: cannot use {self.zoom}"
                    )
            else:
                raise AttributeError(
                    f"Param zoom incorrect: gave {self.zoom} ({type(self.zoom)}); expected length 2 list of floats"
                )
        if self.translate:
            if type(self.translate) is not float:
                raise AttributeError(
                    f"Param translate incorrect: gave {self.translate} ({type(self.translate)}); expected a float"
                )
            if self.translate <= 0.0 or self.translate >= 1.0:
                raise AttributeError(
                    f"Param translate incorrect: must be a float between 0 and 1, received {self.translate}"
                )
        if self.brightness:
            if type(self.brightness) is not float:
                raise AttributeError(
                    f"Param brightness incorrect: gave {self.brightness} ({type(self.brightness)}); expected a float"
                )
            if self.brightness <= 0.0 or self.brightness >= 1.0:
                raise AttributeError(
                    f"Param brightness incorrect: must be a float between 0 and 1, received {self.brightness}"
                )
        if not isinstance(self.deform, bool):
            raise AttributeError(
                f"Param deform must be a boolean: got {type(self.deform)}"
            )

    def augment(self, img, label):
        """Simultaneously augment the image and label together. Both must be numpy arrays."""
        assert (
            type(img) == np.ndarray
        ), f"Image must be a numpy array but type {type(img)} was given"
        assert (
            type(label) == np.ndarray
        ), f"Label must be a numpy array but type {type(label)} was given"
        assert (
            np.max(img) == 1.0 or np.max(img) == 0.0
        ), f"Image must be normalized before augmentation, but max value was not 1. - got {np.max(img)}"
        assert (
            np.min(img) == 0.0
        ), f"Image must be normalized before augmentation, but min value was not 0. - got {np.min(img)}"

        # Apply translation augmentations
        if self.translate:
            h_factor, w_factor = self.rng.uniform(0.0, self.translate, 2)
            d_factor = 0.0  # translating across depth breaks shallow UNet3D inputs so just set it to 0 here
            img = self.apply_translate(img, h_factor, w_factor, d_factor)
            label = self.apply_translate(label, h_factor, w_factor, d_factor)

        if self.rotate:
            axes = [0, 1]
            angle = self.rng.uniform(-self.rotate, self.rotate)
            img = self.apply_rotation(img, axes, angle)
            label = self.apply_rotation(label, axes, angle, interpolation_order=0)

            # Interpolation can affect pixel values, so need to normalize the image here
            img = self.normalize(img)

        # Apply zoom augmentations
        if self.zoom:
            zoom_factor = self.rng.uniform(self.zoom[0], self.zoom[1])
            img = self.apply_zoom(img, zoom_factor)
            label = self.apply_zoom(label, zoom_factor, interpolation_order=0)
            # Interpolation can affect pixel values, so need to normalize the image here
            img = self.normalize(img)

        if self.deform:
            img, label = self.apply_deform(img, label)
            # Interpolation can affect pixel values, so need to normalize here
            img = self.normalize(img)

        # Apply brightness augmentations
        if self.brightness:
            img = self.apply_brightness(img)

        return img, label

    @staticmethod
    def apply_zoom(img, zoom_factor, interpolation_order=3):
        """
        Apply zoom to 3 dimensional images. Adapted from
        https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
        """
        h, w, d = img.shape

        # print(f"apply_zoom: {h=}, {w=}, {d=}")

        # Get a random zoom factor between between the lower and upper bounds of self.zoom
        zoom_tuple = (zoom_factor, zoom_factor, 1)

        # print(f"apply_zoom: {zoom_factor=}, {zoom_tuple=}")

        # Zooming out
        if zoom_factor < 1:
            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            # zd = int(np.ceil(d * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
            # inside = (d - zd) // 2

            # print(f"apply_zoom: {zh=}, {zw=}, {top=}, {left=}")

            # Zero-padding
            out = np.zeros_like(img)
            out[top : top + zh, left : left + zw, ...] = snd.zoom(
                img, zoom_tuple, order=interpolation_order
            )

            # print(f"apply_zoom: {out.shape=}")

        # Zooming in
        else:
            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            # zd = int(np.ceil(d / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
            # inside = (d - zd) // 2

            # print(f"apply_zoom: {zh=}, {zw=}, {top=}, {left=}")

            out = snd.zoom(
                img[top : top + zh, left : left + zw, ...],
                zoom_tuple,
                order=interpolation_order,
            )

            # print(f"apply_zoom: {out.shape=}")

            # `out` might still be slightly larger than `_img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = max((out.shape[0] - h) // 2, 0)
            trim_left = max((out.shape[1] - w) // 2, 0)
            # trim_inside = max((out.shape[2] - d) // 2, 0)

            # print(f"apply_zoom: {trim_top=}, {trim_left=}")

            out = out[trim_top : trim_top + h, trim_left : trim_left + w, ...]

            # print(f"apply_zoom: {out.shape=}")

        return out

    @staticmethod
    def apply_rotation(img, axes, angle, interpolation_order=3):
        """Apply rotation augmentation to the volume."""
        out = snd.rotate(
            img, axes=axes, angle=angle, reshape=False, order=interpolation_order
        )
        return out

    def apply_translate(self, img, h_factor, w_factor, d_factor):
        """Translate the image and label simultaneously by a random amount up to self.translate"""
        h, w, d = img.shape

        # Randomly decide whether to shift forwards or backwards in each direction
        plr = [-1 if x < 0.5 else 1 for x in self.rng.uniform(size=3)]
        # Amount to translate each dimension by, based on self.translate
        th = int(h * h_factor)
        tw = int(w * w_factor)
        td = int(d * d_factor)

        out = np.zeros_like(img)

        # Shift the cube representing the original image inside the cube representing the 0-padded container
        out[
            max(th * plr[0], 0) : min(h, h + th * plr[0]),
            max(tw * plr[1], 0) : min(w, w + tw * plr[1]),
            max(td * plr[2], 0) : min(d, d + td * plr[2]),
        ] = img[
            max(th * -plr[0], 0) : min(h, h + th * -plr[0]),
            max(tw * -plr[1], 0) : min(w, w + tw * -plr[1]),
            max(td * -plr[2], 0) : min(d, d + td * -plr[2]),
        ]

        return out
