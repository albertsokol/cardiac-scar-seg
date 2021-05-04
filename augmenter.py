import numpy as np
from readers import NIIReader
import scipy.ndimage as snd
import elasticdeform
import time
import matplotlib.pyplot as plt


class Augmenter:
    """ Class for data augmentation. """
    def __init__(self, zoom=None, rotate=None, translate=None, deform=False, brightness=None):
        """

        :param zoom: None, or tuple(float) length 2, idx0 = lower bound e.g., 0.95 for zoom out by 5% and idx1 = upper
                     bound e.g., 1.05 for zoom in by 5%
        :param rotate: None, or float: maximal number of degrees to rotate by, recommend 8
        :param translate: None, or float: maximal % of given dimension size to translate by, recommend 0.05
        :param brightness: None, or float: exponential factor to modify brightness by., recommend 0.15
        :param deform: bool: set to True to turn B-spline deformations on. Note, this is slow
        """
        self.rng = np.random.default_rng()
        self.zoom = zoom
        self.rotate = rotate
        self.translate = translate
        self.brightness = brightness
        self.deform = deform
        self.test_params()

    def test_params(self):
        """ Test that the parameters given to the constructor are in the correct format. """
        if self.zoom is not None:
            if type(self.zoom) == tuple and len(self.zoom) == 2 and type(self.zoom[0] == float) and type(self.zoom[1] == float):
                pass
                if self.zoom[0] > self.zoom[1]:
                    raise AttributeError(f'First value in zoom must be less than second value: cannot use {self.zoom}')
            else:
                raise AttributeError(f'Param zoom incorrect: gave {self.zoom} ({type(self.zoom)}); expected length 2 tuple of floats')
        if self.translate is not None:
            if type(self.translate) is not float:
                raise AttributeError(f'Param translate incorrect: gave {self.translate} ({type(self.translate)}); expected a float')
            if self.translate <= 0. or self.translate >= 1.:
                raise AttributeError(f'Param translate incorrect: must be a float between 0 and 1, received {self.translate}')
        if self.brightness is not None:
            if type(self.brightness) is not float:
                raise AttributeError(f'Param brightness incorrect: gave {self.brightness} ({type(self.brightness)}); expected a float')
            if self.brightness <= 0. or self.brightness >= 1.:
                raise AttributeError(f'Param brightness incorrect: must be a float between 0 and 1, received {self.brightness}')
        if not isinstance(self.deform, bool):
            raise AttributeError(f'Param deform must be a boolean: got {type(self.deform)}')

    def augment(self, img, label):
        """ Simultaneously augment the image and label together. Both must be numpy arrays. """
        assert type(img) == np.ndarray, f'Image must be a numpy array but type {type(img)} was given'
        assert type(label) == np.ndarray, f'Label must be a numpy array but type {type(label)} was given'
        assert np.max(img) == 1., f'Image must be normalized before augmentation, but max value was not 1. - got {np.max(img)}'
        assert np.min(img) == 0., f'Image must be normalized before augmentation, but min value was not 0. - got {np.min(img)}'
        assert np.max(label) == 1., f'Label must be normalized before augmentation, but max value was not 1. - got {np.max(label)}'
        assert np.min(label) == 0., f'Label must be normalized before augmentation, but min value was not 0. - got {np.min(label)}'

        print(f'Image characteristics: max: {np.max(img)}, min: {np.min(img)}')
        print(f'Label characteristics: max: {np.max(label)}, min: {np.min(label)}')

        # Apply translation augmentations
        if self.translate is not None:
            h_factor, w_factor, d_factor = self.rng.uniform(0., self.translate, 3)
            img = self.apply_translate(img, h_factor, w_factor, d_factor)
            label = self.apply_translate(label, h_factor, w_factor, d_factor)

        if self.rotate is not None:
            # Select the 2 axes to rotate around randomly
            axes = []
            while len(axes) != 2:
                axis = self.rng.integers(0, 3)
                if axis not in axes:
                    axes += [axis]

            # print(axes)
            angle = self.rng.uniform(-self.rotate, self.rotate)
            # print(angle)
            img = self.apply_rotation(img, axes, angle)
            label = self.apply_rotation(label, axes, angle)

            # Interpolation can affect pixel values, so need to normalize here
            img = self.normalize(img)
            label = self.normalize(label)

        # Apply zoom augmentations
        if self.zoom is not None:
            zoom_factor = self.rng.uniform(self.zoom[0], self.zoom[1])
            img = self.apply_zoom(img, zoom_factor)
            label = self.apply_zoom(label, zoom_factor)
            # Interpolation can affect pixel values, so need to normalize here
            img = self.normalize(img)
            label = self.normalize(label)

        print(f'Image characteristics: max: {np.max(img)}, min: {np.min(img)}')

        if self.deform:
            img, label = self.apply_deform(img, label)
            # Interpolation can affect pixel values, so need to normalize here
            img = self.normalize(img)
            label = self.normalize(label)

        # Apply brightness augmentations
        if self.brightness is not None:
            img = self.apply_brightness(img)

        print(f'Image characteristics: max: {np.max(img)}, min: {np.min(img)}')
        print(f'Label characteristics: max: {np.max(label)}, min: {np.min(label)}')

        return img, label

    @staticmethod
    def normalize(_img):
        """ Normalize voxel values to be within 0 to 1 range. """
        return (_img - np.min(_img)) / (np.max(_img) - np.min(_img))

    def apply_zoom(self, _img, zoom_factor):
        """
        Apply zoom to 3 dimensional images. Adapted from
        https://stackoverflow.com/questions/37119071/scipy-rotate-and-zoom-an-image-without-changing-its-dimensions
        """
        h, w, d = _img.shape

        # Get a random zoom factor between between the lower and upper bounds of self.zoom
        zoom_tuple = (zoom_factor,) * 3
        # print(f'zoom factor: {zoom_factor}')

        # Zooming out
        if zoom_factor < 1:
            # Bounding box of the zoomed-out image within the output array
            zh = int(np.round(h * zoom_factor))
            zw = int(np.round(w * zoom_factor))
            zd = int(np.round(d * zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
            inside = (d - zd) // 2
            # print(f'zh: {zh}, zw: {zw}, zd: {zd}, top: {top}, left: {left}, inside: {inside}')

            # Zero-padding
            out = np.zeros_like(_img)
            out[top:top + zh, left:left + zw, inside:inside + zd] = snd.zoom(_img, zoom_tuple)

        # Zooming in
        else:
            # Bounding box of the zoomed-in region within the input array
            zh = int(np.round(h / zoom_factor))
            zw = int(np.round(w / zoom_factor))
            zd = int(np.round(d / zoom_factor))
            top = (h - zh) // 2
            left = (w - zw) // 2
            inside = (d - zd) // 2
            # print(f'zh: {zh}, zw: {zw}, zd: {zd}, top: {top}, left: {left}, inside: {inside}')

            out = snd.zoom(_img[top:top + zh, left:left + zw, inside:inside + zd], zoom_tuple)

            # `out` might still be slightly larger than `_img` due to rounding, so
            # trim off any extra pixels at the edges
            trim_top = ((out.shape[0] - h) // 2)
            trim_left = ((out.shape[1] - w) // 2)
            trim_inside = ((out.shape[2] - d) // 2)
            # print(f'trim_top: {trim_top}, trim_left: {trim_left}, trim_inside: {trim_inside}')
            out = out[trim_top:trim_top + h, trim_left:trim_left + w, trim_inside:trim_inside + d]
            # print(out_img.shape)

        return out

    def apply_rotation(self, img, axes, angle):
        """ Apply rotation augmentation to the volume. """
        out = snd.rotate(img, axes=axes, angle=angle, reshape=False)
        return out

    def apply_translate(self, _img, h_factor, w_factor, d_factor):
        """ Translate the image and label simultaneously by a random amount up to self.translate """
        h, w, d = _img.shape

        # Randomly decide whether to shift forwards or backwards in each direction
        plr = [-1 if x < 0.5 else 1 for x in self.rng.uniform(size=3)]
        # print(f'polarities: {plr}')
        # Amount to translate each dimension by, based on self.translate
        th = int(h * h_factor)
        tw = int(w * w_factor)
        td = int(d * d_factor)
        # print(f'th: {th}, tw: {tw}, td: {td}')

        out = np.zeros_like(_img)

        # Shift the cube representing the original image inside the cube representing the 0-padded container
        out[max(th * plr[0], 0):min(h, h + th * plr[0]),
            max(tw * plr[1], 0):min(w, w + tw * plr[1]),
            max(td * plr[2], 0):min(d, d + td * plr[2])] = \
            _img[max(th * -plr[0], 0):min(h, h + th * -plr[0]),
                 max(tw * -plr[1], 0):min(w, w + tw * -plr[1]),
                 max(td * -plr[2], 0):min(d, d + td * -plr[2])]

        return out

    def apply_brightness(self, _img):
        """ Apply brightness augmentations. Note that this only needs to be done for the image, and not the label. """
        # Apply brightness change disproportionately; changing only dark values to prevent changing the blacks to make
        # it harder for the model to understand the bounds of the image
        polarity = -1 if self.rng.uniform() < 0.5 else 1
        brightness_factor = 1 - polarity * self.rng.uniform(0, self.brightness)

        return _img ** brightness_factor

    @staticmethod
    def apply_deform(_img, _label):
        """ Apply B-spline deformation to an image and label simultaneously - the same transform applies to both. """
        # Note that the image and label must be the same size in order to use this
        assert _img.shape == _label.shape, \
            f'Image and label must be the same shape but got image: {_img.shape}, label: {_label.shape}'
        # s = time.time()
        _img, _label = elasticdeform.deform_random_grid([_img, _label], sigma=_img.shape[0] / 500., points=10)
        # print(f'time taken: {time.time() - s} seconds')
        return _img, _label


if __name__ == '__main__':
    # TODO: test all this to see if adding augmentation actually improves performance and report on this in diss
    # TODO: tidy up code
    single = '/home/y4tsu/Desktop/data/mri/imagesTr/la_003.nii'
    label = '/home/y4tsu/Desktop/data/mri/labelsTr/la_003.nii'
    folder = '/home/y4tsu/Desktop/data/mri/imagesTr/'

    reader = NIIReader()
    aug = Augmenter(
        zoom=(0.92, 1.08),
        translate=0.06,
        brightness=0.18,
        deform=True,
        rotate=8
    )

    img = reader.read(single)
    img = reader.normalize(reader.numpy(img))
    label_img = reader.read(label)
    label_img = reader.normalize(reader.numpy(label_img))

    img, label_img = aug.augment(img, label_img)
    plt.figure()
    plt.title('img')
    plt.imshow(img[:, :, 65], cmap='gray')

    plt.figure()
    plt.title('label_img')
    plt.imshow(label_img[:, :, 65], cmap='gray')
    plt.show()
    # reader.scroll_view(img)
