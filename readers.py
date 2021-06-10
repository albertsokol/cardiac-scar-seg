import os

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom
from skimage.transform import resize
from tqdm import tqdm


class IndexTracker:
    """
    Implementation of scrollable 3d images in matplotlib adapted from
    https://matplotlib.org/stable/gallery/event_handling/image_slices_viewer.html
    """
    def __init__(self, ax, img, plane):
        self.ax = ax
        assert plane in ['sagittal', 'coronal', 'transverse'], 'plane must be one of: sagittal, coronal, transverse'
        self.plane = plane
        ax.set_title('use scroll wheel to navigate images')

        self.X = img
        if self.plane == 'coronal':
            self.slices = img.shape[0]
        if self.plane == 'sagittal':
            self.slices = img.shape[1]
        if self.plane == 'transverse':
            self.slices = img.shape[2]

        self.ind = self.slices // 2

        if plane == 'coronal':
            self.im = ax.imshow(self.X[self.ind, :, :], cmap='gray')
        if plane == 'sagittal':
            self.im = ax.imshow(self.X[:, self.ind, :], cmap='gray')
        if plane == 'transverse':
            self.im = ax.imshow(self.X[:, :, self.ind], cmap='gray')

        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        if self.plane == 'coronal':
            self.im.set_data(self.X[self.ind, :, :])
        if self.plane == 'sagittal':
            self.im.set_data(self.X[:, self.ind, :])
        if self.plane == 'transverse':
            self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()


class FileReader:
    """ Generic class for common file reading functions. """
    def __init__(self):
        self.fig, self.ax = plt.subplots(1, 1)

    def scroll_view(self, img, plane='transverse'):
        """ Open a scrollable matplotlib window to look through the given MRI image (must be numpy array). """
        assert type(img) == np.ndarray, 'image must be a numpy array to use for scrolling.'
        tracker = IndexTracker(self.ax, img, plane)
        self.fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
        plt.show()

    @staticmethod
    def chop(img, chop_size, overlap=False, logging=False):
        """ Return a list of MRI chunks (separated up into patches). """
        h, w, d = img.shape
        ch, cw, cd = chop_size
        if h % ch != 0:
            raise AttributeError(f'Chop height {ch} does not evenly split the input height {h}')
        if w % cw != 0:
            raise AttributeError(f'Chop width {cw} does not evenly split the input width {w}')
        if d % cd != 0:
            raise AttributeError(f'Chop depth {cd} does not evenly split the input depth {d}')
        split_h = h // ch
        split_w = w // cw
        split_d = d // cd
        if logging:
            print(f'Split values: split_h: {split_h}, split_w: {split_w}, split_d: {split_d}')
        # Split along the height
        chop_h = []
        if split_h > 1:
            for i in range(split_h):
                if logging:
                    print(f'h chop indices: {ch * i}:{ch * (i + 1)}')
                chop_h += [img[ch * i:ch * (i + 1), :, :]]
            if overlap:
                if logging:
                    print(f'h overlap indices: {ch // 2}:{3 * ch // 2}')
                chop_h += [img[ch // 2:3 * ch // 2, :, :]]
        else:
            chop_h = [img]
        if logging:
            for x in chop_h:
                print(f'chop_h shapes: {x.shape}')
        # Split along the width
        if split_w > 1:
            chop_w = []
            for chop in chop_h:
                for i in range(split_w):
                    if logging:
                        print(f'w chop indices: {cw * i}:{cw * (i + 1)}')
                    chop_w += [chop[:, cw * i:cw * (i + 1), :]]
                if overlap:
                    if logging:
                        print(f'w overlap indices: {cw // 2}:{3 * cw // 2}')
                    chop_w += [chop[:, cw // 2:3 * cw // 2, :]]
        else:
            chop_w = chop_h
        if logging:
            for x in chop_w:
                print(f'chop_w shapes: {x.shape}')
        # Split along the depth
        if split_d > 1:
            result = []
            for chop in chop_w:
                for i in range(split_d):
                    if logging:
                        print(f'd chop indices: {cd * i}:{cd * (i + 1)}')
                    result += [chop[:, :, cd * i:cd * (i + 1)]]
                if overlap:
                    if logging:
                        print(f'd overlap indices: {cd // 2}:{3 * cd // 2}')
                    result += [chop[:, :, cd // 2:3 * cd // 2]]
        else:
            result = chop_w
        if logging:
            for x in result:
                print(f'result shapes: {x.shape}')
            print(f'Total: {len(result)} chops')
        return result

    @staticmethod
    def normalize(img):
        """ Normalize voxel values to be within 0 to 1 range. """
        return (img - np.min(img)) / (np.max(img) - np.min(img))


class NIIReader(FileReader):
    """ Class for reading MRI images in .nii file format. """
    def __init__(self, slice_20=True):
        """

        :param slice_20: bool: take only every 20th slice along the sagittal plane as these are just interpolated
                               in the original dataset
        """
        super().__init__()
        self.slice_20 = slice_20

    @staticmethod
    def resize(img, dims=None, magnify=None, interpolation_order=3):
        """
        Resize to the given proportions. Either dims or magnify can be given, but only one can be specified
        dims: len(3) tuple:
            the dimensions to resize the input image to
        magnify: len(3) tuple:
            the scaling factor to apply to the x, y and z axes
        interpolation_order: int (0 - 5 incl.): the order of spline interpolation, i.e., 3 for cubic, 0 for nearest
            neighbour
        """
        assert type(img) == np.ndarray, 'image must be a numpy array to resize.'
        if dims is None and magnify is None:
            raise AttributeError("Both dims and magnify cannot be none - please select one of these for zooming")
        if dims is not None and magnify is not None:
            raise AttributeError("Both dims and magnify cannot be selected - please specify only one")
        if dims is not None:
            magnify = (dims[0] / img.shape[0], dims[1] / img.shape[1], dims[2] / img.shape[2])
        return zoom(img, magnify, order=interpolation_order)

    @staticmethod
    def numpy(img):
        """ Return the image as a numpy array. """
        return img.get_fdata()

    def read(self, f):
        """ Load the image using nibabel and convert to numpy array. """
        if self.slice_20:
            return self.numpy(nib.load(f, mmap=False))[:, :, ::20]
        else:
            return self.numpy(nib.load(f, mmap=False))


class NPYReader(FileReader):
    """ Class for reading image slices in .npy file format. """
    def __init__(self):
        super().__init__()

    @staticmethod
    def resize(img, dims, interpolation_order=3):
        """ Resize the 2D image to the given dimensions. """
        assert type(img) == np.ndarray, 'image must be a numpy array to resize.'
        return resize(img, dims, order=interpolation_order)

    @staticmethod
    def read(f):
        """ Load the image using numpy. """
        return np.load(f)


if __name__ == '__main__':
    reader = NIIReader()

    # Save all 3D as 2D numpy arrays in each plane to allow shuffled loading during 2D training
    for g in ['train', 'val', 'test']:
        for t in ['image', 'label']:
            image_fnames = [
                os.path.join('/media/y4tsu/ml_data/cmr/3d/', g, t, x)
                for x in sorted(os.listdir(f'/media/y4tsu/ml_data/cmr/3d/{g}/{t}'))
            ]

            for i, name in enumerate(tqdm(image_fnames)):
                img = reader.read(name)
                for j in range(img.shape[0]):
                    np.save(f'/media/y4tsu/ml_data/cmr/2d/{g}/coronal/{t}/{i}_coronal_{j}', img[j, :, :])
                    x = np.load(f'/media/y4tsu/ml_data/cmr/2d/{g}/coronal/{t}/{i}_coronal_{j}.npy')
                for j in range(img.shape[1]):
                    np.save(f'/media/y4tsu/ml_data/cmr/2d/{g}/sagittal/{t}/{i}_sagittal_{j}', img[:, j, :])
                    x = np.load(f'/media/y4tsu/ml_data/cmr/2d/{g}/sagittal/{t}/{i}_sagittal_{j}.npy')
                for j in range(img.shape[2]):
                    np.save(f'/media/y4tsu/ml_data/cmr/2d/{g}/transverse/{t}/{i}_transverse_{j}', img[:, :, j])
                    x = np.load(f'/media/y4tsu/ml_data/cmr/2d/{g}/transverse/{t}/{i}_transverse_{j}.npy')
