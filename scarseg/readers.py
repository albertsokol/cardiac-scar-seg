import cv2
import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import zoom


class IndexTracker:
    """
    Implementation of scrollable 3d images in matplotlib adapted from
    https://matplotlib.org/stable/gallery/event_handling/image_slices_viewer.html
    """

    def __init__(self, ax, img, plane):
        self.ax = ax
        assert plane in [
            "sagittal",
            "coronal",
            "transverse",
        ], "plane must be one of: sagittal, coronal, transverse"
        self.plane = plane
        ax.set_title("use scroll wheel to navigate images")

        self.X = img
        if self.plane == "coronal":
            self.slices = img.shape[0]
        if self.plane == "sagittal":
            self.slices = img.shape[1]
        if self.plane == "transverse":
            self.slices = img.shape[2]

        self.ind = self.slices // 2

        if plane == "coronal":
            self.im = ax.imshow(self.X[self.ind, :, :], cmap="gray")
        if plane == "sagittal":
            self.im = ax.imshow(self.X[:, self.ind, :], cmap="gray")
        if plane == "transverse":
            self.im = ax.imshow(self.X[:, :, self.ind], cmap="gray")

        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == "up":
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        if self.plane == "coronal":
            self.im.set_data(self.X[self.ind, :, :])
        if self.plane == "sagittal":
            self.im.set_data(self.X[:, self.ind, :])
        if self.plane == "transverse":
            self.im.set_data(self.X[:, :, self.ind])
        self.ax.set_ylabel("slice %s" % self.ind)
        self.im.axes.figure.canvas.draw()


class __FileReader:
    """Generic class for common file reading functions."""

    def __init__(self):
        pass

    @staticmethod
    def scroll_view(img, plane="transverse"):
        """Open a scrollable matplotlib window to look through the given MRI image (must be numpy array)."""
        fig, ax = plt.subplots(1, 1)
        assert (
            type(img) == np.ndarray
        ), "image must be a numpy array to use for scrolling."
        tracker = IndexTracker(ax, img, plane)
        fig.canvas.mpl_connect("scroll_event", tracker.on_scroll)
        plt.show()

    @staticmethod
    def normalize(img):
        """Normalize voxel values to be within 0 to 1 range."""
        normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
        if np.isnan(np.max(normalized)):
            return np.zeros(img.shape)
        return normalized

    def read(self, f):
        raise NotImplementedError


class NIIReader(__FileReader):
    """Class for reading MRI images in .nii file format."""

    def __init__(self):
        super().__init__()

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
        assert type(img) == np.ndarray, "image must be a numpy array to resize."
        if dims is None and magnify is None:
            raise AttributeError(
                "Both dims and magnify cannot be none - please select one of these for zooming"
            )
        if dims is not None and magnify is not None:
            raise AttributeError(
                "Both dims and magnify cannot be selected - please specify only one"
            )
        if dims is not None:
            magnify = tuple([x / y for x, y in zip(dims, img.shape)])
        return zoom(img, magnify, order=interpolation_order)

    @staticmethod
    def numpy(img):
        """Return the image as a numpy array."""
        return img.get_fdata()

    def read(self, f):
        """Load the image using nibabel and convert to numpy array."""
        return np.squeeze(self.numpy(nib.load(f, mmap=False)))


class NPYReader(__FileReader):
    """Class for reading image slices in .npy file format."""

    def __init__(self):
        super().__init__()

    @staticmethod
    def resize(img, dims, interpolation_order=3):
        """Resize the image to the given dimensions."""
        assert type(img) == np.ndarray, "image must be a numpy array to resize."
        # If 2D then use open CV for resizing
        if len(dims) == 2:
            if interpolation_order == 3:
                return cv2.resize(
                    img, tuple(reversed(dims)), interpolation=cv2.INTER_CUBIC
                )
            elif interpolation_order == 0:
                return cv2.resize(
                    img, tuple(reversed(dims)), interpolation=cv2.INTER_NEAREST
                )
        # Otherwise use scipy zoom
        else:
            magnify = tuple([x / y for x, y in zip(dims, img.shape)])
            return zoom(img, magnify, order=interpolation_order)

    def read(self, f):
        """Load the image using numpy."""
        return np.load(f)
