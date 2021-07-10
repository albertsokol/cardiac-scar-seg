"""Various utility functions for working with the data and models."""
import os

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from tqdm import tqdm

from readers import NIIReader


class Cropper:
    """Class for cropping images and labels."""
    def __init__(self, dataset, mode, pad=8):
        """Initializer for Cropper."""
        self.dataset = dataset
        if mode == "manual":
            self.bboxes = self.get_manual_bboxes()
        self.pad = pad

    def get_manual_bboxes(self):
        """Load the manual segmentation masks from 3D images to get bounding boxes on the fly."""
        # WARNING!: not tested on non-square images, some of the axes in the top/left/bottom/right detection may be wrong
        print(f'Loading bboxes from manual segmentations for {self.dataset} data ... ')
        reader = NIIReader()
        base_folder = f'/media/y4tsu/ml_data/cmr/3D/{self.dataset}'
        roots = sorted(os.listdir(base_folder))
        out = {}

        for i, root in enumerate(tqdm(roots)):
            label = np.squeeze(reader.read(os.path.join(base_folder, root, f'{root}_SAX_mask2.nii.gz')))
            binary_label = np.where(np.equal(label, 0), 0, 1)

            # Order is: top, left, bottom, right
            bbox = [label.shape[1] // 2, label.shape[0] // 2, label.shape[1] // 2, label.shape[0] // 2]

            # Get top, left, bottom and right extents of the label on each slice
            for j in range(binary_label.shape[2]):
                curr = binary_label[..., j]
                try:
                    top = np.where(np.any(curr == 1, axis=1))[0][0]
                except IndexError:
                    continue
                left = np.where(np.any(curr == 1, axis=0))[0][0]
                bottom = curr.shape[0] - np.where(np.any(curr[::-1, ...] == 1, axis=1))[0][0]
                right = curr.shape[1] - np.where(np.any(curr[..., ::-1] == 1, axis=0))[0][0]
                if top < bbox[0]:
                    bbox[0] = top
                if left < bbox[1]:
                    bbox[1] = left
                if bottom > bbox[2]:
                    bbox[2] = bottom
                if right > bbox[3]:
                    bbox[3] = right

            # print(f'dims: {bbox[3] - bbox[1]} x {bbox[2] - bbox[0]}')
            out[root] = {
                'top': bbox[0],
                'left': bbox[1],
                'bottom': bbox[2],
                'right': bbox[3],
            }

        return out

    def manual_crop(self, f, fname):
        """Crop the given image or label file, using the file name to get the bounding box info."""
        # Get the bounding box
        bbox = self.bboxes[fname.split('/')[-1].split('.')[0][:12]]
        top, left, bottom, right = [int(x) for x in (bbox['top'], bbox['left'], bbox['bottom'], bbox['right'])]
        # print(f'top: {top}, left: {left}, bottom: {bottom}, right: {right}')

        # Find the longest dimension (width or height)
        w = right - left
        h = bottom - top
        longest = max(w, h)

        if w == longest:
            # Find the amount to add to top and bottom; if odd, will need to add an extra px
            if (w - h) % 2 != 0:
                add_top = ((w - h) // 2) + 1
                add_bottom = (w - h) // 2
            else:
                add_top = (w - h) // 2
                add_bottom = (w - h) // 2
            # Take the slice of the image
            # print(f'add_top: {add_top}, add_bottom: {add_bottom}')
            cut = f[
                  max(top - self.pad - add_top, 0):min(bottom + self.pad + add_bottom, f.shape[0]),
                  max(left - self.pad, 0):min(right + self.pad, f.shape[1]),
                  ...
                  ]
            # Need to check if going out of bounds
            if cut.shape[0] != cut.shape[1]:
                if top - self.pad - add_top < 0:
                    # Add some zero padding to the top to fill up the gap
                    bleed = (top - self.pad - add_top) * -1
                    if len(f.shape) > 2:
                        padding = np.zeros([bleed, cut.shape[1], f.shape[2]])
                    else:
                        padding = np.zeros([bleed, cut.shape[1]])
                    cut = np.concatenate((padding, cut), axis=0)
                if left - self.pad < 0:
                    # Add some zero padding to the left
                    bleed = (left - self.pad) * -1
                    if len(f.shape) > 2:
                        padding = np.zeros([cut.shape[0], bleed, f.shape[2]])
                    else:
                        padding = np.zeros([cut.shape[0], bleed])
                    cut = np.concatenate((padding, cut), axis=1)
                if bottom + self.pad + add_bottom > f.shape[0]:
                    # Add some zero padding to the bottom
                    bleed = (f.shape[0] - bottom - self.pad - add_bottom) * -1
                    if len(f.shape) > 2:
                        padding = np.zeros([bleed, cut.shape[1], f.shape[2]])
                    else:
                        padding = np.zeros([bleed, cut.shape[1]])
                    cut = np.concatenate((cut, padding), axis=0)
                if right + self.pad > f.shape[1]:
                    # Add some zero padding to the right
                    bleed = (f.shape[1] - right - self.pad) * -1
                    if len(f.shape) > 2:
                        padding = np.zeros([cut.shape[0], bleed, f.shape[2]])
                    else:
                        padding = np.zeros([cut.shape[0], bleed])
                    cut = np.concatenate((cut, padding), axis=1)
        else:
            if (h - w) % 2 != 0:
                add_left = ((h - w) // 2) + 1
                add_right = (h - w) // 2
            else:
                add_left = (h - w) // 2
                add_right = (h - w) // 2
            # print(f'add_left: {add_left}, add_right: {add_right}')
            cut = f[
                  max(top - self.pad, 0):min(bottom + self.pad, f.shape[0]),
                  max(left - self.pad - add_left, 0):min(right + self.pad + add_right, f.shape[1]),
                  ...
                  ]
            # Need to check if going out of bounds
            if cut.shape[0] != cut.shape[1]:
                if top - self.pad < 0:
                    # Add some zero padding to the top to fill up the gap
                    bleed = (top - self.pad) * -1
                    if len(f.shape) > 2:
                        padding = np.zeros([bleed, cut.shape[1], f.shape[2]])
                    else:
                        padding = np.zeros([bleed, cut.shape[1]])
                    cut = np.concatenate((padding, cut), axis=0)
                if left - self.pad - add_left < 0:
                    # Add some zero padding to the left
                    bleed = (left - self.pad - add_left) * -1
                    if len(f.shape) > 2:
                        padding = np.zeros([cut.shape[0], bleed, f.shape[2]])
                    else:
                        padding = np.zeros([cut.shape[0], bleed])
                    cut = np.concatenate((padding, cut), axis=1)
                if bottom + self.pad > f.shape[0]:
                    # Add some zero padding to the bottom
                    bleed = (f.shape[0] - bottom - self.pad) * -1
                    if len(f.shape) > 2:
                        padding = np.zeros([bleed, cut.shape[1], f.shape[2]])
                    else:
                        padding = np.zeros([bleed, cut.shape[1]])
                    cut = np.concatenate((cut, padding), axis=0)
                if right + self.pad + add_right > f.shape[1]:
                    # Add some zero padding to the right
                    bleed = (f.shape[1] - right - self.pad - add_right) * -1
                    if len(f.shape) > 2:
                        padding = np.zeros([cut.shape[0], bleed, f.shape[2]])
                    else:
                        padding = np.zeros([cut.shape[0], bleed])
                    cut = np.concatenate((cut, padding), axis=1)

        return cut


def rotate_incorrect_orientations():
    """
    Rotate a list of manually specified images by 90 degrees anti-clockwise so that all images are in the same
    orientation to simplify the learning task.
    """
    reader = NIIReader()
    base_folder = '/media/y4tsu/ml_data/cmr/3D/test'

    roots = sorted(os.listdir(base_folder))
    ax_thickness = []
    hw = []
    squares = 0
    print(f'{len(roots)} total scans found')

    for i, root in enumerate(roots):
        image = np.squeeze(reader.read(os.path.join(base_folder, root, f'{root}_SAX.nii.gz')))
        shape = image.shape
        label = np.squeeze(reader.read(os.path.join(base_folder, root, f'{root}_SAX_mask2.nii.gz')))

        print(root)
        if root in ['20CA015_N008', '20CA015_N083', '20CA015_N089', '20CA015_N138', '20CA015_N141', '20CA015_N144',
                    '20CA015_N174', '20CA015_N307', '20CA015_N319', '20CA015_N227']:
            # Show the original image
            plt.imshow(label[:, :, 5], cmap='gray')
            plt.show()

            # Rotate the bad images and labels
            rot_image = rotate(image, axes=(0, 1), angle=-90., reshape=False, order=3)
            rot_label = rotate(label, axes=(0, 1), angle=-90., reshape=False, order=0)

            # Set them as nifti type images
            new_img = nib.Nifti1Image(rot_image, np.eye(4))
            new_label = nib.Nifti1Image(rot_label, np.eye(4))

            # Save them, overwriting the original files
            nib.save(new_img, os.path.join(base_folder, root, f'{root}_SAX.nii.gz'))
            nib.save(new_label, os.path.join(base_folder, root, f'{root}_SAX_mask2.nii.gz'))

            # Show the newly rotated image
            plt.imshow(rot_label[:, :, 5], cmap='gray')
            plt.show()

        if shape[0] != shape[1]:
            print(f'{root} is non-square')

        hw += [shape[0]]
        ax_thickness += [shape[2]]
        squares += 1 if shape[0] == shape[1] else 0

    print(f'num squares: {squares}')
    fig, axs = plt.subplots(2, 1)
    axs[0].hist(ax_thickness)
    axs[1].hist(hw)
    plt.show()


if __name__ == '__main__':
    pass
