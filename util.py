"""Various utility functions for working with the data and models."""
import json
import os

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate

from readers import NIIReader


class PColour:
    """Strings denoting pre-saved colours for printing warnings etc."""
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


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
