"""Various utility functions for working with the data and models."""
import os
import csv

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from scipy.ndimage import rotate
from tqdm import tqdm

from readers import NIIReader


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


def get_bounding_boxes(dataset='train'):
    """
    Create a file with the bounding box info from the manual segmentations which can be used for image cropping.
    """
    # WARNING!: not tested on non-square images, some of the axes in the top/left/bottom/right detection may be wrong
    reader = NIIReader()
    base_folder = f'/media/y4tsu/ml_data/cmr/3D/{dataset}'
    roots = sorted(os.listdir(base_folder))
    out = [['fname', 'top', 'left', 'bottom', 'right']]

    for i, root in enumerate(tqdm(roots)):
        if i == 0:
            continue
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

        out += [[root, *bbox]]

    with open(f'bbox_{dataset}.csv', 'w') as f:
        writer = csv.writer(f)
        writer.writerows(out)


if __name__ == '__main__':
    get_bounding_boxes()
