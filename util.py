"""Various utility functions for working with the data and models."""
import json
import os

import nibabel as nib
import numpy as np
from matplotlib import pyplot as plt
from matplotlib import rcParams
from scipy.ndimage import rotate
from tqdm import tqdm

from readers import NIIReader, NPYReader


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


def analyse_quality_labels_frequency():
    """Show the frequency of each of the quality labels and save as a plot."""
    with open('quality_scores.json', 'r') as f:
        qs = json.load(f)
    totals = [0, 0, 0]

    for k in qs:
        curr = qs[k]
        for x in curr:
            totals[int(x)] += 1

    print(totals)

    rcParams['font.family'] = 'Ubuntu'

    axs = plt.axes()
    axs.bar([0, 1, 2], totals, color=["sandybrown", "lightcoral", "cadetblue"])
    plt.xlabel("Slice quality score")
    plt.ylabel("Frequency")
    plt.xticks([0, 1, 2])
    plt.title("Frequency of slice quality scores")
    plt.savefig('plot/slice_quality_frequency.pdf', bbox_inches='tight')
    plt.savefig('plot/slice_quality_frequency.png', dpi=300, bbox_inches='tight')


def analyse_quality_labels_class_wise():
    """Show the frequency of each class for each quality label and save as plots."""
    with open('quality_scores.json', 'r') as f:
        qs = json.load(f)

    totals = {
        0: [0] * 8,
        1: [0] * 8,
        2: [0] * 8,
    }

    total_px = {
        0: 0,
        1: 0,
        2: 0,
    }

    reader = NPYReader()
    base_folder = '/media/y4tsu/ml_data/cmr/2D/'

    for dataset in ['train', 'val', 'test']:
        roots = os.listdir(os.path.join(base_folder, dataset, 'transverse'))
        for root in tqdm(roots):
            labels = [x for x in sorted(os.listdir(os.path.join(base_folder, dataset, 'transverse', root))) if 'label' in x]
            for i, x in enumerate(labels):
                label = reader.read(os.path.join(base_folder, dataset, 'transverse', root, x))
                q_key = root.split('_')[1]
                q = qs[q_key][i]
                total_px[q] += label.shape[0] * label.shape[1]
                for j in range(8):
                    totals[q][j] += np.where(np.equal(label, j), 1, 0).sum()

    labels = ["lv lumen", "lv myo", "scar", "rv lumen", "rv myo", "pap", "aorta"]
    for k in total_px:
        total_px[k] -= totals[k][0]
        totals[k].pop(0)
        totals[k] = [100. * x / total_px[k] for x in totals[k]]

    # Below portion adapted from https://www.python-graph-gallery.com/11-grouped-barplot
    # set width of bars
    rcParams['font.family'] = 'Ubuntu'
    bar_width = 0.25

    # set heights of bars
    q0, q1, q2 = totals[0], totals[1], totals[2]

    # Set position of bar on X axis
    r1 = np.arange(len(q0))
    r2 = [x + bar_width for x in r1]
    r3 = [x + bar_width for x in r2]

    # Make the plot
    plt.bar(r1, q0, color='sandybrown', width=bar_width, edgecolor='white', label='0')
    plt.bar(r2, q1, color='lightcoral', width=bar_width, edgecolor='white', label='1')
    plt.bar(r3, q2, color='cadetblue', width=bar_width, edgecolor='white', label='2')

    # Add xticks on the middle of the group bars
    plt.xlabel('Label')
    plt.ylabel('Proportion of label out of all pixels (%)')
    plt.xticks([r + bar_width for r in range(len(q0))], labels)
    plt.title('Proportion of label representation per quality score')

    # Create legend & Show graphic
    plt.legend(title='Quality score')
    plt.savefig('plot/slice_quality_class_wise.pdf', bbox_inches='tight')
    plt.savefig('plot/slice_quality_class_wise.png', dpi=300, bbox_inches='tight')


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
    analyse_quality_labels_class_wise()
