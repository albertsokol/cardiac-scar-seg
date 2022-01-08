"""Various utility functions for working with the data and models."""
import json
import os

import nibabel as nib
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import stats
from scipy.ndimage import rotate
from tqdm import tqdm

from scarseg.cropper import Cropper
from scarseg.readers import NIIReader, NPYReader


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

    axs = plt.axes()
    axs.bar([0, 1, 2], totals, color=["sandybrown", "lightcoral", "cadetblue"])
    plt.xlabel("Slice certainty score")
    plt.ylabel("Frequency")
    plt.xticks([0, 1, 2])
    plt.title("Frequency of slice certainty scores")
    plt.savefig('plot/slice_certainty_frequency.png', dpi=300, bbox_inches='tight')


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
    base_folder = '/media/y4tsu/ml-fast/cmr/2D/'

    for dataset in ['train', 'val']:
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
    plt.title('Proportion of label representation per certainty score')

    # Create legend & Show graphic
    plt.legend(title='Certainty score')
    plt.savefig('plot/slice_certainty_class_wise.png', dpi=300, bbox_inches='tight')


def rotate_incorrect_orientations():
    """
    Rotate a list of manually specified images by 90 degrees anti-clockwise so that all images are in the same
    orientation to simplify the learning task.
    """
    reader = NIIReader()
    base_folder = '/media/y4tsu/ml-fast/cmr/3D/test'

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


def create_2d_dataset(root):
    """
    Create the 2D images and labels slices so that they can be loaded quickly in a shuffled order during training.

    :param root: str: path to the data root (one level up from the '3D' folder)
    """
    print(f'Creating 2D dataset from data at {os.path.join(root, "3D")}')

    assert not os.path.exists(os.path.join(root, '2D')), f"please delete the {root}/2D folder before proceeding"
    os.mkdir(os.path.join(root, '2D'))
    os.mkdir(os.path.join(root, '2D', 'train'))
    os.mkdir(os.path.join(root, '2D', 'val'))
    os.mkdir(os.path.join(root, '2D', 'train', 'transverse'))
    os.mkdir(os.path.join(root, '2D', 'val', 'transverse'))
    os.mkdir(os.path.join(root, '2D', 'train', 'coronal'))
    os.mkdir(os.path.join(root, '2D', 'val', 'coronal'))
    os.mkdir(os.path.join(root, '2D', 'train', 'sagittal'))
    os.mkdir(os.path.join(root, '2D', 'val', 'sagittal'))

    reader = NIIReader()
    for g in ['train', 'val']:
        for x in tqdm(sorted(os.listdir(os.path.join(root, '3D', g)))):
            image_fname = os.path.join(root, '3D', g, x, f'{x}_SAX.nii.gz')
            label_fname = os.path.join(root, '3D', g, x, f'{x}_SAX_mask2.nii.gz')

            image = reader.read(image_fname)
            label = reader.read(label_fname)

            os.mkdir(os.path.join(root, '2D', g, 'coronal', x))
            os.mkdir(os.path.join(root, '2D', g, 'sagittal', x))
            os.mkdir(os.path.join(root, '2D', g, 'transverse', x))

            for j in range(image.shape[0]):
                np.save(os.path.join(root, '2D', g, 'coronal', f'{x}/{x}_{j:03}_image'), image[j, :, :])
                np.save(os.path.join(root, '2D', g, 'coronal', f'{x}/{x}_{j:03}_label'), label[j, :, :])
            for j in range(image.shape[1]):
                np.save(os.path.join(root, '2D', g, 'sagittal', f'{x}/{x}_{j:03}_image'), image[:, j, :])
                np.save(os.path.join(root, '2D', g, 'sagittal', f'{x}/{x}_{j:03}_label'), label[:, j, :])
            for j in range(image.shape[2]):
                np.save(os.path.join(root, '2D', g, 'transverse', f'{x}/{x}_{j:03}_image'), image[:, :, j])
                np.save(os.path.join(root, '2D', g, 'transverse', f'{x}/{x}_{j:03}_label'), label[:, :, j])


def create_3dshallow_dataset(root, depth=3):
    """
    Create 3D shallow images and save with the given depth to allow fast shuffled loading during training.

    :param root: str: path to the data root (one level up from the '3D' folder)
    :param depth: int: slice thickness to create for images
    """
    assert depth > 1 and depth % 2 == 1, f"depth must be an odd number greater than 1 but got {depth}"
    print(f'Creating depth={depth} 3DShallow dataset from data at {os.path.join(root, "3D")}')
    reader = NIIReader()

    assert not os.path.exists(os.path.join(root, '3DShallow')), f"please delete the {root}/3DShallow folder before proceeding"
    os.mkdir(os.path.join(root, '3DShallow'))
    os.mkdir(os.path.join(root, '3DShallow', 'train'))
    os.mkdir(os.path.join(root, '3DShallow', 'val'))
    os.mkdir(os.path.join(root, '3DShallow', 'train', 'transverse'))
    os.mkdir(os.path.join(root, '3DShallow', 'val', 'transverse'))

    for g in ['train', 'val']:
        for x in tqdm(sorted(os.listdir(os.path.join(root, '3D', g)))):
            image_fname = os.path.join(root, '3D', g, x, f'{x}_SAX.nii.gz')
            label_fname = os.path.join(root, '3D', g, x, f'{x}_SAX_mask2.nii.gz')

            image = reader.read(image_fname)
            label = reader.read(label_fname)

            os.mkdir(os.path.join(root, '3DShallow', g, 'transverse', x))

            for j in range(image.shape[2] - depth + 1):
                np.save(os.path.join(root, '3DShallow', g, 'transverse', f'{x}/{x}_{j:03}_image'), image[:, :, j:j + depth])
                np.save(os.path.join(root, '3DShallow', g, 'transverse', f'{x}/{x}_{j:03}_label'), label[:, :, j:j + depth])


def find_depth_distributions(root):
    """Get the depths of all images in the dataset and display as a distribution."""
    depths = []
    reader = NIIReader()

    for g in ['train', 'val']:
        for x in tqdm(sorted(os.listdir(os.path.join(root, '3D', g)))):
            image_fname = os.path.join(root, '3D', g, x, f'{x}_SAX.nii.gz')
            image = reader.read(image_fname)
            depths += [image.shape[-1]]

    plt.hist(depths, bins=np.arange(9, 17) - 0.5, rwidth=0.75, color='cadetblue')
    plt.xlabel('Depth of image (number of slices)')
    plt.ylabel('Frequency')
    plt.title('Frequency of image depths')

    # Create legend & Show graphic
    plt.savefig('/home/y4tsu/Desktop/diss img/depth_frequencies.png', bbox_inches='tight', dpi=300)
    plt.savefig('/home/y4tsu/Desktop/diss img/depth_frequencies.pdf', bbox_inches='tight')
    plt.show()


def find_widths_distributions(root):
    """Get the widths of all images (square) in the dataset and display as a distribution."""
    depths = []
    reader = NIIReader()

    for g in ['train', 'val']:
        for x in tqdm(sorted(os.listdir(os.path.join(root, '3D', g)))):
            image_fname = os.path.join(root, '3D', g, x, f'{x}_SAX.nii.gz')
            image = reader.read(image_fname)
            depths += [image.shape[0]]

    plt.hist(depths, bins=np.arange(min(depths), max(depths) + 2) - 0.5, rwidth=1, color='cadetblue')
    plt.xlabel('Width of image')
    plt.ylabel('Frequency')
    plt.title('Frequency of image widths')

    # Create legend & Show graphic
    plt.savefig('/home/y4tsu/Desktop/diss img/width_frequencies.png', bbox_inches='tight', dpi=300)
    plt.savefig('/home/y4tsu/Desktop/diss img/width_frequencies.pdf', bbox_inches='tight')
    plt.show()


def find_post_crop_distribution_sizes(root):
    """Get the average size of the images using manual crop."""
    val_cropper = Cropper(root, "val", "manual")
    train_cropper = Cropper(root, "train", "manual")
    all_bboxes = {**val_cropper.bboxes, **train_cropper.bboxes}
    widths = []
    lengths = []
    for k in all_bboxes:
        widths += [all_bboxes[k]["bottom"] - all_bboxes[k]["top"] + 16]
        lengths += [all_bboxes[k]["right"] - all_bboxes[k]["left"] + 16]
        print(all_bboxes[k])
    print(widths)
    print(lengths)

    plt.hist(widths, bins=16, rwidth=0.8, color='cadetblue')
    plt.xlabel('Width of image after manual crop')
    plt.ylabel('Frequency')
    plt.title('Frequency of image widths after applying manual cropping')

    # Create legend & Show graphic
    plt.savefig('/home/y4tsu/Desktop/diss img/width_cropped_frequencies.png', bbox_inches='tight', dpi=300)
    plt.savefig('/home/y4tsu/Desktop/diss img/width_cropped_frequencies.pdf', bbox_inches='tight')
    plt.show()


def plot_extra_metrics():
    plt.rcParams.update({'font.size': 7})

    df = pd.read_csv("volume_hd_metrics.csv")
    print(df.head(5))
    plt.scatter(df["ground truth scar volume"], df["predicted scar volume"], s=10)
    plt.xlabel("Ground truth scar volume (mm3)")
    plt.ylabel("Predicted scar volume (mm3)")
    pearson = stats.pearsonr(df['ground truth scar volume'], df['predicted scar volume'])
    plt.title(f"Predicted vs. ground truth scar volume\n"
              f"Pearson coefficient = {pearson[0]:.4f}")
    plt.savefig('/home/y4tsu/Desktop/diss img/predicted vs ground truth scar volume correlation.png', bbox_inches='tight', dpi=300)
    plt.show()

    min_bin = min(min(df["ground truth scar volume"]), min(df["predicted scar volume"]))
    max_bin = max(max(df["ground truth scar volume"]), max(df["predicted scar volume"]))
    plt.hist([df["ground truth scar volume"], df["predicted scar volume"]], bins=16, rwidth=0.8, color=['cadetblue', 'orange'], range=(min_bin, max_bin))
    plt.xlabel("Scar volume (mm3)")
    plt.ylabel("Frequency")
    plt.title("Frequency of scar volumes for ground truth and predicted scars\n"
              f"Median ground truth scar volume = {np.median(df['ground truth scar volume']):.0f}\n"
              f"Median predicted scar volume = {np.median(df['predicted scar volume']):.0f}"
              )
    plt.legend(["Ground truth scar", "Predicted scar"], title="Legend")
    plt.savefig('/home/y4tsu/Desktop/diss img/predicted vs ground truth scar volume distributions.png', bbox_inches='tight', dpi=300)
    plt.show()

    plt.scatter(df["ground truth scar volume"], df["hausdorff distance"], s=10)
    plt.xlabel("Ground truth scar volume (mm3)")
    plt.ylabel("Hausdorff distance (mm)")
    pearson = stats.pearsonr(df['ground truth scar volume'], df['hausdorff distance'])
    plt.title(f"Hausdorff distance vs. ground truth scar volume\n"
              f"Pearson coefficient = {pearson[0]:.4f}")
    plt.savefig('/home/y4tsu/Desktop/diss img/hausdorff distance vs ground truth scar volume.png', bbox_inches='tight', dpi=300)
    plt.show()


def find_wilcoxon(path_a, path_b):
    """Get statistical significance of differences between tissues for 2 models."""
    a_scar_dice = pd.read_csv(path_a)
    b_scar_dice = pd.read_csv(path_b)
    # TODO: remove skip nums from both dataframes

    skip_files = ["20CA015_N055", "20CA015_N110", "20CA015_N207", "20CA015_N051", "20CA015_N109"]
    a_scar_dice = a_scar_dice[~a_scar_dice["file"].isin(skip_files)]
    b_scar_dice = b_scar_dice[~b_scar_dice["file"].isin(skip_files)]

    a_scar_dice = a_scar_dice["scar"]
    b_scar_dice = b_scar_dice["scar"]

    d = b_scar_dice - a_scar_dice
    w, p = stats.wilcoxon(d)
    print(f'Result (NULL): w={w}, p={p}')


if __name__ == '__main__':
    # create_2d_dataset('/media/y4tsu/ml-fast/cmr_fold_5')
    # create_3dshallow_dataset('/media/y4tsu/ml-fast/cmr_fold_5', depth=5)
    # find_widths_distributions('/media/y4tsu/ml-fast/cmr')
    # search_certainties('/media/y4tsu/ml-fast/cmr')
    # analyse_quality_labels_class_wise()
    # analyse_quality_labels_frequency()
    # plot_extra_metrics()
    find_wilcoxon("dice_res/3D_frozen_dices_1.csv", "dice_res/3D_frozen_dices_plus_certainty_1.csv")