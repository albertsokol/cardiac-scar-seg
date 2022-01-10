"""Various utility functions for working with the data and models."""
import json
import os
import random
import shutil
from pathlib import Path

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

    HEADER = "\033[95m"
    OKBLUE = "\033[94m"
    OKCYAN = "\033[96m"
    OKGREEN = "\033[92m"
    WARNING = "\033[93m"
    FAIL = "\033[91m"
    ENDC = "\033[0m"
    BOLD = "\033[1m"
    UNDERLINE = "\033[4m"


def analyse_quality_labels_frequency():
    """Show the frequency of each of the quality labels and save as a plot."""
    with open("quality_scores.json", "r") as f:
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
    plt.savefig("plot/slice_certainty_frequency.png", dpi=300, bbox_inches="tight")


def analyse_quality_labels_class_wise():
    """Show the frequency of each class for each quality label and save as plots."""
    with open("quality_scores.json", "r") as f:
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
    base_folder = "/media/y4tsu/ml-fast/cmr/2D/"

    for dataset in ["train", "val"]:
        roots = os.listdir(os.path.join(base_folder, dataset, "transverse"))
        for root in tqdm(roots):
            labels = [
                x
                for x in sorted(
                    os.listdir(os.path.join(base_folder, dataset, "transverse", root))
                )
                if "label" in x
            ]
            for i, x in enumerate(labels):
                label = reader.read(
                    os.path.join(base_folder, dataset, "transverse", root, x)
                )
                q_key = root.split("_")[1]
                q = qs[q_key][i]
                total_px[q] += label.shape[0] * label.shape[1]
                for j in range(8):
                    totals[q][j] += np.where(np.equal(label, j), 1, 0).sum()

    labels = ["lv lumen", "lv myo", "scar", "rv lumen", "rv myo", "pap", "aorta"]
    for k in total_px:
        total_px[k] -= totals[k][0]
        totals[k].pop(0)
        totals[k] = [100.0 * x / total_px[k] for x in totals[k]]

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
    plt.bar(r1, q0, color="sandybrown", width=bar_width, edgecolor="white", label="0")
    plt.bar(r2, q1, color="lightcoral", width=bar_width, edgecolor="white", label="1")
    plt.bar(r3, q2, color="cadetblue", width=bar_width, edgecolor="white", label="2")

    # Add xticks on the middle of the group bars
    plt.xlabel("Label")
    plt.ylabel("Proportion of label out of all pixels (%)")
    plt.xticks([r + bar_width for r in range(len(q0))], labels)
    plt.title("Proportion of label representation per certainty score")

    # Create legend & Show graphic
    plt.legend(title="Certainty score")
    plt.savefig("plot/slice_certainty_class_wise.png", dpi=300, bbox_inches="tight")


def get_necessary_files_only():
    """Save to disk all the necessary files from the original data archive."""
    base_folder = Path("/media/y4tsu/4B172BDA26AB3054/cmr_full")
    roots = sorted([x for x in base_folder.iterdir() if x.is_dir()])
    print(f"{len(roots)=}")

    new_base_folder = Path("/media/y4tsu/4B172BDA26AB3054/cmr_clean")
    new_base_folder.mkdir(parents=False, exist_ok=True)
    copied = 0

    for root in tqdm(roots):
        file_num = root.stem[-3:]
        label_path = root / "Assessor 2" / f"20CA015_N{file_num}_SAX_mask2.nii.gz"
        if not label_path.is_file():
            print(f"Label file not found: {label_path}")
            continue
        image_path = root / f"20CA015_N{file_num}_SAX.nii.gz"
        if not image_path.is_file():
            print(f"Image file not found: {image_path}")
            continue
        new_root_folder = new_base_folder / f"20CA015_N{file_num}"
        new_root_folder.mkdir(parents=False, exist_ok=True)
        shutil.copy(label_path, new_root_folder / label_path.name)
        shutil.copy(image_path, new_root_folder / image_path.name)
        copied += 1

    print(f"Finished copying {copied} files.")


def search_incorrect_orientations():
    """
    Look through all the available images to find those which are malrotated with respect to the majority so that they
    can be fixed.
    """
    reader = NIIReader()
    base_folder = Path("/media/y4tsu/4B172BDA26AB3054/cmr_clean")
    roots = sorted([x for x in base_folder.iterdir() if x.is_dir()])
    print(f"{len(roots)=}")

    for root in roots:
        print(f"Reading root {root}")
        file_num = root.stem[-3:]
        image = np.squeeze(reader.read(root / f"20CA015_N{file_num}_SAX.nii.gz"))
        print(f"{image[:20, ...].sum()=}")
        reader.scroll_view(image)


def rotate_incorrect_orientations():
    """
    Rotate a list of manually specified images by 90 degrees anti-clockwise so that all images are in the same
    orientation to simplify the learning task.
    """
    reader = NIIReader()
    base_folder = Path("/media/y4tsu/4B172BDA26AB3054/cmr_clean")
    roots = sorted([x for x in base_folder.iterdir() if x.is_dir()])
    print(f"{len(roots)=}")

    non_squares = []

    for i, root in enumerate(roots):
        file_num = root.stem[-3:]
        image = np.squeeze(reader.read(root / f"20CA015_N{file_num}_SAX.nii.gz"))
        shape = image.shape
        label = np.squeeze(reader.read(root / f"20CA015_N{file_num}_SAX_mask2.nii.gz"))

        if shape[0] != shape[1]:
            print(f"{root} is non-square")
            non_squares += [root]

        if file_num in [
            "008",
            "014",
            "024",
            "030",
            "062",
            "064",
            "083",
            "089",
            "135",
            "138",
            "141",
            "144",
            "156",
            "159",
            "168",
            "174",
            "181",
            "192",
            "213",
            "215",
            "227",
            "262",
            "278",
            "294",
            "304",
            "307",
            "319",
            "330",
            "347",
            "348",
            "353",
            "355",
            "375",
        ]:
            # Show the original image
            # plt.imshow(label[:, :, 5], cmap="gray")
            # plt.show()

            # Rotate the bad images and labels
            rot_image = rotate(image, axes=(0, 1), angle=-90.0, reshape=False, order=3)
            rot_label = rotate(label, axes=(0, 1), angle=-90.0, reshape=False, order=0)

            # Set them as nifti type images
            new_img = nib.Nifti1Image(rot_image, np.eye(4))
            new_label = nib.Nifti1Image(rot_label, np.eye(4))

            # Save them, overwriting the original files
            nib.save(new_img, root / f"20CA015_N{file_num}_SAX.nii.gz")
            nib.save(new_label, root / f"20CA015_N{file_num}_SAX_mask2.nii.gz")

            # Show the newly rotated image
            # plt.imshow(rot_label[:, :, 5], cmap="gray")
            # plt.show()

    print(f"{len(non_squares)=}")


def create_3d_dataset():
    """From the clean image folder, create a 5-fold cross validation random split dataset."""
    dataset_folder = Path("/media/y4tsu/4B172BDA26AB3054/cmr_folds")
    dataset_folder.mkdir(parents=False, exist_ok=True)
    base_folder = Path("/media/y4tsu/4B172BDA26AB3054/cmr_clean")
    roots = [x for x in base_folder.iterdir() if x.is_dir()]
    random.seed(99)
    random.shuffle(roots)
    print(f"{len(roots)=}")
    print(roots)
    num_folds = 5
    fold_size = len(roots) // 5
    extras = len(roots) - (fold_size * num_folds)

    for i in range(num_folds):
        fold_folder = dataset_folder / str(i)
        fold_folder.mkdir(parents=False, exist_ok=True)
        curr_data_folder = fold_folder / "3D"
        curr_data_folder.mkdir(parents=False, exist_ok=True)
        curr_train_folder = curr_data_folder / "train"
        curr_train_folder.mkdir(parents=False, exist_ok=True)
        curr_val_folder = curr_data_folder / "val"
        curr_val_folder.mkdir(parents=False, exist_ok=True)

        curr_val = roots[i * fold_size : (i + 1) * fold_size]
        print(f"{len(curr_val)=}")
        for root in curr_val:
            shutil.copytree(root, curr_val_folder / root.name)

        curr_train = [root for root in roots[:-extras] if root not in curr_val]
        print(f"{len(curr_train)=}")
        for root in curr_train:
            shutil.copytree(root, curr_train_folder / root.name)

    # Deal with the extra roots which didn't divide evenly by randomly assigning them to a fold
    chosen = []
    for extra in range(extras):
        i = extra + 1
        while True:
            add_to = random.choice(list(range(num_folds)))
            if add_to in chosen:
                continue
            else:
                break
        chosen += [add_to]
        root = roots[len(roots) - i]
        print(f"extra root: {root}")
        print(f"{add_to=}")
        for j in range(num_folds):
            if j == add_to:
                add_folder = dataset_folder / str(j) / "3D" / "val"
            else:
                add_folder = dataset_folder / str(j) / "3D" / "train"
            shutil.copytree(root, add_folder / root.name)


def create_2d_dataset():
    """
    Create the 2D images and labels slices so that they can be loaded quickly in a shuffled order during training.
    """
    reader = NIIReader()
    base_folder = Path("/media/y4tsu/4B172BDA26AB3054/cmr_folds")
    folds = [x for x in base_folder.iterdir() if x.is_dir()]

    for fold in folds:
        data_3d = fold / "3D"
        data_2d = fold / "2D"
        data_2d.mkdir(parents=False, exist_ok=True)

        train_data_2d = data_2d / "train"
        val_data_2d = data_2d / "val"
        train_data_2d.mkdir(parents=False, exist_ok=True)
        val_data_2d.mkdir(parents=False, exist_ok=True)

        transverse_train_2d = train_data_2d / "transverse"
        transverse_val_2d = val_data_2d / "transverse"
        transverse_train_2d.mkdir(parents=False, exist_ok=True)
        transverse_val_2d.mkdir(parents=False, exist_ok=True)

        for g in ["train", "val"]:
            for x in tqdm((data_3d / g).iterdir()):
                image_fname = x / f"{x.stem}_SAX.nii.gz"
                label_fname = x / f"{x.stem}_SAX_mask2.nii.gz"

                image = reader.read(image_fname)
                label = reader.read(label_fname)

                curr_dir = data_2d / g / "transverse" / x.stem
                curr_dir.mkdir(parents=False, exist_ok=True)

                for j in range(image.shape[2]):
                    np.save(
                        str(
                            data_2d
                            / g
                            / "transverse"
                            / x.stem
                            / f"{x.stem}_{j:03}_image"
                        ),
                        image[:, :, j],
                    )
                    np.save(
                        str(
                            data_2d
                            / g
                            / "transverse"
                            / x.stem
                            / f"{x.stem}_{j:03}_label"
                        ),
                        label[:, :, j],
                    )


def create_3dshallow_dataset(depth=5):
    """
    Create 3D shallow images and save with the given depth to allow fast shuffled loading during training.
    :param depth: int: slice thickness to create for images
    """
    assert (
        depth > 1 and depth % 2 == 1
    ), f"depth must be an odd number greater than 1 but got {depth}"
    reader = NIIReader()

    base_folder = Path("/media/y4tsu/4B172BDA26AB3054/cmr_folds")
    folds = [x for x in base_folder.iterdir() if x.is_dir()]

    for fold in folds:
        data_3d = fold / "3D"
        data_3d_shallow = fold / "3DShallow"
        data_3d_shallow.mkdir(parents=False, exist_ok=True)

        train_data_3d_shallow = data_3d_shallow / "train"
        val_data_3d_shallow = data_3d_shallow / "val"
        train_data_3d_shallow.mkdir(parents=False, exist_ok=True)
        val_data_3d_shallow.mkdir(parents=False, exist_ok=True)

        transverse_train_2d = train_data_3d_shallow / "transverse"
        transverse_val_2d = val_data_3d_shallow / "transverse"
        transverse_train_2d.mkdir(parents=False, exist_ok=True)
        transverse_val_2d.mkdir(parents=False, exist_ok=True)

        for g in ["train", "val"]:
            for x in tqdm((data_3d / g).iterdir()):
                image_fname = x / f"{x.stem}_SAX.nii.gz"
                label_fname = x / f"{x.stem}_SAX_mask2.nii.gz"

                image = reader.read(image_fname)
                label = reader.read(label_fname)

                curr_dir = data_3d_shallow / g / "transverse" / x.stem
                curr_dir.mkdir(parents=False, exist_ok=True)

                for j in range(image.shape[2] - depth + 1):
                    np.save(
                        str(
                            data_3d_shallow
                            / g
                            / "transverse"
                            / x.stem
                            / f"{x.stem}_{j:03}_image"
                        ),
                        image[:, :, j : j + depth],
                    )
                    np.save(
                        str(
                            data_3d_shallow
                            / g
                            / "transverse"
                            / x.stem
                            / f"{x.stem}_{j:03}_label"
                        ),
                        label[:, :, j : j + depth],
                    )


def find_depth_distributions(root):
    """Get the depths of all images in the dataset and display as a distribution."""
    depths = []
    reader = NIIReader()

    for g in ["train", "val"]:
        for x in tqdm(sorted(os.listdir(os.path.join(root, "3D", g)))):
            image_fname = os.path.join(root, "3D", g, x, f"{x}_SAX.nii.gz")
            image = reader.read(image_fname)
            depths += [image.shape[-1]]

    plt.hist(depths, bins=np.arange(9, 17) - 0.5, rwidth=0.75, color="cadetblue")
    plt.xlabel("Depth of image (number of slices)")
    plt.ylabel("Frequency")
    plt.title("Frequency of image depths")

    # Create legend & Show graphic
    plt.savefig(
        "/home/y4tsu/Desktop/diss img/depth_frequencies.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        "/home/y4tsu/Desktop/diss img/depth_frequencies.pdf", bbox_inches="tight"
    )
    plt.show()


def find_widths_distributions(root):
    """Get the widths of all images (square) in the dataset and display as a distribution."""
    depths = []
    reader = NIIReader()

    for g in ["train", "val"]:
        for x in tqdm(sorted(os.listdir(os.path.join(root, "3D", g)))):
            image_fname = os.path.join(root, "3D", g, x, f"{x}_SAX.nii.gz")
            image = reader.read(image_fname)
            depths += [image.shape[0]]

    plt.hist(
        depths,
        bins=np.arange(min(depths), max(depths) + 2) - 0.5,
        rwidth=1,
        color="cadetblue",
    )
    plt.xlabel("Width of image")
    plt.ylabel("Frequency")
    plt.title("Frequency of image widths")

    # Create legend & Show graphic
    plt.savefig(
        "/home/y4tsu/Desktop/diss img/width_frequencies.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        "/home/y4tsu/Desktop/diss img/width_frequencies.pdf", bbox_inches="tight"
    )
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

    plt.hist(widths, bins=16, rwidth=0.8, color="cadetblue")
    plt.xlabel("Width of image after manual crop")
    plt.ylabel("Frequency")
    plt.title("Frequency of image widths after applying manual cropping")

    # Create legend & Show graphic
    plt.savefig(
        "/home/y4tsu/Desktop/diss img/width_cropped_frequencies.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.savefig(
        "/home/y4tsu/Desktop/diss img/width_cropped_frequencies.pdf",
        bbox_inches="tight",
    )
    plt.show()


def plot_extra_metrics():
    plt.rcParams.update({"font.size": 7})

    df = pd.read_csv("volume_hd_metrics.csv")
    print(df.head(5))
    plt.scatter(df["ground truth scar volume"], df["predicted scar volume"], s=10)
    plt.xlabel("Ground truth scar volume (mm3)")
    plt.ylabel("Predicted scar volume (mm3)")
    pearson = stats.pearsonr(
        df["ground truth scar volume"], df["predicted scar volume"]
    )
    plt.title(
        f"Predicted vs. ground truth scar volume\n"
        f"Pearson coefficient = {pearson[0]:.4f}"
    )
    plt.savefig(
        "/home/y4tsu/Desktop/diss img/predicted vs ground truth scar volume correlation.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()

    min_bin = min(min(df["ground truth scar volume"]), min(df["predicted scar volume"]))
    max_bin = max(max(df["ground truth scar volume"]), max(df["predicted scar volume"]))
    plt.hist(
        [df["ground truth scar volume"], df["predicted scar volume"]],
        bins=16,
        rwidth=0.8,
        color=["cadetblue", "orange"],
        range=(min_bin, max_bin),
    )
    plt.xlabel("Scar volume (mm3)")
    plt.ylabel("Frequency")
    plt.title(
        "Frequency of scar volumes for ground truth and predicted scars\n"
        f"Median ground truth scar volume = {np.median(df['ground truth scar volume']):.0f}\n"
        f"Median predicted scar volume = {np.median(df['predicted scar volume']):.0f}"
    )
    plt.legend(["Ground truth scar", "Predicted scar"], title="Legend")
    plt.savefig(
        "/home/y4tsu/Desktop/diss img/predicted vs ground truth scar volume distributions.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()

    plt.scatter(df["ground truth scar volume"], df["hausdorff distance"], s=10)
    plt.xlabel("Ground truth scar volume (mm3)")
    plt.ylabel("Hausdorff distance (mm)")
    pearson = stats.pearsonr(df["ground truth scar volume"], df["hausdorff distance"])
    plt.title(
        f"Hausdorff distance vs. ground truth scar volume\n"
        f"Pearson coefficient = {pearson[0]:.4f}"
    )
    plt.savefig(
        "/home/y4tsu/Desktop/diss img/hausdorff distance vs ground truth scar volume.png",
        bbox_inches="tight",
        dpi=300,
    )
    plt.show()


def find_wilcoxon(path_a, path_b):
    """Get statistical significance of differences between tissues for 2 models."""
    a_scar_dice = pd.read_csv(path_a)
    b_scar_dice = pd.read_csv(path_b)
    # TODO: remove skip nums from both dataframes

    skip_files = [
        "20CA015_N055",
        "20CA015_N110",
        "20CA015_N207",
        "20CA015_N051",
        "20CA015_N109",
    ]
    a_scar_dice = a_scar_dice[~a_scar_dice["file"].isin(skip_files)]
    b_scar_dice = b_scar_dice[~b_scar_dice["file"].isin(skip_files)]

    a_scar_dice = a_scar_dice["scar"]
    b_scar_dice = b_scar_dice["scar"]

    d = b_scar_dice - a_scar_dice
    w, p = stats.wilcoxon(d)
    print(f"Result (NULL): w={w}, p={p}")


if __name__ == "__main__":
    # create_2d_dataset('/media/y4tsu/ml-fast/cmr_fold_5')
    # create_3dshallow_dataset('/media/y4tsu/ml-fast/cmr_fold_5', depth=5)
    # find_widths_distributions('/media/y4tsu/ml-fast/cmr')
    # search_certainties('/media/y4tsu/ml-fast/cmr')
    # analyse_quality_labels_class_wise()
    # analyse_quality_labels_frequency()
    # plot_extra_metrics()
    # rotate_incorrect_orientations()
    create_3dshallow_dataset(5)
    pass
