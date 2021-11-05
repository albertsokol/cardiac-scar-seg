"""Find the Hausdorff distance and volumes of scar tissue, compared between prediction and ground truth."""

import csv
import os

import SimpleITK as sitk
import numpy as np
from readers import NIIReader


def volume(mask_image):
    # Input:
    # image = sitk.Image, mask or binary image (1 values where organ, 0 values otherwise)
    # Output:
    # vol = float, volume in mm3
    space = mask_image.GetSpacing()         # image spacing
    voxel = np.prod(space)                  # voxel volume
    img = sitk.GetArrayFromImage(mask_image)
    vol = voxel*np.sum(img)

    return vol


def get_results(num):
    print(f"{num=}")
    possible_folders = ["cmr", "cmr_fold_2", "cmr_fold_3", "cmr_fold_4", "cmr_fold_5"]

    label = None
    for folder in possible_folders:
        try:
            label = sitk.ReadImage(f"/media/y4tsu/ml-fast/{folder}/3D/val/20CA015_N{num}/20CA015_N{num}_SAX_mask2.nii.gz")
            break
        except RuntimeError:
            continue
    if label is None:
        raise RuntimeError(f"Unable to load label for {num}")

    pred = sitk.ReadImage(f"saved_preds/20CA015_N{num}_prediction.nii.gz")

    label = label == 3
    pred = pred == 3

    label_volume = volume(label)
    pred_volume = volume(pred)
    print(f"scar {label_volume=:.2f}mm3")
    print(f"scar {pred_volume=:.2f}mm3")

    hausdorff_distance_filter = sitk.HausdorffDistanceImageFilter()
    hausdorff_distance_filter.__setattr__("Tolerance", 1e-9)
    hausdorff_distance_filter.Execute(label, pred)
    hd = hausdorff_distance_filter.GetHausdorffDistance()

    print(f"{hd=:.2f}mm")
    print("---")

    return num, label_volume, pred_volume, hd


def main():
    output = []
    nums = [x.split("_")[1][1:] for x in os.listdir("saved_preds")]

    skip_nums = ["055", "110", "207", "051", "109"]
    for skip_num in skip_nums:
        nums.remove(skip_num)

    for num in nums:
        # try:
        result = get_results(num)
        # except RuntimeError:
        #     continue
        output += [list(result)]

    output = sorted(output, key=lambda x: x[0])
    output = [["file", "ground truth scar volume", "predicted scar volume", "hausdorff distance"]] + output

    with open("volume_hd_metrics.csv", "w") as f:
        writer = csv.writer(f)
        writer.writerows(output)


if __name__ == "__main__":
    main()
