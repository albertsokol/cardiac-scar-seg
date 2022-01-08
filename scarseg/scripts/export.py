"""Run predictions, and then export the predicted segmentations to a .nii.gz file."""

import json
import os
import time

import nibabel as nib
import numpy as np
from tqdm import tqdm

from scarseg.predict import load_predictor
from scarseg.readers import NIIReader


def main():
    start = time.time()

    with open("predict_config.json", "r") as f:
        predict_config = json.load(f)

    # Load the correct Predictor class for the given model type
    p = load_predictor(predict_config)
    plane = p.plane
    cropper = p.cropper

    if p.model_name in ["UNet3D", "VNet", "UNet3DFrozenDepth"]:
        dims = "3D"
        plane = ""
    elif p.model_name in ["UNet3DShallow", "VNetShallow"]:
        dims = "3DShallow"
    else:
        dims = "2D"

    # Get the names of all the scans we are interested in
    roots = sorted(
        os.listdir(
            os.path.join(
                predict_config["data_path"], dims, predict_config["dataset"], plane
            )
        )
    )
    roots = [
        os.path.join(
            predict_config["data_path"], dims, predict_config["dataset"], plane, root
        )
        for root in roots
    ]
    print(f"{roots=}")

    # Get the correct metadata
    headers = {}
    shapes = {}
    affines = {}
    reader = NIIReader()

    # Get roots from the 3D folder
    real_roots = sorted(
        os.listdir(
            os.path.join(predict_config["data_path"], "3D", predict_config["dataset"])
        )
    )
    real_roots = [
        os.path.join(predict_config["data_path"], "3D", predict_config["dataset"], x)
        for x in real_roots
    ]

    # Load important information from the original .nii.gz files for these images
    for real_root in tqdm(real_roots):
        name_end = real_root.split("/")[-1]
        img = nib.load(
            os.path.join(real_root, f"{name_end}_SAX_mask2.nii.gz"), mmap=False
        )
        # Headers = headers from the original NiFTi file
        headers[name_end] = img.header
        # Affine = transformation which maps points to 3D space of the MRI image
        affines[name_end] = img.affine
        # Shape of each image
        shapes[name_end] = np.squeeze(img.get_fdata()).shape

    # Get a prediction for each image and save it in NiFTi format for loading in ITK-SNAP
    for root in tqdm(roots):
        # Get the prediction
        image, label, pred_label = p.predict(fname=root, display=False)
        # Get the key to find the correct header and affine for the output
        name_end = root.split("/")[-1]
        # print(f'{name_end=}')
        save_img = np.zeros(shapes[name_end])
        # print(f'{save_img.shape=}')
        # bbox = cropper.bboxes[name_end]
        # print(f"{bbox=}")
        reverse_size, cut_dims = p.cropper.reverse_crop(shapes[name_end], name_end)
        # print(f"{reverse_size=}")
        # print(f"{cut_dims=}")
        # Resize the prediction to undo the pre-processing
        pred_label = reader.resize(pred_label, reverse_size, interpolation_order=0)
        # Get the correct portion of the predicted label to output
        try:
            save_img[
                cut_dims["top"] : cut_dims["bottom"],
                cut_dims["left"] : cut_dims["right"],
                ...,
            ] = pred_label
        except ValueError:
            print(
                f"Unable to undo pre-processing for image {name_end}! {save_img.shape=}, {pred_label.shape=}"
            )
            continue
        # Save the image in NiFTi format
        # exit()
        save_img = save_img.astype(np.uint16)

        save_img = nib.Nifti1Image(
            save_img,
            affine=affines[name_end],
            header=headers[name_end],
        )

        nib.save(
            save_img,
            os.path.join(
                "/home/y4tsu/PycharmProjects/3d_unet/saved_preds",
                f"{name_end}_prediction.nii.gz",
            ),
        )
        # print('-----')

    print(f"Finished! Process took {time.time() - start:.2f} seconds.")


if __name__ == "__main__":
    main()
