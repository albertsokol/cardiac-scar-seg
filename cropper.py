import json
import os

import cv2
import numpy as np
from tqdm import tqdm

from readers import NIIReader


class Cropper:
    """Class for cropping images and labels."""
    def __init__(self, data_path, dataset, mode, pad=8):
        """Initializer for Cropper."""
        self.data_path = data_path
        self.dataset = dataset
        self.mode = "manual" if mode == "manual" else "auto"
        if mode == "manual":
            self.bboxes = self._get_manual_bboxes()
        else:
            self.bboxes = self._get_auto_bboxes(model_path=mode)
        self.pad = pad

    @staticmethod
    def __find_bbox(f):
        """Get a bounding box around the given segmentation mask."""
        # WARNING!: not tested on non-square images, some of the axes in the top/left/bottom/right detection may be wrong
        # Order is: top, left, bottom, right
        bbox = [f.shape[1] // 2, f.shape[0] // 2, f.shape[1] // 2, f.shape[0] // 2]

        # Get top, left, bottom and right extents of the label on each slice
        for j in range(f.shape[-1]):
            curr = f[..., j]
            try:
                top = np.where(np.any(curr == 1, axis=1))[0][0]
            except IndexError:
                # If the slice is completely background, then skip it
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

        return bbox

    @staticmethod
    def __clean_prediction(f, kernel_size=5):
        """Clean up bits of noise and smaller blobs from the predicted label to get a better bounding box."""
        kernel = np.ones([kernel_size, kernel_size], dtype=np.int8)
        out = np.empty(f.shape, dtype=np.int8)

        for i in range(f.shape[-1]):
            curr = f[..., i].astype(np.uint8)

            # Erode and then dilate the white areas using the kernel to get rid of noise smaller than kernel_size
            curr = cv2.erode(curr, kernel, iterations=1)
            curr = cv2.dilate(curr, kernel, iterations=1)

            out[..., i] = curr

        return out

    def _get_manual_bboxes(self):
        """Load the manual segmentation masks from 3D images to get bounding boxes on the fly."""
        print(f'Loading bboxes from manual segmentations for {self.dataset} data ... ')
        reader = NIIReader()
        base_folder = os.path.join(self.data_path, '3D', self.dataset)
        roots = sorted(os.listdir(base_folder))
        out = {}

        for i, root in enumerate(tqdm(roots)):
            label = np.squeeze(reader.read(os.path.join(base_folder, root, f'{root}_SAX_mask2.nii.gz')))
            binary_label = np.where(np.equal(label, 0), 0, 1)
            bbox = self.__find_bbox(binary_label)

            out[root] = {
                'top': bbox[0],
                'left': bbox[1],
                'bottom': bbox[2],
                'right': bbox[3],
            }

        return out

    def _get_auto_bboxes(self, model_path):
        """Run the segmentation model to get bounding boxes for all the images in the dataset."""
        from predict import load_predictor

        print(f'Predicting bboxes using automatic cropper model at {model_path} for {self.dataset} data ... ')
        reader = NIIReader()
        base_folder = os.path.join(self.data_path, '3D', self.dataset)
        roots = sorted(os.listdir(base_folder))
        out = {}

        # Set up the correct config for the predictor model
        predict_config = {
            'model_path': model_path,
            'data_path': self.data_path,
            'dataset': self.dataset,
            'post_process': False
        }

        # Load the cropper model as a Predictor object
        p = load_predictor(predict_config)

        # Iterate over all the images, getting predicted labels
        for i, root in enumerate(tqdm(roots)):
            # Search for the correct full filename
            fname = None
            for x in p.image_fnames:
                if root in x:
                    fname = x

            if fname is None:
                raise ValueError(f"Unable to find the correct path for image {root}")

            image_size = reader.read(
                os.path.join(self.data_path, '3D', self.dataset, f'{root}/{root}_SAX.nii.gz')
            ).shape

            _, label, pred_label = p.predict(fname, display=False)

            # Remove noise and small islands from the prediction
            pred_label = self.__clean_prediction(pred_label)

            # In the case where an image has different dimensions to model input, needs to be re-scaled
            new_pred_label = np.empty(image_size, dtype=np.int8)
            if image_size[0] != p.image_size[0]:
                for j in range(image_size[-1]):
                    curr = pred_label[..., j]
                    new_pred_label[..., j] = cv2.resize(
                        curr,
                        tuple(reversed(image_size[:2])),
                        interpolation=cv2.INTER_NEAREST
                    )
                pred_label = new_pred_label

            # Now find the bounding box around the segmentation mask
            bbox = self.__find_bbox(pred_label)

            out[root] = {
                'top': bbox[0],
                'left': bbox[1],
                'bottom': bbox[2],
                'right': bbox[3],
            }

        return out

    def crop(self, f, fname):
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
