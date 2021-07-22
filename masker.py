import json
import os
import time

import numpy as np
from tqdm import tqdm

from readers import NIIReader
from util import PColour


class Masker:
    """Class for masking inputs based on the prediction of a previously trained model."""
    def __init__(self, model_path, keep_labels, data_path, dataset, folder, plane=""):
        """
        Initializer for Masker.

        Params
        ------
        :param model_path: str: path to the model to use for masking
        :param keep_labels: list: list of label indices which should be included (i.e., not masked)
        :param data_path: str: path to the images to be masked
        :param dataset: str: name of the dataset, e.g., 'train' or 'val'
        :param folder: str: name of the root folder where the masks folder will be saved
        :param plane: str: plane of image to use, e.g., 'transverse', leave as empty string for 3D models
        """
        # Masker properties
        self.model_path = model_path
        self.keep_labels = keep_labels

        # Data properties
        self.data_path = data_path
        self.dataset = dataset
        self.plane = plane

        # Create the folder that will store the masks
        if not os.path.exists(os.path.join(folder, 'mask')):
            os.mkdir(os.path.join(folder, 'mask'))
        if not os.path.exists(os.path.join(folder, 'mask', dataset)):
            os.mkdir(os.path.join(folder, 'mask', dataset))
        self.mask_folder = os.path.join(folder, 'mask', dataset)

    def create_fname_folder(self, fname):
        """Create folder to store masks for the current fname."""
        # Create folder and wait for it to be created before saving arrays
        if not os.path.exists(os.path.join(self.mask_folder, fname)):
            os.mkdir(os.path.join(self.mask_folder, fname))
        while not os.path.exists(os.path.join(self.mask_folder, fname)):
            time.sleep(0.001)

    def save_2d_mask(self, f, fname):
        """Save masked image in 2D."""
        self.create_fname_folder(fname)
        assert self.plane, f"plane must be given for 2D masking but got '{self.plane}'"

        # Save input array images for the correct plane
        if self.plane == 'coronal':
            for j in range(f.shape[0]):
                np.save(os.path.join(self.mask_folder, fname, f'{fname}_{j:03}_image'), f[j, :, :])
        elif self.plane == 'sagittal':
            for j in range(f.shape[1]):
                np.save(os.path.join(self.mask_folder, fname, f'{fname}_{j:03}_image'), f[:, j, :])
        elif self.plane == 'transverse':
            for j in range(f.shape[2]):
                np.save(os.path.join(self.mask_folder, fname, f'{fname}_{j:03}_image'), f[:, :, j])

    def save_3d_shallow_mask(self, f, fname, depth):
        """Save masked image in 3D shallow with slice thickness of depth."""
        self.create_fname_folder(fname)
        for j in range(f.shape[2] - depth + 1):
            np.save(os.path.join(self.mask_folder, fname, f'{fname}_{j:03}_image'), f[:, :, j:j + depth])

    def save_3d_mask(self, f, fname):
        """Save masked image in 3D."""
        self.create_fname_folder(fname)
        np.save(os.path.join(self.mask_folder, fname, f'{fname}_SAX'), f)

    def create_masks(self):
        """Create masks for the selected dataset of the Masker, e.g., training or validation."""
        if len(os.listdir(self.mask_folder)) != 0:
            print(f"{PColour.OKCYAN}Using existing masks at {self.mask_folder} therefore new masking was skipped, if"
                  f" this is incorrect then delete the files at {self.mask_folder} to re-generate. {PColour.ENDC}")
            return

        from predict import load_predictor

        print(f'Predicting masks using model at {self.model_path} for {self.dataset} data ... ')
        roots = sorted(os.listdir(os.path.join(self.data_path, '3D', self.dataset)))

        # Set up the correct config for the predictor model
        predict_config = {
            'model_path': self.model_path,
            'data_path': self.data_path,
            'dataset': self.dataset,
        }

        with open(os.path.join(predict_config['model_path'], 'train_config.json'), 'r') as train_config_file:
            train_config = json.load(train_config_file)
            print(f'Using training config for nested model: {train_config}')

        # Load the masking model as a Predictor object
        p = load_predictor(predict_config, train_config)

        # Iterate over all the images, getting predicted labels
        for i, root in enumerate(tqdm(roots)):
            # Search for the correct full filename
            fname = None
            for x in p.image_fnames:
                if root in x:
                    fname = x

            if fname is None:
                raise ValueError(f"Unable to find the correct path for image {root}")

            # Get the 3D predicted label for the image
            image, label, pred_label = p.predict(fname, display=False)

            # Mask the image using the predicted label
            masked_image = np.where(np.isin(pred_label, self.keep_labels), image, 0)

            if p.dimensionality == '2D':
                self.save_2d_mask(masked_image, root)
            elif p.dimensionality == '3DShallow':
                self.save_3d_shallow_mask(masked_image, root, depth=train_config['image_size'][-1])
            else:
                self.save_3d_mask(masked_image, root)
