import json
import os
import time

import numpy as np
from scipy.special import softmax
from tqdm import tqdm

from predict import Predictor2D
from readers import NIIReader
from util import PColour


def setup_predictors(data_path, dataset, path_tr, path_sag, path_cor):
    """Get 3 predictors for the transverse, sagittal and coronal planes."""
    # Transverse predictor
    with open(os.path.join(path_tr, 'train_config.json'), 'r') as f:
        p1 = Predictor2D(data_path, dataset, path_tr, json.load(f))

    # Sagittal predictor
    with open(os.path.join(path_sag, 'train_config.json'), 'r') as f:
        p2 = Predictor2D(data_path, dataset, path_sag, json.load(f))

    # Coronal predictor
    with open(os.path.join(path_cor, 'train_config.json'), 'r') as f:
        p3 = Predictor2D(data_path, dataset, path_cor, json.load(f))

    return p1, p2, p3


if __name__ == '__main__':
    nii_reader = NIIReader()
    data_path = "/media/y4tsu/ml_data/cmr"
    dataset = "val"

    p_tr, p_sag, p_cor = setup_predictors(
        data_path,
        dataset,
        "/home/y4tsu/PycharmProjects/3d_unet/checkpoint/2D_tr",
        "/home/y4tsu/PycharmProjects/3d_unet/checkpoint/2D_sag",
        "/home/y4tsu/PycharmProjects/3d_unet/checkpoint/2D_cor",
    )

    roots = sorted(os.listdir(os.path.join(data_path, '3D', dataset)))
    dices = 0
    class_wise_dices = np.zeros([len(p_tr.labels_dict)])
    start = time.time()

    for root in tqdm(roots):
        tr_name = os.path.join(data_path, '2D', dataset, 'transverse', root)
        sag_name = os.path.join(data_path, '2D', dataset, 'sagittal', root)
        cor_name = os.path.join(data_path, '2D', dataset, 'coronal', root)

        im_tr, lab_tr, pred_tr = p_tr.predict_logits(os.path.join(data_path, '2D', dataset, 'transverse', tr_name))
        im_sag, lab_sag, pred_sag = p_sag.predict_logits(os.path.join(data_path, '2D', dataset, 'sagittal', sag_name))
        im_cor, lab_cor, pred_cor = p_cor.predict_logits(os.path.join(data_path, '2D', dataset, 'coronal', cor_name))

        pred_sag = np.moveaxis(pred_sag, [0, 1, 2], [0, 2, 1])
        pred_cor = np.moveaxis(pred_cor, [0, 1, 2], [1, 2, 0])
        pred_sag = nii_reader.resize(pred_sag, pred_tr.shape, interpolation_order=0)
        pred_cor = nii_reader.resize(pred_cor, pred_tr.shape, interpolation_order=0)

        # Take the average of the logits for each volume and then apply softmax to get the final predictions
        avg_logits = (pred_tr + pred_sag + pred_cor) / 3.
        avg_softmax = softmax(avg_logits, axis=-1)
        pred_label = np.argmax(avg_logits, axis=-1)

        label = nii_reader.read(os.path.join(data_path, '3D', dataset, root, f'{root}_SAX_mask2.nii.gz'))
        # Make sure the label and predicted label are the same size for obtaining dice scores
        if label.shape != pred_label.shape:
            label = nii_reader.resize(label, pred_label.shape)

        dices += p_tr.calculate_dice(label, pred_label)
        class_wise_dices += p_tr.calculate_class_wise_dice(label, pred_label)

    print(f'AVG DICE: {dices / len(roots)}')

    for i, key in enumerate(p_tr.labels_dict):
        print(f'{p_tr.labels_dict[key]:<16}:  {class_wise_dices[i] / len(roots):.4f}')

    print(f'{PColour.OKGREEN}Time taken for inference on {len(roots)} samples: {time.time() - start:.2f} seconds.{PColour.ENDC}')
