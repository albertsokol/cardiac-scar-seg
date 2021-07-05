import json
import os

import matplotlib.pyplot as plt
import numpy as np
from scipy.special import softmax

from predict import Predictor2D, Predictor3D
from readers import NIIReader, NPYReader


if __name__ == '__main__':
    nii_reader = NIIReader()
    data_path = "/media/y4tsu/ml_data/cmr"
    dataset = "val"

    # Transverse predictor
    model_path = "/home/y4tsu/PycharmProjects/3d_unet/checkpoint/experiment_tr"
    with open(os.path.join(model_path, 'train_config.json'), 'r') as train_config_file:
        train_config = json.load(train_config_file)
    p1 = Predictor2D(data_path, dataset, model_path, train_config)

    # Sagittal predictor
    model_path = "/home/y4tsu/PycharmProjects/3d_unet/checkpoint/experiment_sag"
    with open(os.path.join(model_path, 'train_config.json'), 'r') as train_config_file:
        train_config = json.load(train_config_file)
    p2 = Predictor2D(data_path, dataset, model_path, train_config)

    # Coronal predictor
    model_path = "/home/y4tsu/PycharmProjects/3d_unet/checkpoint/experiment_cor"
    with open(os.path.join(model_path, 'train_config.json'), 'r') as train_config_file:
        train_config = json.load(train_config_file)
    p3 = Predictor2D(data_path, dataset, model_path, train_config)

    # Get predictions for all 3 planes
    fname = '20CA015_N243'
    im_tr, lab_tr, p_tr = p1.predict_logits(os.path.join(data_path, '2D', dataset, 'transverse', fname))
    im_sag, lab_sag, p_sag = p2.predict_logits(os.path.join(data_path, '2D', dataset, 'sagittal', fname))
    im_cor, lab_cor, p_cor = p3.predict_logits(os.path.join(data_path, '2D', dataset, 'coronal', fname))

    print(p_tr.shape)
    print(p_sag.shape)
    print(p_cor.shape)
    print('=====')

    p_sag = np.moveaxis(p_sag, [0, 1, 2], [0, 2, 1])
    p_cor = np.moveaxis(p_cor, [0, 1, 2], [1, 2, 0])

    print(p_tr.shape)
    print(p_sag.shape)
    print(p_cor.shape)
    print('=====')

    p_sag = nii_reader.resize(p_sag, p_tr.shape, interpolation_order=0)
    p_cor = nii_reader.resize(p_cor, p_tr.shape, interpolation_order=0)

    # print(p_tr.shape)
    # print(p_sag.shape)
    # print(p_cor.shape)

    # fig, axs = plt.subplots(3, 1)
    # axs[0].imshow(np.argmax(p_tr, axis=-1)[..., 0])
    # axs[1].imshow(np.argmax(p_sag, axis=-1)[..., 0])
    # axs[2].imshow(np.argmax(p_cor, axis=-1)[..., 0])
    # plt.show()

    # Take the average of the logits for each volume and then apply softmax to get the final predictions
    avg_logits = (p_tr + p_sag + p_cor) / 3.
    print(avg_logits.shape)
    avg_softmax = softmax(avg_logits, axis=-1)
    print(avg_softmax.shape)
    pred_labels = np.argmax(avg_logits, axis=-1)
    print(pred_labels.shape)

    nii_reader.scroll_view(pred_labels)
