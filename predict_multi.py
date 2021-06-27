import json
import os

import matplotlib.pyplot as plt
import numpy as np

from predict import Predictor2D, Predictor3D, np_dice_coefficient
from readers import NIIReader, NPYReader


if __name__ == '__main__':
    nii_reader = NIIReader()
    data_path = "/media/y4tsu/ml_data/cmr"
    dataset = "val"

    # Transverse predictor
    model_path = "/home/y4tsu/PycharmProjects/3d_unet/checkpoint/2D"
    with open(os.path.join(model_path, 'train_config.json'), 'r') as train_config_file:
        train_config = json.load(train_config_file)
    p1 = Predictor2D(data_path, dataset, model_path, train_config)

    # Sagittal predictor
    model_path = "/home/y4tsu/PycharmProjects/3d_unet/checkpoint/2Dsag"
    with open(os.path.join(model_path, 'train_config.json'), 'r') as train_config_file:
        train_config = json.load(train_config_file)
    p2 = Predictor2D(data_path, dataset, model_path, train_config)

    # Coronal predictor
    model_path = "/home/y4tsu/PycharmProjects/3d_unet/checkpoint/2Dcor"
    with open(os.path.join(model_path, 'train_config.json'), 'r') as train_config_file:
        train_config = json.load(train_config_file)
    p3 = Predictor2D(data_path, dataset, model_path, train_config)

    # Get predictions for all 3 planes
    fname = '20CA015_N258'
    im_tr, lab_tr, p_tr = p1.predict(os.path.join(data_path, '2D', dataset, 'transverse', fname), display=False)
    im_sag, lab_sag, p_sag = p2.predict(os.path.join(data_path, '2D', dataset, 'sagittal', fname), display=False)
    im_cor, lab_cor, p_cor = p3.predict(os.path.join(data_path, '2D', dataset, 'coronal', fname), display=False)

    # Find where the sagittal and coronal labels agree but disagree with the transverse
    # print(p_tr.shape)
    # print(p_sag.shape)
    # print(p_cor.shape)
    # print('=====')

    p_sag = np.moveaxis(p_sag, [0, 1, 2], [0, 2, 1])
    p_cor = np.moveaxis(p_cor, [0, 1, 2], [1, 2, 0])

    # print(p_tr.shape)
    # print(p_sag.shape)
    # print(p_cor.shape)
    # print('=====')

    p_sag = nii_reader.resize(p_sag, p_tr.shape)
    p_cor = nii_reader.resize(p_cor, p_tr.shape)

    # print(p_tr.shape)
    # print(p_sag.shape)
    # print(p_cor.shape)

    # fig, axs = plt.subplots(3, 1)
    # axs[0].imshow(p_tr[..., 0])
    # axs[1].imshow(p_sag[..., 0])
    # axs[2].imshow(p_cor[..., 0])
    # plt.show()

    total = np.where(np.equal(p_sag, p_cor), p_sag, p_tr)
    # print(total.shape)
    # nii_reader.scroll_view(total)

    print(np_dice_coefficient(lab_tr, total))
