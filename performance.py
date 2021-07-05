"""Used for finding the class-wise and average Dice scores of a model on a dataset (e.g., validation or test)."""

import os
import json

import numpy as np
from tqdm import tqdm

from predict import Predictor2D, Predictor3D, Predictor3DCascaded, Predictor3DShallow


if __name__ == '__main__':
    # Load the configs
    with open('predict_config.json', 'r') as predict_config_file:
        predict_config = json.load(predict_config_file)

    with open(os.path.join(predict_config['model_path'], 'train_config.json'), 'r') as train_config_file:
        train_config = json.load(train_config_file)

    plane = train_config['plane']

    # Load the correct Predictor class for the given model type
    if train_config['model'] in ['UNet3D']:
        p = Predictor3D(**predict_config, train_config=train_config)
        dims = '3D'
        plane = ''
    elif train_config['model'] in ['UNet3DShallow']:
        p = Predictor3DShallow(**predict_config, train_config=train_config)
        dims = '3DShallow'
    elif train_config['model'] in ['CascadedUNet3D']:
        p = Predictor3DCascaded(**predict_config, train_config=train_config)
        dims = '3D'
        plane = ''
    else:
        p = Predictor2D(**predict_config, train_config=train_config)
        dims = '2D'

    # Get the names of all the scans we are interested in
    roots = sorted(os.listdir(os.path.join(predict_config['data_path'], dims, predict_config['dataset'], plane)))
    roots = [os.path.join(predict_config['data_path'], dims, predict_config['dataset'], plane, root) for root in roots]
    print(roots)

    dices = 0
    class_wise_dices = np.zeros([len(p.labels_dict)])

    # Run the model in predictions mode and get the dice scores for each scan
    for root in tqdm(roots):
        image, label, pred_label = p.predict(fname=root, display=False)
        dices += p.calculate_dice(label, pred_label)
        class_wise_dices += p.calculate_class_wise_dice(label, pred_label)

    # Print a report with the averages of those dice scores
    print(f'AVG DICE: {dices / len(roots)}')

    for i, key in enumerate(p.labels_dict):
        print(f'{p.labels_dict[key]:<10}: {class_wise_dices[i] / len(roots):.4f}')
