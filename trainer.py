import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
import os
import time
from tqdm import tqdm

from tensorflow.keras.utils import plot_model

from augmenter import Augmenter3D, Augmenter2D
from callbacks import LearningRatePrinter
from generators import Generator3D, Generator3DShallow, Generator2D, CascadedGenerator3D
from losses import (
    SoftmaxLoss,
    WeightedSoftmaxLoss,
    DiceLoss,
    WeightedSoftmaxDiceLoss,
    CascadedWeightedSoftmaxDiceLoss,
    WeightedSoftmaxDiceLossPlusQuality
)
from metrics import DiceMetric, ClassWiseDiceMetric
from models import UNet3D, UNet2D, UNet3DShallow, CascadedUNet3D
from util import PColour


class Trainer:
    """ Trainer class for training models. """

    def __init__(
        self,
        model_save_path,
        data_path,
        model,
        plane,
        num_epochs,
        batch_size,
        image_size,
        use_cropper,
        learning_rate,
        lr_decay,
        warmup,
        labels,
        combine_labels,
        cascade,
        loss_fn,
        quality_weighting,
        augmentation,
    ):
        """
        Initializer for Trainer.
        """
        # File path parameters
        self.model_save_path = model_save_path
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        if model in ['UNet3D']:
            self.dimensionality = '3D'
            self.data_root = '3D'
            print(f'plane ignored since using 3D model: {model}')
            plane = ""
            self.augmenter = Augmenter3D(**augmentation)
        elif model in ['UNet3DShallow']:
            self.dimensionality = '3DShallow'
            self.data_root = '3DShallow'
            self.augmenter = Augmenter3D(**augmentation)
            assert plane in ["transverse", "sagittal",
                             "coronal"], "Plane must be one of: 'transverse', 'sagittal', 'coronal'"
        elif model in ['CascadedUNet3D']:
            self.dimensionality = '3DCascaded'
            self.data_root = '3D'
            self.augmenter = Augmenter3D(**augmentation)
            plane = ""
        else:
            self.dimensionality = '2D'
            self.data_root = '2D'
            self.augmenter = Augmenter2D(**augmentation)
            assert plane in ["transverse", "sagittal",
                             "coronal"], "Plane must be one of: 'transverse', 'sagittal', 'coronal'"

        self.data_path = data_path
        self.train_data_path = os.path.join(data_path, self.data_root, 'train', plane)
        self.val_data_path = os.path.join(data_path, self.data_root, 'val', plane)
        self.test_data_path = os.path.join(data_path, self.data_root, 'test', plane)

        self.model_dict = {
            "UNet3D": UNet3D,
            "UNet3DShallow": UNet3DShallow,
            "UNet2D": UNet2D,
            "CascadedUNet3D": CascadedUNet3D,
        }
        self.gen_dict = {
            "3D": Generator3D,
            "3DShallow": Generator3DShallow,
            "2D": Generator2D,
            "3DCascaded": CascadedGenerator3D,
        }

        assert not (warmup and lr_decay), "currently warmup and lr_decay cannot be used simultaneously"

        # Training parameters
        self.model = model
        self.quality_weighted_mode = True if 'quality' in loss_fn else False
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.warmup = warmup
        self.combine_labels = combine_labels
        if self.quality_weighted_mode:
            monitor = 'val_m_dice'
        else:
            monitor = 'val_dice'
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.model_save_path,
                monitor=monitor if model not in ['CascadedUNet3D'] else 'val_general_out_dice',
                save_best_only=True,
                verbose=1,
                mode='max',
            )
        ]

        loss_dict = {
            'softmax': SoftmaxLoss,
            'weighted softmax': WeightedSoftmaxLoss,
            'dice': DiceLoss,
            'weighted softmax dice': WeightedSoftmaxDiceLoss,
            'quality weighted softmax dice': WeightedSoftmaxDiceLossPlusQuality,
            'cascaded weighted softmax dice': CascadedWeightedSoftmaxDiceLoss,
        }
        assert loss_fn in loss_dict, f'loss function {loss_fn}, not recognised, please pick one of: {loss_dict}'

        # Set up generators
        self.train_gen = self.gen_dict[self.dimensionality](
            model_save_path,
            data_path,
            self.train_data_path,
            plane,
            batch_size,
            image_size,
            labels,
            dataset='train',
            augmenter=self.augmenter,
            use_cropper=use_cropper,
            combine_labels=combine_labels,
            cascade=cascade,
            quality_weighting_scores=quality_weighting if self.quality_weighted_mode else False,
        )
        self.val_gen = self.gen_dict[self.dimensionality](
            model_save_path,
            data_path,
            self.val_data_path,
            plane,
            batch_size,
            image_size,
            labels,
            dataset='val',
            shuffle=False,
            use_cropper=use_cropper,
            combine_labels=combine_labels,
            cascade=cascade,
            quality_weighting_scores=quality_weighting if self.quality_weighted_mode else False,
        )

        # Labels
        self.labels = labels
        self.quality_weighting = quality_weighting if quality_weighting else False
        if loss_fn in ['weighted softmax', 'weighted softmax dice']:
            self.loss_fn = loss_dict[loss_fn](batch_size, image_size, self.calculate_label_weights())
        elif loss_fn in ['quality weighted softmax dice']:
            self.loss_fn = loss_dict[loss_fn](
                batch_size,
                image_size,
                self.calculate_label_weights(),
                dimensionality=self.dimensionality,
            )
        elif loss_fn in ['cascaded weighted softmax dice']:
            self.loss_fn = loss_dict[loss_fn](batch_size, image_size, self.calculate_label_weights_cascaded())
        else:
            self.loss_fn = loss_dict[loss_fn](batch_size, image_size)

        # Learning rate warmup: will be used if true
        if self.warmup:
            self.callbacks += [
                tf.keras.callbacks.LearningRateScheduler(self.warmup_lr)
            ]

    def calculate_label_weights(self):
        """ Calculate beta pixel weighting and its inverse as the label weights for weighted loss functions. """
        print(f'Calculating label weightings across {len(self.train_gen.image_fnames)} label images for use in loss'
              f' function, may take a while ... ')
        if self.combine_labels:
            sums = np.zeros(len(self.combine_labels))
        else:
            sums = np.zeros(len(self.labels))
        for i in tqdm(range(len(self.train_gen.image_fnames) // self.batch_size)):
            _, label_img = self.train_gen.__getitem__(i, weight_mode=True)
            # Get the number of labelled voxels of each class for each label image
            if self.quality_weighted_mode:
                sums += [label_img['m'][..., j].sum() for j in range(label_img['m'].shape[-1])]
            else:
                sums += [label_img[..., j].sum() for j in range(label_img.shape[-1])]
        # Get the total number of voxels in the dataset to normalize the beta
        total_voxels = np.prod(np.array([*self.image_size, len(self.train_gen.image_fnames)]))
        beta = sums / total_voxels
        print(1. / beta)
        # Return weightings: 1 / beta
        return 1. / beta

    def calculate_label_weights_cascaded(self):
        """ Calculate beta pixel weighting and its inverse as the label weights for weighted loss functions. """
        weights = {
            "general": np.zeros(len(self.labels) - 2),
            "scar": 0.,
            "pap": 0.,
        }
        print(f'Calculating label weightings across {len(self.train_gen.image_fnames)} label images for use in loss'
              f' function, may take a while ... ')
        for i in tqdm(range(len(self.train_gen.image_fnames) // self.batch_size)):
            _, y = self.train_gen.__getitem__(i, weight_mode=True)
            # Get the number of labelled voxels of each class for each label image
            weights["general"] += [y["general_out"][..., j].sum() for j in range(y["general_out"].shape[-1])]
            weights["scar"] += y["scar_out"].sum()
            weights["pap"] += y["pap_out"].sum()

        # Get the total number of voxels in the dataset to normalize the beta
        total_voxels = np.prod(np.array([*self.image_size, len(self.train_gen.image_fnames)]))
        weights["general"] = 1. / (weights["general"] / total_voxels)
        weights["scar"] = 1. / (weights["scar"] / total_voxels)
        weights["pap"] = 1. / (weights["pap"] / total_voxels)

        return weights

    def warmup_lr(self, *args):
        """ Warm up learning rate scheduler. Just naively reduce the LR for epoch 1 to get steadier momentum. """
        epoch = args[0]
        if epoch != 0:
            return self.lr
        else:
            print('Training with 10x reduced LR for epoch 1 for warmup ... ')
            return self.lr / 10.

    def plot(self, history):
        # Plot the losses and dice coefficients for the model
        prefix = 'm_' if self.quality_weighted_mode else ''
        e2e_cascading = True if self.model in ['CascadedUNet3D'] else False
        colors = ['coral', 'dodgerblue', 'teal', 'darkorchid', 'crimson', 'limegreen', 'hotpink', 'mediumblue',
                  'darkorange', 'navy', 'darkslategrey', 'purple', 'maroon', 'green', 'deeppink', 'navy']

        fig, axs = plt.subplots(nrows=2, ncols=2)
        fig.set_size_inches(8, 12)

        t_loss_hx = history.history[f'{prefix}loss'] if not e2e_cascading else history.history[f'{prefix}general_out_loss']
        v_loss_hx = history.history[f'val_{prefix}loss'] if not e2e_cascading else history.history[f'val_{prefix}general_out_loss']
        axs[0, 0].plot(range(1, 1 + len(t_loss_hx)), t_loss_hx, 'r-', label='train loss')
        axs[0, 0].plot(range(1, 1 + len(v_loss_hx)), v_loss_hx, 'b-', label='val loss')
        axs[0, 0].set(xlabel='epochs', ylabel='loss')
        axs[0, 0].legend(loc="upper right")

        t_dice_hx = history.history[f'{prefix}dice'] if not e2e_cascading else history.history[f'{prefix}general_out_dice']
        v_dice_hx = history.history[f'val_{prefix}dice'] if not e2e_cascading else history.history[f'val_{prefix}general_out_dice']
        axs[0, 1].plot(range(1, 1 + len(t_dice_hx)), t_dice_hx, 'r-', label='train dice')
        axs[0, 1].plot(range(1, 1 + len(v_dice_hx)), v_dice_hx, 'b-', label='val dice')
        axs[0, 1].set(xlabel='epochs', ylabel='dice coefficient')
        axs[0, 1].legend(loc="lower right")

        t_class_dx_labels = [x for x in list(history.history.keys()) if 'val' not in x and 'loss' not in x and 'dice' not in x]
        v_class_dx_labels = [x for x in list(history.history.keys()) if 'val' in x and 'loss' not in x and 'dice' not in x]

        if not e2e_cascading:
            for x in t_class_dx_labels:
                curr_hx = history.history[x]
                color = colors.pop(0)
                axs[1, 0].plot(range(1, 1 + len(curr_hx)), curr_hx, color=color, label=f'{x} dice')

            for x in v_class_dx_labels:
                curr_hx = history.history[x]
                color = colors.pop(0)
                axs[1, 1].plot(range(1, 1 + len(curr_hx)), curr_hx, color=color, label=f'{x} dice')

            axs[1, 0].set(xlabel='epochs', ylabel='dice coefficient')
            axs[1, 0].legend(loc="lower right")

            axs[1, 1].set(xlabel='epochs', ylabel='dice coefficient')
            axs[1, 1].legend(loc="lower right")

        plt.show()

    def train(self):
        """ Train the model. """
        model = self.model_dict[self.model](
            input_size=self.image_size,
            output_length=len(self.combine_labels) if self.combine_labels else len(self.labels),
            quality_weighted_mode=self.quality_weighted_mode,
        ).create_model()

        plot_model(model, f'plot/{self.model}_plot.png', show_shapes=True)
        model.summary(line_length=160)

        # TODO: assertions on all config stuff to prevent naughty values being given
        # TODO: anatomical auto-encoder if time allows
        # TODO: de-noising auto-encoder if time allows
        # TODO: add black space instead of removing non-square images (could cause bbox problems)
        # TODO: could try replacing UNet3DShallow with e.g., VNet if time allows
        # TODO: end to end cascading networks for configs B and C

        # Learning rate decay: will be used if not 0, otherwise use static LR
        if self.lr_decay:
            schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                self.lr,
                self.num_epochs * len(self.train_gen.image_fnames) // self.batch_size,
                self.lr_decay
            )
            self.callbacks += [LearningRatePrinter()]
        else:
            schedule = self.lr

        optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)

        if self.model not in ['CascadedUNet3D']:
            if self.combine_labels:
                metrics = [DiceMetric(self.batch_size)] + \
                          [ClassWiseDiceMetric(self.batch_size, i, str(i)) for i in
                           range(len(self.combine_labels))]
            else:
                metrics = [DiceMetric(self.batch_size)] + \
                          [ClassWiseDiceMetric(self.batch_size, i, self.labels[label]) for i, label in
                           zip(range(len(self.labels)), self.labels)]
            if self.quality_weighted_mode:
                metrics = {'m': metrics}
        else:
            metrics = {
                'general_out':
                    [DiceMetric(self.batch_size)] +
                    [ClassWiseDiceMetric(self.batch_size, i, self.labels[label]) for i, label in
                     zip(range(6), ['0', '1', '2', '4', '5', '7'])],
                'scar_out': [DiceMetric(self.batch_size)],
                'pap_out': [DiceMetric(self.batch_size)]
            }

        model.compile(
            optimizer=optimizer,
            loss=self.loss_fn,
            metrics=metrics,
        )

        start = time.time()

        history = model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.train_gen.image_fnames) // self.batch_size,
            validation_steps=len(self.val_gen.image_fnames) // self.batch_size,
            callbacks=self.callbacks,
        )

        print(f'{PColour.OKGREEN}Finished training - took {(time.time() - start) / 60.:.2f} minutes.{PColour.ENDC}')

        # Save the config that was used so that e.g., image size can be retrieved later for prediction
        with open(f'{self.model_save_path}/train_config.json', 'w') as f:
            f.write(json.dumps(config, indent=4))
        print(f'Saved training config to {self.model_save_path}/train_config.json')

        # Plot the losses
        self.plot(history)


if __name__ == '__main__':
    with open('train_config.json', 'r') as r:
        config = json.load(r)

    Trainer(**config).train()
