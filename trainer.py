import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
import os
from tqdm import tqdm

from tensorflow.keras.utils import plot_model

from augmenter import Augmenter3D, Augmenter2D
from callbacks import LearningRatePrinter
from generators import Generator3D, Generator2D, CascadedGenerator3D
from losses import SoftmaxLoss, WeightedSoftmaxLoss, DiceLoss, WeightedSoftmaxDiceLoss, CascadedWeightedSoftmaxDiceLoss
from metrics import DiceMetric, ClassWiseDiceMetric
from models import UNet3D, UNet2D, UNet3DShallow, CascadedUNet3D


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
        loss_fn,
        augmentation,
    ):
        """
        Initializer for Trainer.
        """
        # File path parameters
        self.model_save_path = model_save_path
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
            "3DShallow": Generator2D,
            "2D": Generator2D,
            "3DCascaded": CascadedGenerator3D,
        }

        assert not (warmup and lr_decay), "currently warmup and lr_decay cannot be used simultaneously"

        # Training parameters
        self.model = model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.warmup = warmup
        self.combine_labels = combine_labels
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.model_save_path,
                monitor='val_scar' if model not in ['CascadedUNet3D'] else 'val_general_out_dice',
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
            'cascaded weighted softmax dice': CascadedWeightedSoftmaxDiceLoss,
        }
        assert loss_fn in loss_dict, f'loss function {loss_fn}, not recognised, please pick one of: {loss_dict}'

        # Set up generators
        self.train_gen = self.gen_dict[self.dimensionality](
            data_path,
            self.train_data_path,
            batch_size,
            image_size,
            labels,
            dataset='train',
            augmenter=self.augmenter,
            use_cropper=use_cropper,
            combine_labels=combine_labels,
        )
        self.val_gen = self.gen_dict[self.dimensionality](
            data_path,
            self.val_data_path,
            batch_size,
            image_size,
            labels,
            dataset='val',
            shuffle=False,
            use_cropper=use_cropper,
            combine_labels=combine_labels,
        )

        # Labels
        self.labels = labels
        if loss_fn in ['weighted softmax', 'weighted softmax dice']:
            self.loss_fn = loss_dict[loss_fn](batch_size, image_size, self.calculate_label_weights())
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
        if self.model not in ['CascadedUNet3D']:
            loss_history = history.history['loss']
            val_loss_history = history.history['val_loss']
            dice_coefficient_history = history.history['dice']
            val_dice_coefficient_history = history.history['val_dice']
        else:
            loss_history = history.history['general_out_loss']
            val_loss_history = history.history['val_general_out_loss']
            dice_coefficient_history = history.history['general_out_dice']
            val_dice_coefficient_history = history.history['val_general_out_dice']
            scar_loss_history = history.history['scar_out_loss']
            val_scar_loss_history = history.history['val_scar_out_loss']
            scar_dice_history = history.history['scar_out_dice']
            val_scar_dice_history = history.history['val_scar_out_dice']
            pap_loss_history = history.history['pap_out_loss']
            val_pap_loss_history = history.history['val_pap_out_loss']
            pap_dice_history = history.history['pap_out_dice']
            val_pap_dice_history = history.history['val_pap_out_dice']

        fig, axs = plt.subplots(nrows=2, ncols=2)
        fig.set_size_inches(8, 12)

        if self.model not in ['CascadedUNet3D']:
            axs[0, 0].plot(range(1, 1 + len(loss_history)), loss_history, 'r-', label='train loss')
            axs[0, 0].plot(range(1, 1 + len(val_loss_history)), val_loss_history, 'b-', label='val loss')
        else:
            axs[0, 0].plot(range(1, 1 + len(loss_history)), loss_history, 'r-', label='gen train loss')
            axs[0, 0].plot(range(1, 1 + len(val_loss_history)), val_loss_history, 'b-', label='gen val loss')
            axs[0, 0].plot(range(1, 1 + len(scar_loss_history)), scar_loss_history, 'c-', label='scar train loss')
            axs[0, 0].plot(range(1, 1 + len(val_scar_loss_history)), val_scar_loss_history, 'g-', label='scar val loss')
            axs[0, 0].plot(range(1, 1 + len(pap_loss_history)), pap_loss_history, 'm-', label='pap train loss')
            axs[0, 0].plot(range(1, 1 + len(val_pap_loss_history)), val_pap_loss_history, 'y-', label='pap val loss')

        axs[0, 0].set(xlabel='epochs', ylabel='loss')
        axs[0, 0].legend(loc="upper right")

        if self.model not in ['CascadedUNet3D']:
            axs[0, 1].plot(range(1, 1 + len(dice_coefficient_history)), dice_coefficient_history, 'r-', label='train dice')
            axs[0, 1].plot(range(1, 1 + len(val_dice_coefficient_history)), val_dice_coefficient_history, 'b-', label='val dice')
        else:
            axs[0, 1].plot(range(1, 1 + len(dice_coefficient_history)), dice_coefficient_history, 'r-', label='gen train dice')
            axs[0, 1].plot(range(1, 1 + len(val_dice_coefficient_history)), val_dice_coefficient_history, 'b-', label='gen val dice')
            axs[0, 1].plot(range(1, 1 + len(scar_dice_history)), scar_dice_history, 'c-', label='scar train dice')
            axs[0, 1].plot(range(1, 1 + len(val_scar_dice_history)), val_scar_dice_history, 'g-', label='scar val dice')
            axs[0, 1].plot(range(1, 1 + len(pap_dice_history)), pap_dice_history, 'm-', label='pap train dice')
            axs[0, 1].plot(range(1, 1 + len(val_pap_dice_history)), val_pap_dice_history, 'y-', label='pap val dice')

        axs[0, 1].set(xlabel='epochs', ylabel='dice coefficient')
        axs[0, 1].legend(loc="lower right")

        if len(self.labels) > 8:
            colors = ['red'] * (len(self.labels) * 2)
        else:
            colors = [
                'darkorange',
                'coral',
                'navy',
                'dodgerblue',
                'darkslategrey',
                'teal',
                'purple',
                'darkorchid',
                'maroon',
                'crimson',
                'green',
                'limegreen',
                'deeppink',
                'hotpink',
                'navy',
                'mediumblue'
            ]

        if self.model not in ['CascadedUNet3D']:
            i = 0
            for label in list(self.labels.values()):
                curr_train_hx = history.history[label]
                curr_val_hx = history.history[f'val_{label}']
                axs[1, 0].plot(
                    range(1, 1 + len(dice_coefficient_history)),
                    curr_train_hx,
                    color=colors[i],
                    label=f'train {label} dice'
                )
                i += 1
                axs[1, 1].plot(
                    range(1, 1 + len(dice_coefficient_history)),
                    curr_val_hx,
                    color=colors[i],
                    label=f'val {label} dice'
                )
                i += 1

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
        ).create_model()

        plot_model(model, f'{self.model}_plot.png', show_shapes=True)
        model.summary(line_length=160)

        # TODO: assertions on all config stuff to prevent naughty values being given
        # TODO: fix plot function
        # TODO: cascaded networks: utilise masking of the input based on n previous models (customisable)
        # TODO: anatomical auto-encoder if time allows
        # TODO: slice quality weighted loss functions
        # TODO: add black space instead of removing non-square images (could cause bbox problems)
        # TODO: could try replacing UNet3DShallow with e.g., VNet if time allows

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
                          [ClassWiseDiceMetric(self.batch_size, i, f'{i}_dice') for i in
                           range(len(self.combine_labels))]
            else:
                metrics = [DiceMetric(self.batch_size)] + \
                          [ClassWiseDiceMetric(self.batch_size, i, self.labels[label]) for i, label in
                           zip(range(len(self.labels)), self.labels)]
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

        history = model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.train_gen.image_fnames) // self.batch_size,
            validation_steps=len(self.val_gen.image_fnames) // self.batch_size,
            callbacks=self.callbacks,
        )

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
