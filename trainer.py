import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import json
from tqdm import tqdm

from tensorflow.keras.utils import plot_model

from augmenter import Augmenter
from callbacks import LearningRateFinder, LearningRatePrinter
from generators import NIIGenerator3D
from losses import SoftmaxLoss, WeightedSoftmaxLoss, DiceLoss, WeightedSoftmaxDiceLoss
from metrics import DiceMetric, ClassWiseDiceMetric
from models import UNet3D


class Trainer:
    """ Trainer class for training models. """
    def __init__(self, model_save_path, train_image_path, train_label_path, val_image_path, val_label_path,
                 test_image_path, test_label_path, mode, model, slice_20, num_epochs, batch_size, image_size,
                 learning_rate, lr_decay, warmup, labels, loss_fn, augmentation):
        """
        Initializer for Trainer.
        """
        # File path parameters
        self.model_save_path = model_save_path
        self.train_image_path = train_image_path
        self.train_label_path = train_label_path
        self.val_image_path = val_image_path
        self.val_label_path = val_label_path
        self.test_image_path = test_image_path
        self.test_label_path = test_label_path

        # Mode: "lrf" for learning rate finder, "train" for normal model training
        assert mode in ['lrf', 'train'], f'mode \'{mode}\' does not exist, please use \'lrf\' or \'train\''
        self.mode = mode

        assert not (warmup and lr_decay), "currently warmup and lr_decay cannot be used simultaneously"

        # Training parameters
        self.model = model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.warmup = warmup
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.model_save_path,
                monitor='val_dice',
                save_best_only=True,
                verbose=1,
                mode='max'
            )
        ]

        loss_dict = {
            'softmax': SoftmaxLoss,
            'weighted softmax': WeightedSoftmaxLoss,
            'dice': DiceLoss,
            'weighted softmax dice': WeightedSoftmaxDiceLoss
        }
        assert loss_fn in loss_dict, f'loss function {loss_fn}, not recognised, please pick one of: {loss_dict}'

        # Set up generator and augmentation etc.
        self.augmenter = Augmenter(**augmentation)
        self.train_gen = NIIGenerator3D(
            train_image_path,
            train_label_path,
            batch_size,
            image_size,
            labels,
            slice_20,
            augmenter=self.augmenter
        )
        self.val_gen = NIIGenerator3D(
            val_image_path,
            val_label_path,
            batch_size,
            image_size,
            labels,
            slice_20,
            shuffle=False
        )

        # Labels
        self.labels = labels
        if loss_fn in ['weighted softmax', 'weighted softmax dice']:
            self.loss_fn = loss_dict[loss_fn](batch_size, image_size, self.calculate_label_weights())
        else:
            self.loss_fn = loss_dict[loss_fn](batch_size, image_size)

        # Learning rate warmup: will be used if true
        if self.warmup:
            self.callbacks += [
                tf.keras.callbacks.LearningRateScheduler(self.warmup_lr)
            ]

    def calculate_label_weights(self):
        """ Calculate beta pixel weighting and its inverse as the label weights for weighted loss functions. """
        sums = np.zeros(len(self.labels))
        print(f'Calculating label weightings across {len(self.train_gen.image_fnames)} label images for use in loss'
              f' function, may take a while ... ')
        for i in tqdm(range(len(self.train_gen.image_fnames))):
            _, label_img = self.train_gen.__getitem__(i, weight_mode=True)
            # Get the number of labelled voxels of each class for each label image
            sums += [label_img[..., j].sum() for j in range(0, len(self.labels))]
        # Get the total number of voxels in the dataset to normalize the beta
        total_voxels = np.prod(np.array([*self.image_size, len(self.train_gen.image_fnames)]))
        beta = sums / total_voxels
        print(total_voxels)
        print(beta)
        # Return weightings: 1 / beta
        return 1. / beta

    def lrf(self, model):
        """ Run the learning rate finder. """
        self.num_epochs = 3
        lrf = LearningRateFinder(self.num_epochs * len(self.train_gen.image_fnames) // self.batch_size)
        optimizer = tf.keras.optimizers.Adam()
        model.compile(optimizer=optimizer, loss=self.loss_fn)
        model.fit(self.train_gen,
                  validation_data=self.val_gen,
                  epochs=self.num_epochs,
                  steps_per_epoch=len(self.train_gen.image_fnames) // self.batch_size,
                  validation_steps=len(self.val_gen.image_fnames) // self.batch_size,
                  callbacks=[lrf]
                  )
        lrf.plot()

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
        loss_history = history.history['loss']
        val_loss_history = history.history['val_loss']
        dice_coefficient_history = history.history['dice_coefficient']
        val_dice_coefficient_history = history.history['val_dice_coefficient']

        fig, axs = plt.subplots(2)
        fig.set_size_inches(8, 12)

        axs[0].plot(range(1, 1 + len(loss_history)), loss_history, 'r-', label='train loss')
        axs[0].plot(range(1, 1 + len(val_loss_history)), val_loss_history, 'b-', label='val loss')
        axs[0].set(xlabel='epochs', ylabel='loss')
        axs[0].legend(loc="upper right")

        axs[1].plot(range(1, 1 + len(dice_coefficient_history)), dice_coefficient_history, 'g-', label='train dice')
        axs[1].plot(range(1, 1 + len(val_dice_coefficient_history)), val_dice_coefficient_history, 'm-',
                    label='val dice')
        axs[1].set(xlabel='epochs', ylabel='dice coefficient')
        axs[1].legend(loc="upper right")

        plt.show()

    def train(self):
        """ Train the model. """
        model = UNet3D(input_size=self.image_size, output_length=len(self.labels)).create_model()
        # plot_model(model, 'UNet3Dplot.png', show_shapes=True)
        model.summary(line_length=160)

        # Exit to learning rate finder if that mode has been selected
        if self.mode == 'lrf':
            self.lrf(model)
            return

        # TODO: plotting loss and metric history at the end
        # TODO: implement other models
        # TODO: assertions on all config stuff to prevent naughty values being given
        # TODO: 2D generator

        # Learning rate decay: will be used if not 0. Recommend 0.96 if using. Otherwise use static LR
        if self.lr_decay:
            schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                self.lr,
                self.num_epochs * len(self.train_gen.image_fnames) // self.batch_size,
                self.lr_decay
            )
            # self.callbacks += [LearningRatePrinter()]
        else:
            schedule = self.lr

        optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
        model.compile(
            optimizer=optimizer,
            loss=self.loss_fn,
            metrics=[DiceMetric(self.batch_size)] +
                    [ClassWiseDiceMetric(self.batch_size, int(i), self.labels[i]) for i in self.labels]
        )

        history = model.fit(self.train_gen,
                            validation_data=self.val_gen,
                            epochs=self.num_epochs,
                            steps_per_epoch=len(self.train_gen.image_fnames) // self.batch_size,
                            validation_steps=len(self.val_gen.image_fnames) // self.batch_size,
                            callbacks=self.callbacks
                            )

        # Save the config that was used so that e.g., image size can be retrieved later for prediction
        with open(f'{self.model_save_path}/train_config.json', 'w') as f:
            f.write(json.dumps(config))

        # Plot the losses
        self.plot(history)


if __name__ == '__main__':
    with open('config.json', 'r') as r:
        config = json.load(r)

    Trainer(**config).train()
