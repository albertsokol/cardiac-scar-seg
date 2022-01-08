import json
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tqdm import tqdm

from scarseg.callbacks import LearningRatePrinter
from scarseg.generators import DAEGenerator
from scarseg.losses import WeightedSoftmaxLoss
from scarseg.metrics import DiceMetric, ClassWiseDiceMetric
from scarseg.models import DenoisingUNet
from scarseg.util import PColour


class DAETrainer:
    """Trainer class for training denoising auto-encoders."""

    def __init__(
        self,
        model_save_path,
        data_path,
        num_epochs,
        batch_size,
        image_size,
        use_cropper,
        learning_rate,
        lr_decay,
        warmup,
        labels,
    ):
        """
        Initializer for Trainer.
        """
        # File path parameters
        self.model_save_path = model_save_path
        if not os.path.exists(model_save_path):
            os.mkdir(model_save_path)

        plane = "transverse"
        self.dimensionality = "2D"
        self.data_root = "2D"

        self.data_path = data_path
        self.train_data_path = os.path.join(data_path, self.data_root, "train", plane)
        self.val_data_path = os.path.join(data_path, self.data_root, "val", plane)

        assert not (
            warmup and lr_decay
        ), "currently warmup and lr_decay cannot be used simultaneously"

        # Training parameters
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.lr = learning_rate
        self.lr_decay = lr_decay
        self.warmup = warmup
        self.callbacks = [
            tf.keras.callbacks.ModelCheckpoint(
                self.model_save_path,
                monitor="val_dice",
                save_best_only=True,
                verbose=1,
                mode="max",
            )
        ]

        # Set up generators
        self.train_gen = DAEGenerator(
            data_path,
            self.train_data_path,
            batch_size,
            image_size,
            labels,
            dataset="train",
            shuffle=True,
            use_cropper=use_cropper,
        )
        self.val_gen = DAEGenerator(
            data_path,
            self.val_data_path,
            batch_size,
            image_size,
            labels,
            dataset="val",
            shuffle=False,
            use_cropper=use_cropper,
            zoom_aug=False,
        )

        # Labels
        self.labels = labels
        self.loss_fn = WeightedSoftmaxLoss(
            batch_size, image_size, self.calculate_label_weights()
        )

        # Learning rate warmup: will be used if true
        if self.warmup:
            self.callbacks += [tf.keras.callbacks.LearningRateScheduler(self.warmup_lr)]

    def calculate_label_weights(self):
        """Calculate beta pixel weighting and its inverse as the label weights for weighted loss functions."""
        print(
            f"Calculating label weightings across {len(self.train_gen.in_fnames)} label images for use in loss"
            f" function, may take a while ... "
        )
        sums = np.zeros(len(self.labels))
        for i in tqdm(range(len(self.train_gen.in_fnames) // self.batch_size)):
            _, label_img = self.train_gen.__getitem__(i, weight_mode=True)
            # Get the number of labelled voxels of each class for each label image
            sums += [label_img[..., j].sum() for j in range(label_img.shape[-1])]
        # Get the total number of voxels in the dataset to normalize the beta
        total_voxels = np.prod(
            np.array([*self.image_size, len(self.train_gen.in_fnames)])
        )
        beta = sums / total_voxels
        print(1.0 / beta)
        # Return weightings: 1 / beta
        return 1.0 / beta

    def plot(self, history):
        # Plot the losses and dice coefficients for the model
        colors = [
            "coral",
            "dodgerblue",
            "teal",
            "darkorchid",
            "crimson",
            "limegreen",
            "hotpink",
            "mediumblue",
            "darkorange",
            "navy",
            "darkslategrey",
            "purple",
            "maroon",
            "green",
            "deeppink",
            "navy",
        ]

        fig, axs = plt.subplots(nrows=2, ncols=2)
        fig.set_size_inches(8, 12)

        t_loss_hx = history.history[f"loss"]
        v_loss_hx = history.history[f"val_loss"]
        axs[0, 0].plot(
            range(1, 1 + len(t_loss_hx)), t_loss_hx, "r-", label="train loss"
        )
        axs[0, 0].plot(range(1, 1 + len(v_loss_hx)), v_loss_hx, "b-", label="val loss")
        axs[0, 0].set(xlabel="epochs", ylabel="loss")
        axs[0, 0].legend(loc="upper right")

        t_dice_hx = history.history[f"dice"]
        v_dice_hx = history.history[f"val_dice"]
        axs[0, 1].plot(
            range(1, 1 + len(t_dice_hx)), t_dice_hx, "r-", label="train dice"
        )
        axs[0, 1].plot(range(1, 1 + len(v_dice_hx)), v_dice_hx, "b-", label="val dice")
        axs[0, 1].set(xlabel="epochs", ylabel="dice coefficient")
        axs[0, 1].legend(loc="lower right")

        t_class_dx_labels = [
            x
            for x in list(history.history.keys())
            if "val" not in x and "loss" not in x and "dice" not in x
        ]
        v_class_dx_labels = [
            x
            for x in list(history.history.keys())
            if "val" in x and "loss" not in x and "dice" not in x
        ]

        for x in t_class_dx_labels:
            curr_hx = history.history[x]
            color = colors.pop(0)
            axs[1, 0].plot(
                range(1, 1 + len(curr_hx)), curr_hx, color=color, label=f"{x} dice"
            )

        for x in v_class_dx_labels:
            curr_hx = history.history[x]
            color = colors.pop(0)
            axs[1, 1].plot(
                range(1, 1 + len(curr_hx)), curr_hx, color=color, label=f"{x} dice"
            )

        axs[1, 0].set(xlabel="epochs", ylabel="dice coefficient")
        axs[1, 0].legend(loc="lower right")

        axs[1, 1].set(xlabel="epochs", ylabel="dice coefficient")
        axs[1, 1].legend(loc="lower right")

        plt.show()

    def train(self):
        """Train the model."""
        s = tf.distribute.MirroredStrategy()
        print(tf.config.experimental.list_physical_devices("GPU"))
        with s.scope():
            model = DenoisingUNet(
                input_size=self.image_size,
                output_length=len(self.labels),
            ).create_model()

            plot_model(model, f"../plot/DenoisingUnet_plot.png", show_shapes=True)
            model.summary(line_length=160)

            # Learning rate decay: will be used if not 0, otherwise use static LR
            if self.lr_decay:
                schedule = tf.keras.optimizers.schedules.ExponentialDecay(
                    self.lr,
                    self.num_epochs * len(self.train_gen.in_fnames) // self.batch_size,
                    self.lr_decay,
                )
                self.callbacks += [LearningRatePrinter()]
            else:
                schedule = self.lr

            optimizer = tf.keras.optimizers.Adam(learning_rate=schedule)
            metrics = [DiceMetric(self.batch_size)] + [
                ClassWiseDiceMetric(self.batch_size, i, self.labels[label])
                for i, label in zip(range(len(self.labels)), self.labels)
            ]

            model.compile(
                optimizer=optimizer,
                loss=self.loss_fn,
                metrics=metrics,
            )

        print(f"{PColour.OKBLUE}Compiled model! Starting training ... {PColour.ENDC}")

        start = time.time()

        history = model.fit(
            self.train_gen,
            validation_data=self.val_gen,
            epochs=self.num_epochs,
            steps_per_epoch=len(self.train_gen.in_fnames) // self.batch_size,
            validation_steps=len(self.val_gen.in_fnames) // self.batch_size,
            callbacks=self.callbacks,
        )

        print(
            f"{PColour.OKGREEN}Finished training - took {(time.time() - start) / 60.:.2f} minutes.{PColour.ENDC}"
        )

        # Save the config that was used so that e.g., image size can be retrieved later for prediction
        with open(f"{self.model_save_path}/train_config.json", "w") as f:
            f.write(json.dumps(config, indent=4))
        print(f"Saved training config to {self.model_save_path}/train_config.json")

        # Plot the losses
        self.plot(history)


if __name__ == "__main__":
    with open("dae_train_config.json", "r") as r:
        config = json.load(r)

    DAETrainer(**config).train()
