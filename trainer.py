import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tensorflow.keras import callbacks
from tensorflow.keras import optimizers
import json
from tqdm import tqdm

from callbacks import LearningRateFinder
from losses import weighted_pixel_bce_loss, dice_loss, combined_dice_wpce_loss, SoftmaxLoss, WeightedSoftmaxLoss
from metrics import dice_coefficient_wrapper
from models import create_segmentation_model, unet_pp_pretrain_model, UNet3D
from tensorflow.keras.utils import plot_model

from augmenter import Augmenter
from generators import NIIGenerator


class Trainer:
    """ Trainer class for training models. """
    def __init__(self, model_save_path, image_path, label_path, mode, model, num_epochs, batch_size, image_size,
                 train_val_test_splits, learning_rate, labels, loss_fn):
        """
        Initializer for Trainer.
        """
        # File path parameters
        self.model_save_path = model_save_path
        self.image_path = image_path
        self.label_path = label_path

        # Mode: "lrf" for learning rate finder, "train" for normal model training
        assert mode in ['lrf', 'train'], f'mode \'{mode}\' does not exist, please use \'lrf\' or \'train\''
        self.mode = mode

        # Training parameters
        self.model = model
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.image_size = image_size
        self.train_val_test_splits = train_val_test_splits
        self.lr = learning_rate

        loss_dict = {
            'softmax': SoftmaxLoss,
            'weighted softmax': WeightedSoftmaxLoss
        }
        assert loss_fn in loss_dict, f'loss function {loss_fn}, not recognised, please pick one of: {loss_dict}'

        # Set up generator and augmentation etc.
        self.augmenter = Augmenter()
        self.generator = NIIGenerator(image_path, label_path, batch_size, image_size, labels, augmenter=self.augmenter)

        # Labels
        self.labels = labels
        if loss_fn == 'weighted softmax':
            self.loss_fn = loss_dict[loss_fn](batch_size, image_size, self.calculate_label_weights())
        else:
            self.loss_fn = loss_dict[loss_fn](batch_size, image_size)

    def calculate_label_weights(self):
        """ Calculate beta pixel weighting and it's inverse as the label weights for weighted loss functions. """
        sums = np.zeros(len(self.labels))
        print(f'Calculating label weightings across {len(self.generator.image_fnames)} label images for use in loss'
              f' function, may take a while ... ')
        for i in tqdm(range(len(self.generator.image_fnames))):
            _, label_img = self.generator.__getitem__(i)
            # Get the number of labelled voxels of each class for each label image
            sums += [label_img[..., j].sum() for j in range(0, len(self.labels))]
            print(sums)
        # Get the total number of voxels in the dataset to normalize the beta
        total_voxels = np.prod(np.array([*self.image_size, len(self.generator.image_fnames)]))
        beta = sums / total_voxels
        print(total_voxels)
        print(beta)
        # Return weightings: 1 / beta
        return 1. / beta

    def train(self):
        """ Train the model. """
        model = UNet3D(input_size=self.image_size,
                       output_length=len(self.labels)).create_model()
        # plot_model(model, 'UNet3Dplot.png', show_shapes=True)
        model.summary(line_length=160)


if __name__ == '__main__':
    with open('config.json', 'r') as r:
        config = json.load(r)

    trainer = Trainer(**config)
    trainer.train()

    # Loading data and training constants
    overall_df = pd.read_csv(csv_path, index_col='ImageId')
    overall_df_size = len(overall_df)
    train_num = int(overall_df_size * TRAIN_PROP)
    print('Number of training samples:', train_num, 'Number of validation samples:', overall_df_size - train_num)
    TRAIN_STEPS = train_num // BATCH_SIZE
    VAL_STEPS = (overall_df_size - train_num) // BATCH_SIZE

    # Set up dataframes and generators
    train_df = overall_df[:train_num]
    val_df = overall_df[train_num:]
    train_generator = SegGenerator(train_df, image_path, BATCH_SIZE, resize_to=RESIZE_TO)
    val_generator = SegGenerator(val_df, image_path, BATCH_SIZE, resize_to=RESIZE_TO, aug=False)

    # Set up loss functions and metrics - by default, combined_loss is used
    weighted_bce_loss = weighted_pixel_bce_loss(beta_pixel_weighting, BATCH_SIZE)
    dice_loss = dice_loss()
    combined_loss = combined_dice_wpce_loss(beta_pixel_weighting, BATCH_SIZE)
    dice_coefficient = dice_coefficient_wrapper()

    # Set up callbacks - if using 'train' mode, model with best val dice coefficient will be saved to save_path
    if mode == 'lrf':
        lrf = LearningRateFinder(NUM_EPOCHS * TRAIN_STEPS)
        cbs = [lrf]
    if mode == 'train':
        checkpoint = callbacks.ModelCheckpoint(save_path, monitor='val_dice_coefficient', save_best_only=True,
                                               verbose=1, mode='max')
        cbs = [checkpoint]

    # Load classification UNet++ trained on ImageNet (224x224 input)
    imgnet_model = unet_pp_pretrain_model(224)
    imgnet_model.load_weights(imgnet_pretrain_path)

    # Create a new segmentation UNet++/UNet (512x512 input)
    model = create_segmentation_model(RESIZE_TO)

    # Because the input sizes and output heads are different, transfer weights like this
    print('Loading following layer weights from ImageNet pretrained model to new segmentation model...')
    for i in range(1, len(imgnet_model.layers) - 2):
        wts = imgnet_model.layers[i].get_weights()
        model.layers[i].set_weights(wts)
        print('imgnet:', imgnet_model.layers[i].name, '--------> seg:', model.layers[i].name)

    """
    Change and uncomment this if you want to freeze layers during training. 
    
    for l in model.layers:
        if l.name == 'conv2d_14' or l.name == 'conv2d_15':
            l.trainable = False
    """

    # Set up Adam optimizer and compile the model - can change the loss for experimenting
    opt = optimizers.Adam(learning_rate=lr)
    model.compile(optimizer=opt, loss=combined_loss, metrics=[dice_coefficient])

    # Begin training
    history = model.fit(train_generator,
                        validation_data=val_generator,
                        epochs=NUM_EPOCHS,
                        steps_per_epoch=TRAIN_STEPS,
                        validation_steps=VAL_STEPS,
                        callbacks=cbs)

    # Plot callback graphs if used
    if mode == 'lrf':
        lrf.plot()

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
    axs[1].plot(range(1, 1 + len(val_dice_coefficient_history)), val_dice_coefficient_history, 'm-', label='val dice')
    axs[1].set(xlabel='epochs', ylabel='dice coefficient')
    axs[1].legend(loc="upper right")

    plt.show()
