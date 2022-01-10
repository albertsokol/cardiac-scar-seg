# Deep learning for ventricular scar segmentation in cardiac magnetic resonance imaging   

This repo contains the code for training and evaluating the models for the project titled above. 

## Contents
* [Installation](#installation)
* [Files](#files)
* [Structuring the input data](#structuring-the-input-data)
* [Usage](#usage)
* [Training settings](#training-settings-train_configjson)
* [Prediction settings](#prediction-settings-predict_configjson)

## Installation 

To install for usage only, in a Python 3.9 virtual environment:

```shell script
pip install .
```

To install for development:

```shell script
pip install -e . 
```

## Files
An overview of the included files is given here.

#### Configuration:
* `trainer_config.json` - update this with all the training settings, create this from `train_config.default.json` by copying and renaming
* `predict_config.json` - update this with the prediction settings
* `dae/dae_train_config.json` - update this with settings for training denoising auto-encoders

#### Public:
* `get_quality_scores.py` - generates a .json file from .csv (not required; for reference only)
* `performance.py` - evaluates overall class-wise performance on the validation set
* `predict.py` - contains functionality for running trained models on an input, running this file will display the result on a single image 
* `predict_multi.py` - used to aggregate predictions and measure performance for the 2D multi-planar approach
* `trainer.py` - used to train models 
* `util.py` - various utility functions for creating plots, generating 2D and 3D shallow data from 3D files
* `dae/dae_trainer.py` - used to train denoising auto-encoder models 

#### Private:
* `augmenter.py` - contains functionality for data augmentation in 2D and 3D
* `callbacks.py` - contains LearningRatePrinter callback to display LR during training
* `cropper.py` - contains functionality for cropper models
* `generators.py` - contains data generators to form training x and y tensors
* `losses.py` - contains all loss functions, may be added to by following the abstract base class
* `masker.py` - contains functionality for masking for cascaded model training
* `metrics.py` - contains functionality for calculating overall and class-wise Dice score on tensors
* `models.py` - contains all model code, may be added to by sub-classing the existing models 
* `readers.py` - contains functionality for data loading in `.nii.gz` and `.npy` format
* `dae/denoiser.py` - contains functionality for running a denoising auto-encoder on an image

## Structuring the input data  

In order to use this library with new data, data should be in the following format:
* Original data should be inside a folder named `3D`
* This folder should have 2 sub-folders: `train` and `val` for the training and validation splits 
* These folders should have sub-folders for every training sample, with a name in the format `"{project_name}_NXXX"`, where `{project_name}` can be any arbitrary string, `XXX` is an integer denoting the ID of the sample, and the `_N` portion is essential for splitting filenames correctly within the library
* Each of these folders should have have the MRI and label segmentation inside with the naming convention:
  * MRI: `"{project_name}_NXXX_SAX.nii.gz"`
  * Label: `"{project_name}_NXXX_SAX_mask2.nii.gz"`

The folder structure should therefore look like this:
```
3D
└─ train 
│   └─ {project_name}_NXXX
│       └─ {project_name}_NXXX_SAX.nii.gz
│       └─ {project_name}_NXXX_SAX_mask2.nii.gz
│   ...
│
└─ val
    └─ {project_name}_NXXX
        └─ {project_name}_NXXX_SAX.nii.gz
        └─ {project_name}_NXXX_SAX_mask2.nii.gz
    ...
```

After this has been created, the following functions from `util.py` can be used to generate the 2D and 3D shallow datasets:

* `create_2d_dataset()` - create a dataset of 2D slices in the short axis, horizontal and vertical axes
* `create_3dshallow_dataset(depth=n)` - create a dataset of 3D shallow slices from the original images of depth n along the short axis 

If using certainty scores, a `quality_scores.json` file should also be present, which is structured as:
```json
{
  "N000": [0, 1, 2, ..., 2],
  ...
  "N999": [2, 2, 1, ..., 2]
}
```

Where the `N000` to `N999` should cover all the `NXXX` numbers available in the full dataset, and the list contains a certainty score for each slice in image `NXXX`.

## Usage

After setting the data to the correct format, the process for training segmentation models is:
1. Set up the configuration you'd like in `train_config.json`
2. Run `trainer.py` to train a model on the data using that configuration
3. Update `predict_config.json` to contain the path to the newly trained model
4. Run `predict.py` to check the performance on a single, random image
5. Run `performance.py` to get the overall performance on the validation set

For denoising auto-encoders, the process is:
1. Set up the configuration you'd like in `dae_train_config.json`
2. Run `dae_trainer.py` to train a denoising auto-encoder using that configuration
3. Set `"post_process"` in `predict_config.json` as the path to the newly trained model to use it for post-processing outputs 

### Tensorboard

To view all Dice scores and losses during training, Tensorboard is integrated in this repo. After starting a training 
run, pass the following command in a new terminal to open Tensorboard:

```shell script
tensorboard --logdir ./logs
```

And navigate to `http://localhost:6006/` to view graphs.

## Training settings: `train_config.json`

All the training settings you might need to adjust are found in `train_config.json`. Update this file, then run `trainer.py` to train new models 

* `"model_save_path"`: path to save the trained models, must be `str`
* `"data_path"`: root to the MRI images/labels folder, must be `str`
* `"model"`: name of the model to train, must be `str`, possible options are
  * `"UNet2D"`
  * `"UNet3D"`
  * `"UNet3DShallow"`
  * `"CascadedUNet3D"`   
* `"plane"`: if using a 2D or shallow 3D UNet, use one of the following planes, must be `str`
  * `"transverse"`: short axis through the heart 
  * `"sagittal"`: horizontal axis 
  * `"coronal"`: vertical axis 
* `"num_epochs"`: number of epochs to run during training, must be `int`
* `"batch_size"`: batch size to use during training, must be `int`
* `"image_size"`: size of input images to the model e.g., `[336, 336]`, must be `list[int]`, 2 values for 2D networks, otherwise 3 values required 
* `"use_cropper"`: cropping method to use for training, possible options are:
  * `false`: do not use cropping, must be JSON `bool` and not the string `"false"`
  * `"manual"`: use the labelled segmentation maps to form cropping bounding boxes, must be `str`
  * `"/path/to/cropping/model`: path to a model if using an automatic cropper, must be `str`
* `"learning_rate"`: learning rate for model training, recommend `5e-2`, must be `int` or `float`
* `"lr_decay"`: rate of learning rate decay, must be `float`, `0.0` to switch off decay
* `"warmup"`: whether to reduce learning rate for the first epoch of training, must be `bool`, cannot be used together with `"lr_decay"` 
* `"loss_fn"`: the loss function to use during training, recommend `"weighted softmax dice"`, must be `str`, possible options are
  * `"softmax"`: basic categorical cross entropy loss on softmax output  
  * `"weighted softmax"`: softmax loss weighted by inverse class frequency for each label in `"labels"` or `"combine_labels"` if not none 
  * `"dice"`: basic dice loss on softmax output 
  * `"weighted softmax dice"`: mixed dice and weighted softmax loss functions 
  * `"quality weighted softmax dice"`: mixed dice and weighed softmax loss functions, also weighted by quality scores from `"quality_weighting"`
  * `"cascaded weighted softmax dice"`: weighted softmax dice loss for use with end-to-end cascaded network
* `"quality_weighting"`: dictionary of `str: float` pairs, where keys are quality scores, and values are the proportion by which the loss for that key is multiplied 
* `"labels"`: dictionary of `str: str` pairs, where keys are value given to a class in the label file, and values are class names, e.g., `"1": "lv myo"`. This should represent the labels as per the manual segmentations. Must be `dict`
* `"combine_labels"`: must be a `list` of `list[str]`, where any label class names occurring in an inner list will be combined to form a single label. Set to an empty list `[]` to switch off label combining. Label combining is used for training cascaded networks or automatic cropping models. For example, to create an automatic cropper, all heart tissue labels should be combined into one, so the correct value for this setting would be: 
  ```json
  [
    ["bg"],
    ["lv lumen", "lv myo", "scar", "pap", "rv lumen", "rv myo", "aorta"]
  ]
  ```
* `"cascade"`: using the output of prior models to mask the current model sequentially, must be `dict` with following format, or left empty to ignore this setting
  ```json
    "cascade": {
      "model_path": "/path/to/previous/model",
      "keep_labels": [2]  
    }
  ```
  where `"model_path"` is the `str` path to the previous pre-trained model, and `"keep_labels"` is a `list` of indices of output predictions to keep 
* `"augmentation"`: parameters for simultaneous image and label data augmentation. Must be `dict` with the following settings, where `false` turns off that setting: 
  * `"zoom"`: `false` or `list(float)` length 2, index 0 = lower bound e.g., 0.95 for zoom out by 5% and index 1 = upper bound e.g., 1.05 for zoom in by 5%
  * `"rotate"`: `false`, or `float`: maximal number of degrees to rotate by, recommend 8
  * `"translate"`: `false`, or `float`: maximal % of given dimension size to translate by, recommend not using due to convolution translation equivariance property 
  * `"brightness"`: `false`, or `float`: exponential factor to modify brightness by., recommend not using due to training instability with this
  * `"deform"`: `false` or `true`: set to `true` to turn B-spline deformations on. Note, this is slow 


## Prediction settings: `predict_config.json`

First, set the parameters in `predict_config.json`:
* `"model_path"`: path to the model to use for prediction, must be `str` or `list`
  * `str`: use a single model
  * `list`: use with staggered cascaded networks - will automatically build a tree-style execution graph from all model paths and select the correct model for each label in the output  
* `"data_path"`: root to the data path to use 
* `"dataset"`: the dataset split to use e.g., `train`, `val` and `test`
* `"post_process"`: the post-processing method to apply to the data, possible options are:
  * `false`: use no post-processing
  * `"erosion dilation"`: use erosion and dilation for post-processing
  * `path to a DAE model (str)`: use the DAE model at this path for post-processing

Then, you can use:
* `predict.py` to run prediction on a random file from the selected dataset and display the result in a scrollable window
* `performance.py` to get the average global and class-wise dice scores for all files in the dataset
