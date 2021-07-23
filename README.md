# Ventricular scar segmentation  

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
* `"augmentation"`: parameters for simultaneous image and label data augmentation. Must be `dict` with the following settings: 
  * `"zoom"`: 
  * `"rotate"`: 
  * `"translate"`: 
  * `"brightness"`: 
  * `"deform"`: 


## Prediction settings: `predict_config.json`

First, set the parameters in `predict_config.json`:
* `"model_path"`: path to the model to use for prediction
* `"data_path"`: root to the data path to use 
* `"dataset"`: the dataset split to use e.g., `train`, `val` and `test`

Then, you can use:
* `predict.py` to run prediction on a random file from the selected dataset and display the result in a scrollable window
* `performance.py` to get the average global and class-wise dice scores for all files in the dataset
