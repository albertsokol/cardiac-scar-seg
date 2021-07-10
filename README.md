x

## Training settings: `train_config.json`

* `"model_save_path"`: path to save the trained models, must be `str`
* `"data_path"`: root to the MRI images/labels folder, must be `str`
* `"model"`: name of the model to train, must be `str`, possible options are
  * `"UNet2D"`
  * `"UNet3D"`
  * `"UNet3DShallow"`
  * `"CascadedUNet3D"`   
* `"plane"`: if using a 2D or shallow 3D UNet, use one of the following planes
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
  * `"cascaded weighted softmax dice"`: weighted softmax dice loss for use with end-to-end cascaded network 
* `"labels"`: dictionary of `str: str` pairs, where keys are value given to a class in the label file, and values are class names, e.g., `"1": "lv myo"`. This should represent the labels as per the manual segmentations. Must be `dict`
* `"combine_labels"`: must be a `list` of `list[str]`, where any label class names occurring in an inner list will be combined to form a single label. Used for training cascaded networks or automatic cropping models. For example, to create an automatic cropper, all heart tissue labels should be combined into one, so the correct value for this setting would be: 
```json
    [
        ["bg"],
        ["lv lumen", "lv myo", "scar", "pap", "rv lumen", "rv myo", "aorta"]
    ]
```
* `"augmentation"`: parameters for simultaneous image and label data augmentation. Must be `dict` with the following settings: 
  * `"zoom"`: 
  * `"rotate"`: 
  * `"translate"`: 
  * `"brightness"`: 
  * `"deform"`: 
