from abc import ABC

from tensorflow.keras import applications
from tensorflow.keras import layers
from tensorflow.keras import models


class SegModel(ABC):
    """ Class implementing generic and common segmentation model methods. """
    def __init__(self, input_size, output_length, kernel_size):
        self.input_size = input_size
        self.output_length = output_length
        self.kernel_size = kernel_size

    def create_model(self):
        raise NotImplementedError


class SegModel2D(SegModel, ABC):
    """ Class implementing generic and common 2D segmentation model methods. """
    def __init__(self, input_size, output_length, kernel_size, conv2d_transpose_kernel_size):
        assert len(input_size) == 2, f"The input size for 2D models must have 2 dimensions, but got {input_size}"
        super().__init__(input_size, output_length, kernel_size)
        self.conv2d_transpose_kernel_size = conv2d_transpose_kernel_size


class SegModel3D(SegModel, ABC):
    """ Class implementing generic and common 3D segmentation model methods. """
    def __init__(self, input_size, output_length, kernel_size, conv3d_transpose_kernel_size):
        assert len(input_size) == 3, f"The input size for 3D models must have 3 dimensions, but got {input_size}"
        super().__init__(input_size, output_length, kernel_size)
        self.conv3d_transpose_kernel_size = conv3d_transpose_kernel_size


class UNet3D(SegModel3D):
    """ Implementation of UNet-3D as per the 2016 paper: https://arxiv.org/pdf/1606.06650.pdf """
    def __init__(self, input_size, output_length, kernel_size=(3, 3, 3), conv3d_transpose_kernel_size=(2, 2, 2)):
        super().__init__(input_size, output_length, kernel_size, conv3d_transpose_kernel_size)

    def down_conv_block(self, m, filters_a, filters_b):
        """ 3D down-convolution block generic to all models? We'll see ... """
        m = layers.Conv3D(filters_a, self.kernel_size, padding='same', activation='relu')(m)
        m = layers.BatchNormalization()(m)

        m = layers.Conv3D(filters_b, self.kernel_size, padding='same', activation='relu')(m)
        m = layers.BatchNormalization()(m)

        return m

    def up_conv_block(self, m, prev, filters_a, filters_b):
        """ 3D up-convolution block. """
        m = layers.Conv3DTranspose(
            filters_a,
            self.conv3d_transpose_kernel_size,
            strides=(2, 2, 2),
            padding='same',
            activation='relu'
        )(m)
        m = layers.BatchNormalization()(m)

        m = layers.Concatenate()([m, prev])

        m = layers.Conv3D(filters_b, self.kernel_size, padding='same', activation='relu')(m)
        m = layers.BatchNormalization()(m)

        m = layers.Conv3D(filters_b, self.kernel_size, padding='same', activation='relu')(m)
        m = layers.BatchNormalization()(m)

        return m

    def define_architecture(self, model_input):
        """ Build the UNet-3D model. """
        # Downsampling / encoding portion
        conv0 = self.down_conv_block(model_input, 32, 64)
        pool0 = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv0)

        conv1 = self.down_conv_block(pool0, 64, 128)
        pool1 = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv1)

        conv2 = self.down_conv_block(pool1, 128, 256)
        pool2 = layers.MaxPooling3D((2, 2, 2), strides=(2, 2, 2))(conv2)

        # Middle of network
        conv3 = self.down_conv_block(pool2, 256, 512)

        # Upsampling / decoding portion
        uconv2 = self.up_conv_block(conv3, conv2, 512, 256)
        uconv1 = self.up_conv_block(uconv2, conv1, 256, 128)
        uconv0 = self.up_conv_block(uconv1, conv0, 128, 64)

        out = layers.Conv3D(self.output_length, (1, 1, 1), padding='same', activation='softmax')(uconv0)

        return out

    def create_model(self):
        """ Create a Model object which can be used for training. """
        # Input is the 3D volume size plus a dimension representing the channels
        model_input = layers.Input((self.input_size[0], self.input_size[1], self.input_size[2], 1))

        return models.Model(inputs=model_input, outputs=self.define_architecture(model_input))


class VNet(SegModel3D):
    """ Implementation of VNet as per the paper. """
    def __init__(self):
        super().__init__()


class VNetPlusPlus(SegModel3D):
    """ Experiments on my own novel architecture. """
    def __init__(self):
        super().__init__()


class UNet2D(SegModel2D):
    def __init__(self, input_size, output_length, kernel_size=(3, 3), conv2d_transpose_kernel_size=(2, 2), depth=3):
        assert depth in [3, 4], f"Only depth 3 or 4 supported for UNet2D, but got {depth}"
        super().__init__(input_size, output_length, kernel_size, conv2d_transpose_kernel_size)
        self.depth = depth

    def down_conv_block(self, m, filters):
        """ 2D down-convolution block. """
        m = layers.Conv2D(filters, self.kernel_size, padding='same', activation='relu')(m)
        m = layers.BatchNormalization()(m)

        m = layers.Conv2D(filters, self.kernel_size, padding='same', activation='relu')(m)
        m = layers.BatchNormalization()(m)

        return m

    def up_conv_block(self, m, prev, filters):
        """ 2D up-convolution block. """
        m = layers.Conv2DTranspose(
            filters,
            self.conv2d_transpose_kernel_size,
            strides=(2, 2),
            padding='same',
            activation='relu'
        )(m)
        m = layers.BatchNormalization()(m)

        m = layers.Concatenate()([m, prev])

        m = layers.Conv2D(filters, self.kernel_size, padding='same', activation='relu')(m)
        m = layers.BatchNormalization()(m)

        return m

    def define_architecture(self, model_input):
        """ Build the UNet2D model. """
        # Downsampling / encoding portion
        conv0 = self.down_conv_block(model_input, 64)
        pool0 = layers.MaxPooling2D((2, 2))(conv0)

        conv1 = self.down_conv_block(pool0, 128)
        pool1 = layers.MaxPooling2D((2, 2))(conv1)

        conv2 = self.down_conv_block(pool1, 256)
        pool2 = layers.MaxPooling2D((2, 2))(conv2)

        # Middle of network
        conv3 = self.down_conv_block(pool2, 512)

        if self.depth == 4:
            pool3 = layers.MaxPooling2D((2, 2))(conv3)

            # Middle of network
            conv4 = self.down_conv_block(pool3, 1024)

            # Upsampling / decoding portion
            uconv3 = self.up_conv_block(conv4, conv3, 512)
            uconv2 = self.up_conv_block(uconv3, conv2, 256)
        else:
            # Upsampling / decoding portion
            uconv2 = self.up_conv_block(conv3, conv2, 256)

        uconv1 = self.up_conv_block(uconv2, conv1, 128)
        uconv0 = self.up_conv_block(uconv1, conv0, 64)

        out = layers.Conv2D(self.output_length, (1, 1), padding='same', activation='softmax')(uconv0)

        return out

    def create_model(self):
        """ Create a Model object which can be used for training. """
        # Input is the 2D image size plus a dimension representing the channels
        model_input = layers.Input((self.input_size[0], self.input_size[1], 1))

        return models.Model(inputs=model_input, outputs=self.define_architecture(model_input))


class UNetPP2D(SegModel2D):
    def __init__(self, input_size, output_length, kernel_size, conv2d_transpose_kernel_size):
        super().__init__(input_size, output_length, kernel_size, conv2d_transpose_kernel_size)

    def create_model(self):
        pass


def down_conv_block(m, filter_mult, filters, kernel_size, name=None):
    m = layers.Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
    m = layers.BatchNormalization()(m)

    m = layers.Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)
    m = layers.BatchNormalization(name=name)(m)

    return m


def up_conv_block(m, prev, filter_mult, filters, kernel_size, prev_2=None, prev_3=None, prev_4=None, name=None):
    m = layers.Conv2DTranspose(filter_mult * filters, kernel_size, strides=(2, 2), padding='same', activation='relu')(m)
    m = layers.BatchNormalization()(m)

    # Concatenate layers; varies between UNet and UNet++
    if prev_4 is not None:
        m = layers.Concatenate()([m, prev, prev_2, prev_3, prev_4])
    elif prev_3 is not None:
        m = layers.Concatenate()([m, prev, prev_2, prev_3])
    elif prev_2 is not None:
        m = layers.Concatenate()([m, prev, prev_2])
    else:
        m = layers.Concatenate()([m, prev])

    m = layers.Conv2D(filter_mult * filters, kernel_size, padding='same', activation='relu')(m)

    m = layers.BatchNormalization(name=name)(m)

    return m


def build_unet(model_input, filters, kernel_size):
    # Downsampling / encoding portion
    conv0 = down_conv_block(model_input, 1, filters, kernel_size)
    pool0 = layers.MaxPooling2D((2, 2))(conv0)

    conv1 = down_conv_block(pool0, 2, filters, kernel_size)
    pool1 = layers.MaxPooling2D((2, 2))(conv1)

    conv2 = down_conv_block(pool1, 4, filters, kernel_size)
    pool2 = layers.MaxPooling2D((2, 2))(conv2)

    conv3 = down_conv_block(pool2, 8, filters, kernel_size)
    pool4 = layers.MaxPooling2D((2, 2))(conv3)

    # Middle of network
    conv4 = down_conv_block(pool4, 16, filters, kernel_size)

    # Upsampling / decoding portion
    uconv3 = up_conv_block(conv4, conv3, 8, filters, kernel_size)

    uconv2 = up_conv_block(uconv3, conv2, 4, filters, kernel_size)

    uconv1 = up_conv_block(uconv2, conv1, 2, filters, kernel_size)

    uconv0 = up_conv_block(uconv1, conv0, 1, filters, kernel_size)

    return uconv0


def build_unet_plus_plus(model_input, filters, kernel_size, l):
    # Variables names follow the UNet++ paper: [successively downsampled layers_successively upsampled layers)
    # First stage of backbone: downsampling
    conv0_0 = down_conv_block(model_input, 1, filters, kernel_size, name='conv0_0')
    pool0_0 = layers.MaxPooling2D((2, 2))(conv0_0)
    conv1_0 = down_conv_block(pool0_0, 2, filters, kernel_size, name='conv1_0')

    if l > 1:
        # Second stage
        pool1_0 = layers.MaxPooling2D((2, 2))(conv1_0)
        conv2_0 = down_conv_block(pool1_0, 4, filters, kernel_size, name='conv2_0')

        if l > 2:
            # Third stage
            pool2_0 = layers.MaxPooling2D((2, 2))(conv2_0)
            conv3_0 = down_conv_block(pool2_0, 8, filters, kernel_size, name='conv3_0')

            if l > 3:
                # Fourth stage
                pool3_0 = layers.MaxPooling2D((2, 2))(conv3_0)
                conv4_0 = down_conv_block(pool3_0, 16, filters, kernel_size, name='conv4_0')

    # First stage of upsampling and skip connections
    conv0_1 = up_conv_block(conv1_0, conv0_0, 1, filters, kernel_size, name='conv0_1')
    out = conv0_1

    if l > 1:
        # Second stage
        conv1_1 = up_conv_block(conv2_0, conv1_0, 2, filters, kernel_size, name='conv1_1')
        conv0_2 = up_conv_block(conv1_1, conv0_1, 1, filters, kernel_size, prev_2=conv0_0, name='conv0_2')
        out = conv0_2

        if l > 2:
            # Third stage
            conv2_1 = up_conv_block(conv3_0, conv2_0, 4, filters, kernel_size, name='conv2_1')
            conv1_2 = up_conv_block(conv2_1, conv1_1, 2, filters, kernel_size, prev_2=conv1_0, name='conv1_2')

            conv0_3 = up_conv_block(conv1_2, conv0_2, 1, filters, kernel_size, prev_2=conv0_1, prev_3=conv0_0,
                                    name='conv0_3')
            out = conv0_3

            if l > 3:
                # Fourth stage
                conv3_1 = up_conv_block(conv4_0, conv3_0, 8, filters, kernel_size, name='conv3_1')
                conv2_2 = up_conv_block(conv3_1, conv2_1, 4, filters, kernel_size, prev_2=conv2_0, name='conv2_2')
                conv1_3 = up_conv_block(conv2_2, conv1_2, 2, filters, kernel_size, prev_2=conv1_1, prev_3=conv1_0,
                                        name='conv1_3')
                conv0_4 = up_conv_block(conv1_3, conv0_3, 1, filters, kernel_size, prev_2=conv0_2, prev_3=conv0_1,
                                        prev_4=conv0_0, name='conv0_4')
                out = conv0_4

    return out


def create_segmentation_model(input_size, architecture='unet_plus_plus', l=3):
    """
    Create a new segmentation model.

    Parameters:
    input_size: int:
        the input size to the segmentation model in pixels. I used 512
    architecture: string:
        'unet' or 'unet_plus_plus' are both acceptable arguments. UNet++ follows the UNet++ paper:
         see more details at https://arxiv.org/pdf/1912.05074.pdf
    l:
        UNet depth; the maximal number of down-convolution and up-convolution blocks
    """
    model_input = layers.Input((input_size, input_size, 1))

    assert l in range(1, 5), f'UNet++ depth {l} not allowed. l must be in range: 1, 2, 3, 4.'

    if architecture == 'unet_plus_plus':
        model_output = build_unet_plus_plus(model_input, 32, (3, 3), l)
    elif architecture == 'unet':
        model_output = build_unet(model_input, 32, kernel_size=(3, 3))
    else:
        raise AttributeError(f'Network architecture {architecture} does not exist.')

    # Finally - the output sigmoid layer
    output_layer = layers.Conv2D(1, (1, 1), padding='same', activation='sigmoid', name='output_conv')(model_output)

    model = models.Model(inputs=model_input, outputs=output_layer)
    model.summary()

    return model


def create_classification_model(input_size, bb='DenseNet169'):
    """
    Create a classification model by loading pretrained ImageNet weights. Currently supports DenseNet backbones only
    (I found these to have the best performance).

    Parameters:
    input_size: int:
        the input size to the classification model in pixels. I used 512
    bb: string:
        the backbone to use for image classification. Acceptable arguments are DenseNet121, DenseNet169 or DenseNet201
    """

    assert bb in ['DenseNet121', 'DenseNet169', 'DenseNet201'], \
        f'Backbone {bb} not in list: DenseNet121, DenseNet169, DenseNet201'

    if bb == 'DenseNet201':
        pretrained_model = applications.DenseNet201(include_top=False,
                                                    weights='imagenet',
                                                    input_shape=(input_size, input_size, 3),
                                                    pooling=None)
    if bb == 'DenseNet169':
        pretrained_model = applications.DenseNet169(include_top=False,
                                                    weights='imagenet',
                                                    input_shape=(input_size, input_size, 3),
                                                    pooling=None)
    if bb == 'DenseNet121':
        pretrained_model = applications.DenseNet121(include_top=False,
                                                    weights='imagenet',
                                                    input_shape=(input_size, input_size, 3),
                                                    pooling=None)

    # test this - how much on/off is good here ? currently 50% of the model is trainable which is ... a lot
    for l in pretrained_model.layers:
        if not l.name.startswith('conv5_block2') or l.name.startswith('conv5_block3'):
            l.trainable = False

    model_output = layers.GlobalAveragePooling2D()(pretrained_model.output)
    model_output = layers.Dense(44, activation='relu')(model_output)
    model_output = layers.Dense(1, activation='sigmoid')(model_output)

    model = models.Model(inputs=pretrained_model.input, outputs=model_output)
    model.summary()

    return model


def unet_pp_pretrain_model(input_size, l=3):
    """
    Create a UNet++ segmentation model with a 1000-way classification head for pretraining on ImageNet. I've pretrained
    an l3 model on ImageNet, you can find a link in the readme file

    Parameters:
    input_size: int:
        the input size to the segmentation model in pixels. I used 512
    l:
        UNet depth; the maximal number of down-convolution and up-convolution blocks
    """
    model_input = layers.Input((input_size, input_size, 1))
    model_output = build_unet_plus_plus(model_input, 32, (3, 3), l)

    imagenet_head = layers.GlobalAveragePooling2D()(model_output)
    imagenet_head = layers.Dense(1000, activation='softmax')(imagenet_head)

    model = models.Model(inputs=model_input, outputs=imagenet_head)
    model.summary()

    return model
