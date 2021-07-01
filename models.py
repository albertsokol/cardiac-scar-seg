from abc import ABC

from tensorflow.keras.utils import plot_model

import tensorflow as tf
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
        """ 3D down-convolution block. """
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

    def define_architecture(self, model_input, final_activation='softmax', out_name=None):
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

        out = layers.Conv3D(self.output_length, (1, 1, 1), padding='same', activation=None)(uconv0)
        if out_name is not None:
            out = layers.Activation(final_activation, name=out_name)(out)
        else:
            out = layers.Activation(final_activation)(out)

        return out

    def create_model(self):
        """ Create a Model object which can be used for training. """
        # Input is the 3D volume size plus a dimension representing the channels
        model_input = layers.Input((self.input_size[0], self.input_size[1], self.input_size[2], 1))

        return models.Model(inputs=model_input, outputs=self.define_architecture(model_input))


class UNet3DShallow(SegModel3D):
    """ Implementation of shallow UNet-3D as per Fahmy et al's 2019 paper: https://pubs.rsna.org/doi/10.1148/radiol.2019190737 """
    def __init__(self, input_size, output_length, kernel_size=(3, 3, 3), conv3d_transpose_kernel_size=(2, 2, 2)):
        super().__init__(input_size, output_length, kernel_size, conv3d_transpose_kernel_size)

    def down_conv_block(self, m, filters_a, filters_b):
        """ 3D down-convolution block. """
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
            strides=(2, 2, 1),
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
        pool0 = layers.MaxPooling3D((2, 2, 1), strides=(2, 2, 1))(conv0)

        conv1 = self.down_conv_block(pool0, 64, 128)
        pool1 = layers.MaxPooling3D((2, 2, 1), strides=(2, 2, 1))(conv1)

        conv2 = self.down_conv_block(pool1, 128, 256)
        pool2 = layers.MaxPooling3D((2, 2, 1), strides=(2, 2, 1))(conv2)

        # Middle of network
        conv3 = self.down_conv_block(pool2, 256, 512)

        # Upsampling / decoding portion
        uconv2 = self.up_conv_block(conv3, conv2, 512, 256)
        uconv1 = self.up_conv_block(uconv2, conv1, 256, 128)
        uconv0 = self.up_conv_block(uconv1, conv0, 128, 64)

        out = layers.Conv3D(self.output_length, (1, 1, 1), padding='same', activation=None)(uconv0)
        out = layers.Activation('softmax')(out)

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

        out = layers.Conv2D(self.output_length, (1, 1), padding='same', activation=None)(uconv0)
        out = layers.Activation('softmax')(out)

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


class CascadedUNet3D(SegModel3D):
    def __init__(self, input_size, output_length, kernel_size=(3, 3, 3), conv3d_transpose_kernel_size=(2, 2, 2)):
        super().__init__(input_size, output_length, kernel_size, conv3d_transpose_kernel_size)

    def define_architecture(self, model_input):
        unet_1 = UNet3D(self.input_size, 6, self.kernel_size, self.conv3d_transpose_kernel_size)
        unet_1_out = unet_1.define_architecture(model_input, out_name='general_out')
        unet_1_out_softmax = tf.math.argmax(unet_1_out, axis=-1, output_type=tf.dtypes.int32)
        unet_1_out_softmax = tf.expand_dims(unet_1_out_softmax, axis=-1)

        # Get masked myocardium and lumen based on the predictions which will be used by the cascaded models
        mask_lv_myo = tf.where(tf.math.equal(unet_1_out_softmax, 2), model_input, 0)
        mask_lv_lumen = tf.where(tf.math.equal(unet_1_out_softmax, 1), model_input, 0)

        # Build the cascaded models
        unet_scar = UNet3D(self.input_size, 1, self.kernel_size, self.conv3d_transpose_kernel_size)
        unet_scar_out = unet_scar.define_architecture(mask_lv_myo, 'sigmoid', out_name='scar_out')

        unet_pap = UNet3D(self.input_size, 1, self.kernel_size, self.conv3d_transpose_kernel_size)
        unet_pap_out = unet_pap.define_architecture(mask_lv_lumen, 'sigmoid', out_name='pap_out')

        return unet_1_out, unet_scar_out, unet_pap_out

    def create_model(self):
        model_input = layers.Input((*self.input_size, 1))

        return models.Model(inputs=model_input, outputs=self.define_architecture(model_input))


if __name__ == '__main__':
    c = CascadedUNet3D([160, 160, 16], 8).create_model()
    c.summary(line_length=160)
    plot_model(c, 'CascadedUNet3Dplot.png', show_shapes=True)
