from abc import ABC

import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import models


class SegModel(ABC):
    """Class implementing generic and common segmentation model methods."""

    def __init__(
        self,
        input_size,
        output_length,
        quality_weighted_mode,
        kernel_size,
        transpose_kernel_size,
    ):
        self.input_size = input_size
        self.output_length = output_length
        self.quality_weighted_mode = quality_weighted_mode
        self.kernel_size = kernel_size
        self.transpose_kernel_size = transpose_kernel_size

    def define_architecture(self, model_input, out_name):
        raise NotImplementedError

    def create_model(self):
        """Create a Model object which can be used for training."""
        # Input is the 2D image size plus a dimension representing the channels
        model_input = layers.Input(
            (*self.input_size, 1), name="model_in", dtype=tf.float32
        )
        model = models.Model(
            inputs=model_input, outputs=self.define_architecture(model_input)
        )

        if self.quality_weighted_mode:
            in_shape = self.input_size[-1] if len(self.input_size) > 2 else 1
            qw_in = layers.Input(in_shape, name="qw_in")
            return models.Model(
                inputs=[model.input, qw_in],
                outputs=[layers.Layer(name="qw_out")(qw_in), model.output],
            )
        else:
            return model


class UNet2D(SegModel):
    def __init__(
        self,
        input_size,
        output_length,
        quality_weighted_mode,
        kernel_size=(3, 3),
        transpose_kernel_size=(2, 2),
        depth=4,
    ):
        assert depth in [
            3,
            4,
        ], f"Only depth 3 or 4 supported for UNet2D, but got {depth}"
        super().__init__(
            input_size,
            output_length,
            quality_weighted_mode,
            kernel_size,
            transpose_kernel_size,
        )
        self.depth = depth

    def down_conv_block(self, m, filters):
        """2D down-convolution block."""
        m = layers.Conv2D(filters, self.kernel_size, padding="same", activation="relu")(
            m
        )
        m = layers.BatchNormalization()(m)

        m = layers.Conv2D(filters, self.kernel_size, padding="same", activation="relu")(
            m
        )
        m = layers.BatchNormalization()(m)

        return m

    def up_conv_block(self, m, prev, filters):
        """2D up-convolution block."""
        m = layers.Conv2DTranspose(
            filters,
            self.transpose_kernel_size,
            strides=(2, 2),
            padding="same",
            activation="relu",
        )(m)
        m = layers.BatchNormalization()(m)

        m = layers.Concatenate()([m, prev])

        m = layers.Conv2D(filters, self.kernel_size, padding="same", activation="relu")(
            m
        )
        m = layers.BatchNormalization()(m)

        return m

    def define_architecture(
        self, model_input, final_activation="softmax", out_name="m"
    ):
        """Build the UNet2D model."""
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

        out = layers.Conv2D(
            self.output_length, (1, 1), padding="same", activation=None
        )(uconv0)
        out = layers.Activation(final_activation, name=out_name)(out)

        return out


class UNet2DPositional(UNet2D):
    def __init__(
        self,
        input_size,
        output_length,
        quality_weighted_mode,
        kernel_size=(3, 3),
        transpose_kernel_size=(2, 2),
        depth=4,
    ):
        super(UNet2DPositional, self).__init__(
            input_size,
            output_length,
            quality_weighted_mode,
            kernel_size,
            transpose_kernel_size,
            depth,
        )

    def define_architecture(
        self, image_input, position_input, final_activation="softmax", out_name="m"
    ):
        """Build the UNet2D model."""
        # Downsampling / encoding portion
        conv0 = self.down_conv_block(image_input, 64)
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

        # Broadcast here and concatenate the positional feature
        position_input = tf.expand_dims(position_input, axis=-1)
        position_input = tf.expand_dims(position_input, axis=-1)
        position_input = tf.broadcast_to(
            input=position_input,
            shape=(tf.shape(position_input)[0], *self.input_size, 1),
        )
        uconv0 = tf.concat([uconv0, position_input], axis=-1, name="concat_position")

        out = layers.Conv2D(
            self.output_length, (1, 1), padding="same", activation=None
        )(uconv0)
        out = layers.Activation(final_activation, name=out_name)(out)

        return out

    def create_model(self):
        """Create a Model object which can be used for training."""
        # Input is the 2D image size plus a dimension representing the channels
        image_input = layers.Input((*self.input_size, 1), name="model_in")
        position_input = layers.Input((1,), name="position_in")
        model = models.Model(
            inputs=[image_input, position_input],
            outputs=self.define_architecture(image_input, position_input),
        )

        # if self.quality_weighted_mode:
        #     in_shape = self.input_size[-1] if len(self.input_size) > 2 else 1
        #     qw_in = layers.Input(in_shape, name='qw_in')
        #     return models.Model(inputs=[model.input, qw_in], outputs=[layers.Layer(name='qw_out')(qw_in), model.output])
        # else:
        return model


class UNet3D(SegModel):
    """Implementation of UNet-3D as per the 2016 paper: https://arxiv.org/pdf/1606.06650.pdf"""

    def __init__(
        self,
        input_size,
        output_length,
        quality_weighted_mode,
        kernel_size=(3, 3, 3),
        transpose_kernel_size=(2, 2, 2),
    ):
        super().__init__(
            input_size,
            output_length,
            quality_weighted_mode,
            kernel_size,
            transpose_kernel_size,
        )

    def down_conv_block(self, m, filters_a, filters_b):
        """3D down-convolution block."""
        m = layers.Conv3D(
            filters_a, self.kernel_size, padding="same", activation="relu"
        )(m)
        m = layers.BatchNormalization()(m)

        m = layers.Conv3D(
            filters_b, self.kernel_size, padding="same", activation="relu"
        )(m)
        m = layers.BatchNormalization()(m)

        return m

    def up_conv_block(self, m, prev, filters_a, filters_b):
        """3D up-convolution block."""
        m = layers.Conv3DTranspose(
            filters_a,
            self.transpose_kernel_size,
            strides=(2, 2, 2),
            padding="same",
            activation="relu",
        )(m)
        m = layers.BatchNormalization()(m)

        m = layers.Concatenate()([m, prev])

        m = layers.Conv3D(
            filters_b, self.kernel_size, padding="same", activation="relu"
        )(m)
        m = layers.BatchNormalization()(m)

        m = layers.Conv3D(
            filters_b, self.kernel_size, padding="same", activation="relu"
        )(m)
        m = layers.BatchNormalization()(m)

        return m

    def define_architecture(
        self, model_input, final_activation="softmax", out_name="m"
    ):
        """Build the UNet-3D model."""
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

        out = layers.Conv3D(
            self.output_length, (1, 1, 1), padding="same", activation=None
        )(uconv0)
        out = layers.Activation(final_activation, name=out_name)(out)

        return out


class UNet3DFrozenDepth(SegModel):
    """Implementation of UNet-3D, keeping the dimensionality of the depth axis consistent with the input."""

    def __init__(
        self,
        input_size,
        output_length,
        quality_weighted_mode,
        kernel_size=(3, 3, 3),
        transpose_kernel_size=(2, 2, 2),
    ):
        super().__init__(
            input_size,
            output_length,
            quality_weighted_mode,
            kernel_size,
            transpose_kernel_size,
        )

    def down_conv_block(self, m, filters_a, filters_b):
        """3D down-convolution block."""
        m = layers.Conv3D(
            filters_a, self.kernel_size, padding="same", activation="relu"
        )(m)
        m = layers.BatchNormalization()(m)

        m = layers.Conv3D(
            filters_b, self.kernel_size, padding="same", activation="relu"
        )(m)
        m = layers.BatchNormalization()(m)

        return m

    def up_conv_block(self, m, prev, filters_a, filters_b):
        """3D up-convolution block."""
        m = layers.Conv3DTranspose(
            filters_a,
            self.transpose_kernel_size,
            strides=(2, 2, 1),
            padding="same",
            activation="relu",
        )(m)
        m = layers.BatchNormalization()(m)

        m = layers.Concatenate()([m, prev])

        m = layers.Conv3D(
            filters_b, self.kernel_size, padding="same", activation="relu"
        )(m)
        m = layers.BatchNormalization()(m)

        m = layers.Conv3D(
            filters_b, self.kernel_size, padding="same", activation="relu"
        )(m)
        m = layers.BatchNormalization()(m)

        return m

    def define_architecture(
        self, model_input, final_activation="softmax", out_name="m"
    ):
        """Build the UNet-3D model."""
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

        out = layers.Conv3D(
            self.output_length, (1, 1, 1), padding="same", activation=None
        )(uconv0)
        out = layers.Activation(final_activation, name=out_name)(out)

        return out

    def create_model(self):
        """Create a Model object which can be used for training."""
        # Input is the 2D image size plus None for the depth and then a 1-dimension representing the channels
        model_input = layers.Input((*self.input_size[:2], None, 1), name="model_in")
        model = models.Model(
            inputs=model_input, outputs=self.define_architecture(model_input)
        )

        if self.quality_weighted_mode:
            qw_in = layers.Input(shape=(None,), name="qw_in")
            return models.Model(
                inputs=[model.input, qw_in],
                outputs=[layers.Layer(name="qw_out")(qw_in), model.output],
            )
        else:
            return model


class UNet3DShallow(SegModel):
    """Implementation of shallow UNet-3D as per Fahmy et al's 2019 paper: https://pubs.rsna.org/doi/10.1148/radiol.2019190737"""

    def __init__(
        self,
        input_size,
        output_length,
        quality_weighted_mode,
        kernel_size=(3, 3, 3),
        transpose_kernel_size=(2, 2, 2),
    ):
        super().__init__(
            input_size,
            output_length,
            quality_weighted_mode,
            kernel_size,
            transpose_kernel_size,
        )

    def down_conv_block(self, m, filters_a, filters_b):
        """3D down-convolution block."""
        m = layers.Conv3D(
            filters_a, self.kernel_size, padding="same", activation="relu"
        )(m)
        m = layers.BatchNormalization()(m)

        m = layers.Conv3D(
            filters_b, self.kernel_size, padding="same", activation="relu"
        )(m)
        m = layers.BatchNormalization()(m)

        return m

    def up_conv_block(self, m, prev, filters_a, filters_b):
        """3D up-convolution block."""
        m = layers.Conv3DTranspose(
            filters_a,
            self.transpose_kernel_size,
            strides=(2, 2, 1),
            padding="same",
            activation="relu",
        )(m)
        m = layers.BatchNormalization()(m)

        m = layers.Concatenate()([m, prev])

        m = layers.Conv3D(
            filters_b, self.kernel_size, padding="same", activation="relu"
        )(m)
        m = layers.BatchNormalization()(m)

        m = layers.Conv3D(
            filters_b, self.kernel_size, padding="same", activation="relu"
        )(m)
        m = layers.BatchNormalization()(m)

        return m

    def define_architecture(
        self, model_input, final_activation="softmax", out_name="m"
    ):
        """Build the UNet-3D model."""
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

        out = layers.Conv3D(
            self.output_length, (1, 1, 1), padding="same", activation=None
        )(uconv0)
        out = layers.Activation(final_activation, name=out_name)(out)

        return out


class VNet(SegModel):
    """Implementation of V-Net as per the 2016 paper: https://arxiv.org/pdf/1606.04797.pdf"""

    def __init__(
        self,
        input_size,
        output_length,
        quality_weighted_mode,
        kernel_size=(5, 5, 5),
        transpose_kernel_size=(2, 2, 2),
    ):
        super().__init__(
            input_size,
            output_length,
            quality_weighted_mode,
            kernel_size,
            transpose_kernel_size,
        )

    def side_conv_block(self, m_in, filters, length=1, add=True):
        """3D residual sideways convolution block."""
        m = layers.Conv3D(filters, self.kernel_size, padding="same")(m_in)
        m = layers.PReLU()(m)
        m = layers.BatchNormalization()(m)

        if length >= 2:
            m = layers.Conv3D(filters, self.kernel_size, padding="same")(m)
            m = layers.PReLU()(m)
            m = layers.BatchNormalization()(m)

        if length >= 3:
            m = layers.Conv3D(filters, self.kernel_size, padding="same")(m)
            m = layers.PReLU()(m)
            m = layers.BatchNormalization()(m)

        if add:
            return m + m_in
        else:
            return m

    def up_conv_block(self, m_in, prev, filters, length=1, strides=(2, 2, 2)):
        """3D up-convolution block."""
        m_up = layers.Conv3DTranspose(
            filters,
            self.transpose_kernel_size,
            strides=strides,
            padding="same",
        )(m_in)
        m = layers.PReLU()(m_up)
        m = layers.BatchNormalization()(m)

        m = layers.Concatenate()([m, prev])
        m = self.side_conv_block(m, filters, length, add=False)

        return m + m_up

    def define_architecture(
        self, model_input, final_activation="softmax", out_name="m"
    ):
        """Build the V-Net model."""
        # Downsampling / encoding portion
        conv0 = self.side_conv_block(model_input, 16, length=1)
        down0 = layers.Conv3D(32, (2, 2, 2), strides=(2, 2, 2), padding="same")(conv0)
        down0 = layers.PReLU()(down0)

        conv1 = self.side_conv_block(down0, 32, length=2)
        down1 = layers.Conv3D(64, (2, 2, 2), strides=(2, 2, 2), padding="same")(conv1)
        down1 = layers.PReLU()(down1)

        conv2 = self.side_conv_block(down1, 64, length=3)
        down2 = layers.Conv3D(128, (2, 2, 2), strides=(2, 2, 2), padding="same")(conv2)
        down2 = layers.PReLU()(down2)

        conv3 = self.side_conv_block(down2, 128, length=3)
        down3 = layers.Conv3D(256, (2, 2, 2), strides=(2, 2, 2), padding="same")(conv3)
        down3 = layers.PReLU()(down3)

        # Middle of network
        conv4 = self.side_conv_block(down3, 256, length=3)

        # Upsampling / decoding portion
        uconv3 = self.up_conv_block(conv4, conv3, 256, length=3)
        uconv2 = self.up_conv_block(uconv3, conv2, 128, length=3)
        uconv1 = self.up_conv_block(uconv2, conv1, 64, length=2)
        uconv0 = self.up_conv_block(uconv1, conv0, 32, length=1)

        out = layers.Conv3D(
            self.output_length, (1, 1, 1), padding="same", activation=None
        )(uconv0)
        out = layers.Activation(final_activation, name=out_name)(out)

        return out


class VNetShallow(VNet):
    """Adjusted V-Net to work in the shallow setting."""

    def __init__(
        self,
        input_size,
        output_length,
        quality_weighted_mode,
        kernel_size=(5, 5, 5),
        transpose_kernel_size=(2, 2, 2),
    ):
        super().__init__(
            input_size,
            output_length,
            quality_weighted_mode,
            kernel_size,
            transpose_kernel_size,
        )

    def define_architecture(
        self, model_input, final_activation="softmax", out_name="m"
    ):
        """Build the V-Net model."""
        # Downsampling / encoding portion
        conv0 = self.side_conv_block(model_input, 16, length=1)
        down0 = layers.Conv3D(32, (2, 2, 2), strides=(2, 2, 1), padding="same")(conv0)
        down0 = layers.PReLU()(down0)

        conv1 = self.side_conv_block(down0, 32, length=2)
        down1 = layers.Conv3D(64, (2, 2, 2), strides=(2, 2, 1), padding="same")(conv1)
        down1 = layers.PReLU()(down1)

        conv2 = self.side_conv_block(down1, 64, length=3)
        down2 = layers.Conv3D(128, (2, 2, 2), strides=(2, 2, 1), padding="same")(conv2)
        down2 = layers.PReLU()(down2)

        conv3 = self.side_conv_block(down2, 128, length=3)
        down3 = layers.Conv3D(256, (2, 2, 2), strides=(2, 2, 1), padding="same")(conv3)
        down3 = layers.PReLU()(down3)

        # Middle of network
        conv4 = self.side_conv_block(down3, 256, length=3)

        # Upsampling / decoding portion
        uconv3 = self.up_conv_block(conv4, conv3, 256, length=3, strides=(2, 2, 1))
        uconv2 = self.up_conv_block(uconv3, conv2, 128, length=3, strides=(2, 2, 1))
        uconv1 = self.up_conv_block(uconv2, conv1, 64, length=2, strides=(2, 2, 1))
        uconv0 = self.up_conv_block(uconv1, conv0, 32, length=1, strides=(2, 2, 1))

        out = layers.Conv3D(
            self.output_length, (1, 1, 1), padding="same", activation=None
        )(uconv0)
        out = layers.Activation(final_activation, name=out_name)(out)

        return out


class DenoisingUNet(UNet2D):
    """UNet2D-based denoising auto-encoder model."""

    def __init__(
        self,
        input_size,
        output_length,
        quality_weighted_mode=False,
        kernel_size=(3, 3),
        transpose_kernel_size=(2, 2),
    ):
        super().__init__(
            input_size,
            output_length,
            quality_weighted_mode,
            kernel_size,
            transpose_kernel_size,
        )

    def create_model(self):
        """Create a Model object which can be used for training."""
        # Input is the 2D image size plus a dimension representing the 8 different labels
        model_input = layers.Input(
            (*self.input_size, self.output_length), name="model_in"
        )
        return models.Model(
            inputs=model_input, outputs=self.define_architecture(model_input)
        )
