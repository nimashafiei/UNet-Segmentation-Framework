
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Dropout, concatenate, add
)
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import RMSprop
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras import backend as K

class DilatedUNet:
    def __init__(self, input_shape=(512, 512, 1), classes=1, dilate=True, dilate_rate=1, addition=True):
        """
        Dilated U-Net Model Initialization.

        :param input_shape: Shape of the input image.
        :param classes: Number of output classes.
        :param dilate: Boolean flag to use dilated convolutions.
        :param dilate_rate: Rate for dilated convolutions.
        :param addition: Boolean flag to use addition in the bottleneck dilated layers.
        """
        self.input_shape = input_shape
        self.classes = classes
        self.dilate = dilate
        self.dilate_rate = dilate_rate
        self.addition = addition

    def build_model(self):
        """
        Build the Dilated U-Net model.
        
        :return: Compiled Keras Model.
        """
        inputs = Input(shape=self.input_shape)

        # Encoder
        down1, down1pool = self._encoder_block(inputs, 44, 0.3, self.dilate_rate)
        down2, down2pool = self._encoder_block(down1pool, 88, 0.3, self.dilate_rate)
        down3, down3pool = self._encoder_block(down2pool, 176, 0.3, self.dilate_rate)

        # Bottleneck with dilation
        bottleneck = self._dilated_bottleneck(down3pool, 176) if self.dilate else self._conv_block(down3pool, 176)

        # Decoder
        up3 = self._decoder_block(bottleneck, down3, 88)
        up2 = self._decoder_block(up3, down2, 44)
        up1 = self._decoder_block(up2, down1, 22)

        # Output layer
        outputs = Conv2D(self.classes, 1, activation='sigmoid')(up1)

        # Build the model
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=RMSprop(learning_rate=1e-4), loss=self.bce_dice_loss, metrics=[self.dice_coef])

        return model

    def _encoder_block(self, inputs, filters, dropout_rate, dilation_rate):
        """
        Encoder block for the U-Net.

        :param inputs: Input tensor.
        :param filters: Number of filters for Conv2D.
        :param dropout_rate: Dropout rate.
        :param dilation_rate: Dilation rate for Conv2D.
        :return: Output tensor and pooled tensor.
        """
        conv = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        conv = BatchNormalization()(conv)
        conv = Dropout(rate=dropout_rate)(conv)
        conv = Conv2D(filters, 3, activation='relu', padding='same', dilation_rate=dilation_rate, kernel_initializer='he_normal')(conv)
        conv = BatchNormalization()(conv)
        pool = MaxPooling2D((2, 2))(conv)
        return conv, pool

    def _decoder_block(self, inputs, skip_connection, filters):
        """
        Decoder block for the U-Net with upsampling and concatenation.

        :param inputs: Input tensor.
        :param skip_connection: Skip connection from the encoder block.
        :param filters: Number of filters for Conv2D.
        :return: Output tensor after upsampling and convolution.
        """
        upsample = UpSampling2D((2, 2))(inputs)
        upsample = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(upsample)
        concat = concatenate([skip_connection, upsample])
        concat = BatchNormalization()(concat)
        return Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(concat)

    def _dilated_bottleneck(self, inputs, filters):
        """
        Bottleneck with dilated convolutions in the U-Net.

        :param inputs: Input tensor.
        :param filters: Number of filters for Conv2D.
        :return: Output tensor after applying dilated convolutions.
        """
        dilate_layers = []
        for rate in [1, 2, 4, 8, 16, 32]:
            dilated_conv = Conv2D(filters, 3, activation='relu', padding='same', dilation_rate=rate, kernel_initializer='he_normal')(inputs)
            dilated_conv = BatchNormalization()(dilated_conv)
            dilate_layers.append(dilated_conv)

        # Combine all dilated layers (either by addition or by selecting the last one)
        if self.addition:
            return add(dilate_layers)
        else:
            return dilate_layers[-1]

    def _conv_block(self, inputs, filters):
        """
        Basic convolution block used if no dilation is applied in the bottleneck.

        :param inputs: Input tensor.
        :param filters: Number of filters for Conv2D.
        :return: Output tensor after applying convolutions.
        """
        conv = Conv2D(filters, 3, activation='relu', padding='same', kernel_initializer='he_normal')(inputs)
        return BatchNormalization()(conv)

