import tensorflow as tf
from tensorflow.keras import layers as L

class UNetPlusPlus:
    def __init__(self, input_shape, num_classes=1):
        """
        UNet+++ Model Initialization.
        
        :param input_shape: Shape of the input image.
        :param num_classes: Number of output classes.
        """
        self.input_shape = input_shape
        self.num_classes = num_classes

    def build_model(self):
        """ Build the UNet+++ model """
        inputs = L.Input(self.input_shape, name="input_layer")

        """ Encoder Path """
        e1, p1 = self._encoder_block(inputs, 64)
        e2, p2 = self._encoder_block(p1, 128)
        e3, p3 = self._encoder_block(p2, 256)
        e4, p4 = self._encoder_block(p3, 512)

        """ Bottleneck """
        e5 = self._conv_block(p4, 1024)
        e5 = self._conv_block(e5, 1024)

        """ Decoder Path """
        d4 = self._decoder_block(e5, [e1, e2, e3, e4], [8, 4, 2, 1], 64*5)
        d3 = self._decoder_block(e5, [e1, e2, e3, d4], [4, 2, 1, 0], 64*5)
        d2 = self._decoder_block(e5, [e1, e2, d3, d4], [2, 1, 0, 0], 64*5)
        d1 = self._decoder_block(e5, [e1, d2, d3, d4], [1, 0, 0, 0], 64*5)

        """ Output """
        outputs = self._output_block(d1)

        model = tf.keras.Model(inputs, outputs, name="UNet+++")
        return model

    def _conv_block(self, x, num_filters, act=True):
        """
        Convolution block with BatchNormalization and ReLU activation.
        
        :param x: Input tensor.
        :param num_filters: Number of filters.
        :param act: Whether to apply activation after convolution.
        :return: Output tensor after convolution block.
        """
        x = L.Conv2D(num_filters, kernel_size=3, padding="same")(x)

        if act:
            x = L.BatchNormalization()(x)
            x = L.Activation("relu")(x)

        return x

    def _encoder_block(self, x, num_filters):
        """
        Encoder block consisting of two convolution blocks and max pooling.
        
        :param x: Input tensor.
        :param num_filters: Number of filters.
        :return: Tuple of (encoded output, pooled output).
        """
        x = self._conv_block(x, num_filters)
        x = self._conv_block(x, num_filters)
        p = L.MaxPool2D((2, 2))(x)
        return x, p

    def _decoder_block(self, bottleneck, encoders, pool_factors, num_filters):
        """
        Decoder block for upsampling and concatenation of skip connections.
        
        :param bottleneck: The bottleneck layer to upsample.
        :param encoders: List of encoder layers.
        :param pool_factors: List of pooling factors for the encoder layers.
        :param num_filters: Number of filters for the output.
        :return: Decoded output tensor.
        """
        upsampled = L.UpSampling2D((2 ** max(pool_factors), 2 ** max(pool_factors)), interpolation="bilinear")(bottleneck)
        conv_layers = [self._conv_block(encoder, 64) if factor == 0 else self._pooled_conv(encoder, factor) 
                       for encoder, factor in zip(encoders, pool_factors)]
        concatenated = L.Concatenate()(conv_layers + [upsampled])
        return self._conv_block(concatenated, num_filters)

    def _pooled_conv(self, x, pool_factor):
        """
        Apply MaxPooling and a Conv block to match resolution.
        
        :param x: Input tensor.
        :param pool_factor: Pooling factor (how much to downsample).
        :return: Output tensor after pooling and convolution.
        """
        pooled = L.MaxPool2D((2 ** pool_factor, 2 ** pool_factor))(x)
        return self._conv_block(pooled, 64)

    def _output_block(self, x):
        """
        Output block for the final segmentation map.
        
        :param x: Input tensor.
        :return: Output tensor (segmentation map).
        """
        y = L.Conv2D(self.num_classes, kernel_size=3, padding="same")(x)
        y = L.Activation("sigmoid")(y)
        return y

